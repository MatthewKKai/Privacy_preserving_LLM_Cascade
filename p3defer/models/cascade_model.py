"""
Cascade Model for P3Defer.

Wraps the local LLM (M_l) and server LLM (M_s) with:
- Privacy detection head (binary classifier on top of local model)
- Quality estimation via generation confidence (transition scores)
- Knowledge distillation from server to local model
- Multi-objective loss combining task loss, privacy loss, and distillation loss

Implements the architecture described in Section 2.1 and 2.3 of the paper.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

logger = logging.getLogger(__name__)


class PrivacyHead(nn.Module):
    """Binary classification head for privacy detection.

    Attached on top of the local model's hidden states to predict
    whether a query contains personally identifiable information.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Classify privacy from the last hidden state.

        Args:
            hidden_states: [batch, seq_len, hidden_size] from the local model.

        Returns:
            [batch, 2] privacy logits.
        """
        # Use the last non-padding token's hidden state
        pooled = hidden_states[:, -1, :]
        return self.classifier(pooled)


class CascadeModel(nn.Module):
    """Full cascade model combining local LLM, server LLM, and privacy head.

    The model supports three modes:
    1. Local inference: Generate with the local model only
    2. Server inference: Forward (masked) query to server model
    3. Training: Multi-objective loss with distillation

    The multi-objective loss (Eq. 9-10 in the paper):
        L = L_task + alpha * L_privacy + beta * L_KD
    where:
        L_task = cross-entropy loss for text generation
        L_privacy = cross-entropy loss for privacy classification
        L_KD = KL divergence between server and local distributions
    """

    def __init__(
        self,
        local_model_name: str = "google/gemma-2-2b-it",
        server_model_name: str = "google/gemma-2-9b-it",
        load_server: bool = True,
        alpha: float = 0.4,
        beta: float = 0.1,
        device: Optional[str] = None,
    ):
        """Initialize the cascade model.

        Args:
            local_model_name: HuggingFace model name for the local LLM.
            server_model_name: HuggingFace model name for the server LLM.
            load_server: Whether to load the server model (False for eval-only).
            alpha: Weight for privacy loss.
            beta: Weight for distillation loss.
            device: Device to load models on.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading local model: %s", local_model_name)
        self.local_model = transformers.AutoModelForCausalLM.from_pretrained(
            local_model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            output_hidden_states=True,
            trust_remote_code=True,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            local_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # Privacy detection head
        hidden_size = self.local_model.config.hidden_size
        self.privacy_head = PrivacyHead(hidden_size)

        # Server model (loaded lazily or not at all for local-only eval)
        self.server_model = None
        self.server_tokenizer = None
        if load_server:
            logger.info("Loading server model: %s", server_model_name)
            self.server_model = transformers.AutoModelForCausalLM.from_pretrained(
                server_model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                output_hidden_states=True,
                trust_remote_code=True,
            )
            self.server_tokenizer = transformers.AutoTokenizer.from_pretrained(
                server_model_name, trust_remote_code=True
            )
            if self.server_tokenizer.pad_token is None:
                self.server_tokenizer.pad_token = self.server_tokenizer.eos_token
            # Freeze server model
            for param in self.server_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        privacy_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-objective loss computation.

        Args:
            input_ids: [batch, seq_len] input token IDs.
            attention_mask: [batch, seq_len] attention mask.
            labels: [batch, seq_len] target token IDs for language modeling.
            privacy_labels: [batch] binary privacy labels.

        Returns:
            Dictionary with loss, logits, privacy_logits, and hidden_states.
        """
        # Local model forward
        local_outputs = self.local_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        hidden_states = local_outputs.hidden_states[-1]
        privacy_logits = self.privacy_head(hidden_states)

        result = {
            "logits": local_outputs.logits,
            "privacy_logits": privacy_logits,
            "hidden_states": hidden_states,
        }

        # Compute losses if labels are provided
        if labels is not None:
            task_loss = local_outputs.loss

            # Privacy classification loss
            privacy_loss = torch.tensor(0.0, device=input_ids.device)
            if privacy_labels is not None:
                privacy_loss = F.cross_entropy(privacy_logits, privacy_labels)

            # Knowledge distillation loss
            kd_loss = torch.tensor(0.0, device=input_ids.device)
            if self.server_model is not None and self.beta > 0:
                with torch.no_grad():
                    server_outputs = self.server_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                # KL divergence between server and local distributions
                local_log_probs = F.log_softmax(local_outputs.logits, dim=-1)
                server_probs = F.softmax(server_outputs.logits, dim=-1)
                kd_loss = F.kl_div(
                    local_log_probs, server_probs, reduction="batchmean"
                )

            # Combined multi-objective loss (Eq. 9)
            total_loss = task_loss + self.alpha * privacy_loss + self.beta * kd_loss

            result["loss"] = total_loss
            result["task_loss"] = task_loss
            result["privacy_loss"] = privacy_loss
            result["kd_loss"] = kd_loss

        return result

    def generate_local(
        self,
        input_text: str,
        max_new_tokens: int = 256,
        return_confidence: bool = True,
    ) -> Dict[str, object]:
        """Generate a response using the local model.

        Args:
            input_text: Input prompt text.
            max_new_tokens: Maximum number of new tokens to generate.
            return_confidence: Whether to compute confidence scores.

        Returns:
            Dictionary with generated_text, confidence, and privacy_prob.
        """
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        gen_config = transformers.GenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

        with torch.no_grad():
            outputs = self.local_model.generate(
                **inputs, generation_config=gen_config
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[:, prompt_len:]
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        result = {"generated_text": generated_text}

        if return_confidence and outputs.scores:
            # Compute mean transition probability as confidence
            transition_scores = self.local_model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            import numpy as np
            raw_probs = np.exp(transition_scores.cpu().numpy())
            result["confidence"] = float(np.mean(raw_probs))
            result["median_confidence"] = float(np.median(raw_probs))

        # Privacy detection
        with torch.no_grad():
            local_out = self.local_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            hidden = local_out.hidden_states[-1]
            priv_logits = self.privacy_head(hidden)
            priv_probs = F.softmax(priv_logits, dim=-1)
            result["privacy_prob"] = float(priv_probs[0, 1].cpu())

        return result

    def generate_server(
        self,
        input_text: str,
        max_new_tokens: int = 256,
    ) -> Dict[str, object]:
        """Generate a response using the server model.

        Args:
            input_text: Input prompt text (should be privacy-masked).
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Dictionary with generated_text.
        """
        if self.server_model is None:
            raise RuntimeError("Server model not loaded. Initialize with load_server=True.")

        tokenizer = self.server_tokenizer or self.tokenizer
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        with torch.no_grad():
            outputs = self.server_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, prompt_len:]
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {"generated_text": generated_text}

    def get_privacy_signal(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get the privacy detection signal for the state encoder.

        Args:
            input_ids: [batch, seq_len] input token IDs.
            attention_mask: [batch, seq_len] attention mask.

        Returns:
            [batch, 2] privacy logits.
        """
        with torch.no_grad():
            outputs = self.local_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            hidden = outputs.hidden_states[-1]
            return self.privacy_head(hidden)

    def get_quality_signal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get the quality/confidence signal for the state encoder.

        Computes the mean log-probability of the generated tokens as
        a proxy for answer quality confidence.

        Args:
            input_ids: [batch, seq_len] input token IDs.
            attention_mask: [batch, seq_len] attention mask.

        Returns:
            [batch, 1] confidence score.
        """
        with torch.no_grad():
            outputs = self.local_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = outputs.logits
            # Compute mean log-probability of the input tokens as confidence
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            token_log_probs = log_probs.gather(
                2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            # Mask padding
            mask = attention_mask[:, 1:].float()
            mean_log_prob = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            # Convert to confidence in [0, 1]
            confidence = torch.exp(mean_log_prob).unsqueeze(-1)
            return confidence
