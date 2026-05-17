"""
Multi-Objective Loss Tuning for P3Defer.

Implements the joint training objective (Section 2.3, Eq. 9-10) that combines:
- Task loss (L_task): Cross-entropy for text generation
- Privacy loss (L_privacy): Binary classification for privacy detection
- Distillation loss (L_KD): KL divergence from server to local model
- Deferral-aware weighting: Heaviside step function H(.) based on deferral decision

The combined loss:
    L = L_task + alpha * L_privacy + beta * H(confidence < threshold) * L_KD

This is a portable replacement for the Google-internal loss_tuning.py.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers

logger = logging.getLogger(__name__)


class MultiTaskDataset(Dataset):
    """Dataset for multi-objective training."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        target_text = item["output"]
        privacy_label = item.get("privacy", 0)

        full_text = input_text + target_text + self.tokenizer.eos_token
        encoded = self.tokenizer(
            full_text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        input_only = self.tokenizer(input_text, truncation=True, max_length=self.max_length)
        input_len = len(input_only["input_ids"])
        labels[:input_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "privacy_label": torch.tensor(privacy_label, dtype=torch.long),
        }


class MultiObjectiveModel(nn.Module):
    """Multi-objective model with task head, privacy head, and distillation.

    Wraps a local causal LM with:
    1. A privacy classification head
    2. Optional distillation from a server model
    """

    def __init__(
        self,
        local_model: nn.Module,
        server_model: Optional[nn.Module] = None,
        hidden_size: int = 768,
        alpha: float = 0.4,
        beta: float = 0.1,
        confidence_threshold: float = 0.6,
    ):
        super().__init__()
        self.local_model = local_model
        self.server_model = server_model
        self.alpha = alpha
        self.beta = beta
        self.confidence_threshold = confidence_threshold

        # Privacy classification head
        self.privacy_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )

        # Freeze server model if provided
        if self.server_model is not None:
            for param in self.server_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None, privacy_label=None):
        # Local model forward
        local_outputs = self.local_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        hidden_states = local_outputs.hidden_states[-1]
        privacy_logits = self.privacy_head(hidden_states[:, -1, :])

        result = {
            "logits": local_outputs.logits,
            "privacy_logits": privacy_logits,
        }

        if labels is not None:
            # Task loss
            task_loss = local_outputs.loss

            # Privacy loss
            privacy_loss = torch.tensor(0.0, device=input_ids.device)
            if privacy_label is not None:
                privacy_loss = F.cross_entropy(privacy_logits, privacy_label)

            # Distillation loss with Heaviside step function
            kd_loss = torch.tensor(0.0, device=input_ids.device)
            if self.server_model is not None and self.beta > 0:
                with torch.no_grad():
                    server_outputs = self.server_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                # Compute confidence from local model logits
                local_probs = F.softmax(local_outputs.logits, dim=-1)
                confidence = local_probs.max(dim=-1).values.mean()

                # Heaviside step function: H(confidence < threshold)
                # Only apply distillation when confidence is low
                heaviside = 1.0 if confidence < self.confidence_threshold else 0.0

                if heaviside > 0:
                    local_log_probs = F.log_softmax(local_outputs.logits, dim=-1)
                    server_probs = F.softmax(server_outputs.logits, dim=-1)
                    kd_loss = F.kl_div(
                        local_log_probs, server_probs, reduction="batchmean"
                    )

            # Combined loss (Eq. 9-10)
            total_loss = task_loss + self.alpha * privacy_loss + self.beta * kd_loss

            result["loss"] = total_loss
            result["task_loss"] = task_loss
            result["privacy_loss"] = privacy_loss
            result["kd_loss"] = kd_loss

        return result


class MultiObjectiveLossTuner:
    """Trainer for multi-objective loss tuning."""

    def __init__(
        self,
        local_model_name: str = "google/gemma-2-2b-it",
        server_model_name: Optional[str] = "google/gemma-2-9b-it",
        alpha: float = 0.4,
        beta: float = 0.1,
        confidence_threshold: float = 0.6,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        max_length: int = 512,
        output_dir: str = "./output/loss_tuning",
        device: str = "auto",
    ):
        self.local_model_name = local_model_name
        self.server_model_name = server_model_name
        self.alpha = alpha
        self.beta = beta
        self.confidence_threshold = confidence_threshold
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_dir = output_dir
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """Run multi-objective loss tuning.

        Args:
            train_data: List of {input, output, privacy} training examples.
            eval_data: Optional evaluation examples.

        Returns:
            Dictionary with training metrics.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.local_model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load models
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.info("Loading local model: %s", self.local_model_name)
        local_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.local_model_name, torch_dtype=dtype, trust_remote_code=True,
            output_hidden_states=True,
        )
        hidden_size = local_model.config.hidden_size

        server_model = None
        if self.server_model_name:
            logger.info("Loading server model: %s", self.server_model_name)
            server_model = transformers.AutoModelForCausalLM.from_pretrained(
                self.server_model_name, torch_dtype=dtype, trust_remote_code=True,
            )

        # Create multi-objective model
        model = MultiObjectiveModel(
            local_model=local_model,
            server_model=server_model,
            hidden_size=hidden_size,
            alpha=self.alpha,
            beta=self.beta,
            confidence_threshold=self.confidence_threshold,
        )
        model = model.to(self.device)

        # Create datasets
        train_dataset = MultiTaskDataset(train_data, tokenizer, self.max_length)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Optimizer (only train local model and privacy head)
        trainable_params = [
            p for p in model.local_model.parameters() if p.requires_grad
        ] + list(model.privacy_head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

        # Training loop
        logger.info("Starting multi-objective loss tuning...")
        metrics_history = []

        for epoch in range(self.num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_task_loss = 0.0
            epoch_privacy_loss = 0.0
            epoch_kd_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                privacy_labels = batch["privacy_label"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    privacy_label=privacy_labels,
                )

                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                epoch_task_loss += outputs["task_loss"].item()
                epoch_privacy_loss += outputs["privacy_loss"].item()
                epoch_kd_loss += outputs["kd_loss"].item()
                num_batches += 1

            avg_metrics = {
                "epoch": epoch + 1,
                "loss": epoch_loss / num_batches,
                "task_loss": epoch_task_loss / num_batches,
                "privacy_loss": epoch_privacy_loss / num_batches,
                "kd_loss": epoch_kd_loss / num_batches,
            }
            metrics_history.append(avg_metrics)
            logger.info("Epoch %d: %s", epoch + 1, avg_metrics)

        # Save
        model.local_model.save_pretrained(os.path.join(self.output_dir, "local_model"))
        tokenizer.save_pretrained(os.path.join(self.output_dir, "local_model"))
        torch.save(
            model.privacy_head.state_dict(),
            os.path.join(self.output_dir, "privacy_head.pt"),
        )

        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_history, f, indent=2)

        return metrics_history[-1] if metrics_history else {}
