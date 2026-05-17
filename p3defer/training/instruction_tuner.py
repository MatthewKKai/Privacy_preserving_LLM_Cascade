"""
CoT-enhanced Instruction Tuning for P3Defer.

Implements the instruction tuning stage (Section 2.3) that fine-tunes the
local LLM with Chain-of-Thought (CoT) enhanced prompts. The tuning teaches
the model to:
1. Detect privacy-sensitive content in queries
2. Solve the task with step-by-step reasoning
3. Self-critique its answer quality
4. Decide whether the query needs rewriting/deferral

This is a portable replacement for the Google-internal instruction_tuning.py.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """Dataset for instruction tuning with CoT prompts."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        target_text = item["output"]

        # Concatenate input and output for causal LM training
        full_text = input_text + target_text + self.tokenizer.eos_token

        encoded = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels: mask the input portion
        labels = input_ids.clone()
        input_only = self.tokenizer(
            input_text, truncation=True, max_length=self.max_length
        )
        input_len = len(input_only["input_ids"])
        labels[:input_len] = -100  # Mask input tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InstructionTuner:
    """Instruction tuner for the local LLM with LoRA.

    Fine-tunes the local model using CoT-enhanced instruction prompts
    with LoRA for parameter efficiency.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        max_length: int = 512,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        output_dir: str = "./output/instruction_tuning",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.device = device

    def prepare_cot_data(
        self,
        raw_data: List[Dict],
        dataset_name: str = "gsm8k",
    ) -> List[Dict]:
        """Prepare CoT-enhanced instruction tuning data.

        Converts raw dataset examples into instruction-tuning format with
        Chain-of-Thought reasoning steps.

        Args:
            raw_data: List of dataset examples with question, answer, privacy fields.
            dataset_name: Name of the dataset for prompt selection.

        Returns:
            List of {input, output} dictionaries.
        """
        tuning_data = []

        for item in raw_data:
            question = item.get("question", "")
            answer = item.get("answer", "")
            has_privacy = item.get("privacy", 0) == 1
            privacy_label = "Yes" if has_privacy else "No"

            if dataset_name == "gsm8k":
                input_text = (
                    "Assume you're a student working on mathematical problems. "
                    "You need to do four tasks:\n"
                    "a. Check if the question contains personal information, "
                    "output Yes or No;\n"
                    "b. Solve the question step by step;\n"
                    "c. Self-critique your answer with a confidence level "
                    "(low, moderate, high);\n"
                    "d. Decide if the question needs to be rewritten for privacy. "
                    "If it contains personal information and confidence is not high, "
                    "output Yes, else No.\n\n"
                    f"Question: {question}\n"
                    "Output:\n"
                    "Let's think step by step:\n"
                )
                output_text = (
                    f"a. Contains Personal Information: {privacy_label}\n"
                    f"b. Answer: {answer}\n"
                    "c. Confidence Level: High\n"
                    f"d. Rewritten question: {'Yes' if has_privacy else 'No'}\n"
                )
            elif dataset_name in ("medsum", "emailsum"):
                task_desc = "medical question" if dataset_name == "medsum" else "email thread"
                input_text = (
                    f"You will be given a {task_desc}. You need to:\n"
                    "a. Check if it contains personal information, output Yes or No;\n"
                    "b. Provide a concise summary;\n"
                    "c. Rate your confidence (low, moderate, high);\n"
                    "d. Decide if the content needs privacy rewriting.\n\n"
                    f"Content: {question}\n"
                    "Output:\n"
                )
                output_text = (
                    f"a. Contains Personal Information: {privacy_label}\n"
                    f"b. Summary: {answer}\n"
                    "c. Confidence Level: High\n"
                    f"d. Rewritten question: {'Yes' if has_privacy else 'No'}\n"
                )
            else:
                input_text = f"Question: {question}\nAnswer:\n"
                output_text = answer

            tuning_data.append({"input": input_text, "output": output_text})

        logger.info("Prepared %d CoT instruction tuning examples", len(tuning_data))
        return tuning_data

    def train(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """Run instruction tuning with LoRA.

        Args:
            train_data: List of {input, output} training examples.
            eval_data: Optional list of {input, output} evaluation examples.

        Returns:
            Dictionary with training metrics.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Load model and tokenizer
        logger.info("Loading model: %s", self.model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        # Detect target modules for LoRA
        target_modules = self._detect_target_modules(model)
        logger.info("LoRA target modules: %s", target_modules)

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Create datasets
        train_dataset = InstructionDataset(train_data, tokenizer, self.max_length)
        eval_dataset = InstructionDataset(eval_data, tokenizer, self.max_length) if eval_data else None

        # Training arguments
        training_args = transformers.TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            logging_steps=10,
            save_steps=200,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=200 if eval_dataset else None,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none",
            load_best_model_at_end=True if eval_dataset else False,
            remove_unused_columns=False,
        )

        # Trainer
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        # Train
        logger.info("Starting instruction tuning...")
        train_result = trainer.train()

        # Save
        trainer.save_model(os.path.join(self.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(self.output_dir, "final"))

        metrics = {
            "train_loss": train_result.training_loss,
            "train_steps": train_result.global_step,
        }
        logger.info("Instruction tuning complete: %s", metrics)
        return metrics

    def _detect_target_modules(self, model) -> List[str]:
        """Auto-detect appropriate LoRA target modules for the model architecture."""
        module_names = [name for name, _ in model.named_modules()]
        module_str = " ".join(module_names)

        if "q_proj" in module_str:
            return ["q_proj", "v_proj"]
        elif "c_attn" in module_str:
            return ["c_attn"]
        elif "query" in module_str:
            return ["query", "value"]
        elif "self_attn" in module_str:
            return ["self_attn"]
        else:
            # Fallback: find linear layers
            linear_names = set()
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    parts = name.split(".")
                    linear_names.add(parts[-1])
            return list(linear_names)[:2] if linear_names else ["q_proj", "v_proj"]
