"""
Step 3: CoT-enhanced Instruction Tuning for P3Defer.

Fine-tunes the local LLM with Chain-of-Thought enhanced prompts
using LoRA for parameter efficiency.

Usage:
    python run_instruction_tuning.py \
        --dataset gsm8k \
        --data_dir ./data/gsm8k \
        --model_name google/gemma-2-2b-it \
        --output_dir ./output/instruction_tuning \
        --num_epochs 3 \
        --batch_size 4 \
        --lora_rank 16
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CoT-enhanced instruction tuning")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "medsum", "emailsum"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--output_dir", type=str, default="./output/instruction_tuning")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_split", type=float, default=0.1, help="Fraction of data for evaluation")
    args = parser.parse_args()

    from p3defer.data import get_processor
    from p3defer.training import InstructionTuner

    # Load data
    processor = get_processor(args.dataset)
    train_data, test_data = processor.load_raw(args.data_dir)

    # Prepare CoT instruction data
    tuner = InstructionTuner(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
    )

    cot_train = tuner.prepare_cot_data(train_data, args.dataset)

    # Split for evaluation
    eval_size = int(len(cot_train) * args.eval_split)
    if eval_size > 0:
        cot_eval = cot_train[-eval_size:]
        cot_train = cot_train[:-eval_size]
    else:
        cot_eval = None

    logger.info("Training on %d examples, evaluating on %d examples",
                len(cot_train), len(cot_eval) if cot_eval else 0)

    # Train
    metrics = tuner.train(cot_train, cot_eval)
    logger.info("Instruction tuning complete: %s", metrics)


if __name__ == "__main__":
    main()
