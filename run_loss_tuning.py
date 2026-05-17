"""
Step 4: Multi-Objective Loss Tuning for P3Defer.

Trains the local model with the combined loss (Eq. 9-10):
    L = L_task + alpha * L_privacy + beta * H(.) * L_KD

Usage:
    python run_loss_tuning.py \
        --dataset gsm8k \
        --data_dir ./data/gsm8k \
        --local_model google/gemma-2-2b-it \
        --server_model google/gemma-2-9b-it \
        --output_dir ./output/loss_tuning \
        --alpha 0.4 \
        --beta 0.1
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Multi-objective loss tuning")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "medsum", "emailsum"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--local_model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--server_model", type=str, default=None, help="Server model (None to skip distillation)")
    parser.add_argument("--output_dir", type=str, default="./output/loss_tuning")
    parser.add_argument("--alpha", type=float, default=0.4, help="Privacy loss weight")
    parser.add_argument("--beta", type=float, default=0.1, help="Distillation loss weight")
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    from p3defer.data import get_processor
    from p3defer.training import MultiObjectiveLossTuner
    from p3defer.training.instruction_tuner import InstructionTuner

    # Load data
    processor = get_processor(args.dataset)
    train_data, _ = processor.load_raw(args.data_dir)

    # Prepare instruction data format
    tuner_helper = InstructionTuner(model_name=args.local_model)
    formatted_data = tuner_helper.prepare_cot_data(train_data, args.dataset)
    # Add privacy labels
    for item, raw in zip(formatted_data, train_data):
        item["privacy"] = raw.get("privacy", 0)

    # Train
    loss_tuner = MultiObjectiveLossTuner(
        local_model_name=args.local_model,
        server_model_name=args.server_model,
        alpha=args.alpha,
        beta=args.beta,
        confidence_threshold=args.confidence_threshold,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )

    metrics = loss_tuner.train(formatted_data)
    logger.info("Loss tuning complete: %s", metrics)


if __name__ == "__main__":
    main()
