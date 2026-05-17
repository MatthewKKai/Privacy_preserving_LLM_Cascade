"""
Step 2: Build Private Memory for P3Defer.

Scans the training corpus to extract private tokens and build the
private memory module used for query masking during deferral.

Usage:
    python build_memory.py \
        --dataset gsm8k \
        --data_dir ./data/gsm8k \
        --output_path ./output/private_memory.json \
        --threshold 0.3
"""

import argparse
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build private memory for P3Defer")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "medsum", "emailsum"])
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing prepared data")
    parser.add_argument("--output_path", type=str, default="./output/private_memory.json")
    parser.add_argument("--threshold", type=float, default=0.3, help="Levenshtein distance threshold")
    args = parser.parse_args()

    from p3defer.data import get_processor
    from p3defer.memory import PrivateMemory

    # Load training data
    processor = get_processor(args.dataset)
    train_data, _ = processor.load_raw(args.data_dir)

    # Build private memory
    memory = PrivateMemory(threshold=args.threshold)
    texts = [d.get("question", "") for d in train_data]
    labels = [d.get("privacy", 0) for d in train_data]

    added = memory.add_tokens_from_corpus(texts, labels)
    logger.info("Built private memory with %d tokens (%d added from corpus)", memory.size, added)

    # Compute detection metrics on training data
    metrics = memory.compute_detection_metrics(texts, labels)
    logger.info("Detection metrics on training data: %s", metrics)

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    memory.save(args.output_path)
    logger.info("Private memory saved to %s", args.output_path)

    # Also save detection metrics
    metrics_path = args.output_path.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
