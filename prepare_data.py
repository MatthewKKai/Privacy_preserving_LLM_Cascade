"""
Step 1: Data Preparation for P3Defer.

Downloads and prepares datasets with privacy labels.
Supports GSM8K, MedSum, and EmailSum.

Usage:
    python prepare_data.py --dataset gsm8k --output_dir ./data
    python prepare_data.py --dataset gsm8k --output_dir ./data --use_openai_labels
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for P3Defer")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "medsum", "emailsum"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--use_openai_labels",
        action="store_true",
        help="Use OpenAI API to refine privacy labels",
    )
    args = parser.parse_args()

    from p3defer.data import download_and_prepare_dataset

    logger.info("Preparing dataset: %s", args.dataset)
    output_path = download_and_prepare_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        use_openai_for_labels=args.use_openai_labels,
    )
    logger.info("Dataset prepared at: %s", output_path)


if __name__ == "__main__":
    main()
