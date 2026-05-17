#!/bin/bash
# Step 1: Data Preparation
# Downloads and prepares datasets with privacy labels.
# Supports: gsm8k, medsum, emailsum

DATASET=${1:-gsm8k}
OUTPUT_DIR=${2:-./data}

echo "=== Step 1: Preparing dataset: ${DATASET} ==="

python prepare_data.py \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR}

echo "=== Data preparation complete ==="
