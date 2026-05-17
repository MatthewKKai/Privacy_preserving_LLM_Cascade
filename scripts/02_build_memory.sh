#!/bin/bash
# Step 2: Build Private Memory
# Extracts private tokens from training data and builds the memory module.

DATASET=${1:-gsm8k}
DATA_DIR=${2:-./data/gsm8k}
OUTPUT_PATH=${3:-./output/private_memory.json}
THRESHOLD=${4:-0.3}

echo "=== Step 2: Building private memory ==="

python build_memory.py \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --output_path ${OUTPUT_PATH} \
    --threshold ${THRESHOLD}

echo "=== Private memory built ==="
