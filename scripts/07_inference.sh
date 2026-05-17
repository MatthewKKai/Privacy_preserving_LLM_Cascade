#!/bin/bash
# Step 7: Interactive Inference
# Runs the full P3Defer cascade in interactive mode.

DATASET=${1:-gsm8k}
MEMORY_PATH=${2:-./output/private_memory.json}
POLICY_DIR=${3:-./output/ppo_policy}

echo "=== Step 7: Interactive Inference ==="

python run_inference.py \
    --dataset ${DATASET} \
    --memory_path ${MEMORY_PATH} \
    --policy_dir ${POLICY_DIR} \
    --interactive

echo "=== Inference session ended ==="
