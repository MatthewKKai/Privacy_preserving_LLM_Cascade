#!/bin/bash
# Step 6: Evaluation
# Runs the full cascade on the test set and computes all metrics.

DATASET=${1:-gsm8k}
DATA_DIR=${2:-./data/gsm8k}
POLICY_DIR=${3:-./output/ppo_policy}
MEMORY_PATH=${4:-./output/private_memory.json}
OUTPUT_DIR=${5:-./output/eval_results}

echo "=== Step 6: Evaluation ==="

python run_evaluation.py \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --policy_dir ${POLICY_DIR} \
    --memory_path ${MEMORY_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --use_api_server

echo "=== Evaluation complete ==="
