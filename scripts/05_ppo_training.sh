#!/bin/bash
# Step 5: PPO Deferral Policy Training
# Trains the RL-based deferral policy (Algorithm 1).

DATASET=${1:-gsm8k}
DATA_DIR=${2:-./data/gsm8k}
MEMORY_PATH=${3:-./output/private_memory.json}
OUTPUT_DIR=${4:-./output/ppo_policy}

echo "=== Step 5: PPO Deferral Policy Training ==="

python run_ppo_training.py \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --memory_path ${MEMORY_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_iterations 50 \
    --samples_per_iter 100 \
    --lambda_privacy 0.5 \
    --lr_policy 3e-4 \
    --lr_value 1e-3 \
    --ppo_epochs 4

echo "=== PPO training complete ==="
