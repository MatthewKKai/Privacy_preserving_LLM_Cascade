#!/bin/bash
# Step 4: Multi-Objective Loss Tuning
# Trains with combined loss: L = L_task + alpha * L_privacy + beta * L_KD

DATASET=${1:-gsm8k}
DATA_DIR=${2:-./data/gsm8k}
LOCAL_MODEL=${3:-google/gemma-2-2b-it}
SERVER_MODEL=${4:-google/gemma-2-9b-it}
OUTPUT_DIR=${5:-./output/loss_tuning}

echo "=== Step 4: Multi-Objective Loss Tuning ==="

python run_loss_tuning.py \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --local_model ${LOCAL_MODEL} \
    --server_model ${SERVER_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --alpha 0.4 \
    --beta 0.1 \
    --num_epochs 3 \
    --batch_size 4

echo "=== Loss tuning complete ==="
