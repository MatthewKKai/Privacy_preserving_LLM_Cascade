#!/bin/bash
# Step 3: CoT-enhanced Instruction Tuning
# Fine-tunes the local LLM with Chain-of-Thought prompts using LoRA.

DATASET=${1:-gsm8k}
DATA_DIR=${2:-./data/gsm8k}
MODEL_NAME=${3:-google/gemma-2-2b-it}
OUTPUT_DIR=${4:-./output/instruction_tuning}

echo "=== Step 3: Instruction Tuning ==="

python run_instruction_tuning.py \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs 3 \
    --batch_size 4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps 4

echo "=== Instruction tuning complete ==="
