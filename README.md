# P3Defer: Privacy-Preserved LLM Cascade via CoT-Enhanced Policy Learning

This repository implements **P3Defer**, a privacy-preserving LLM cascade framework that uses Chain-of-Thought (CoT)-enhanced policy learning with a private memory module. The system learns when to answer queries locally, defer to a more capable server model, or apply privacy masking before deferral.

> **Paper:** Kai Zhang, Congchao Wang, Liqian Peng, Alec Go, Xiaozhong Liu. "Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning." In *Proceedings of the ACM Web Conference (WWW)*, 2025.

## Overview

On-device LLMs are limited by hardware constraints, so LLM cascading routes difficult queries to a stronger server model. However, this introduces **privacy risks** when user queries contain sensitive information. P3Defer addresses this by:

1. **RL-based Deferral Policy** (Algorithm 1): A PPO-trained agent that observes privacy and quality signals to decide among three actions:
   - **a1 (Local):** Answer using the local LLM
   - **a2 (Defer):** Route the query to the server LLM
   - **a3 (Mask + Defer):** Mask private tokens, then route to the server LLM

2. **Private Memory Module:** A dynamic token store that detects and masks personally identifiable information using Levenshtein distance matching before queries are sent to the server.

3. **CoT-Enhanced Instruction Tuning:** Fine-tunes the local LLM with Chain-of-Thought reasoning to improve both task performance and privacy awareness.

4. **Multi-Objective Loss Tuning:** Combines task loss, privacy classification loss, and knowledge distillation loss with a Heaviside step function for deferral-aware training.

## Architecture

```
Query x → [Local LLM Φ(L)] → y^L (local response)
                ↓
        [State Encoder] → s_t = [e^p; e^q]
                ↓
        [Policy π_θ (PPO)] → action ∈ {a1, a2, a3}
                ↓
        ┌───────┼───────────┐
        a1      a2          a3
     (local)  (defer)   (mask+defer)
        ↓       ↓           ↓
      y=y^L   y=Φ(S)(x)  y=Φ(S)(M(x))
```

## Project Structure

```
Privacy_preserving_LLM_Cascade/
├── p3defer/                        # Core P3Defer package
│   ├── __init__.py
│   ├── data/                       # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── datasets.py             # GSM8K, MedSum, EmailSum processors
│   ├── memory/                     # Private memory module
│   │   ├── __init__.py
│   │   └── private_memory.py       # Levenshtein-based token masking
│   ├── models/                     # Neural network models
│   │   ├── __init__.py
│   │   ├── policy_network.py       # Policy π_θ, Value V_φ, State Encoder
│   │   └── cascade_model.py        # CascadeModel with privacy head
│   ├── training/                   # Training modules
│   │   ├── __init__.py
│   │   ├── ppo_trainer.py          # PPO deferral training (Algorithm 1)
│   │   ├── instruction_tuner.py    # CoT-enhanced instruction tuning
│   │   └── loss_tuner.py           # Multi-objective loss tuning (Eq. 9-10)
│   └── evaluation/                 # Evaluation metrics
│       ├── __init__.py
│       └── evaluator.py            # Acc, ROUGE, CR, SCR, leakage, P/R
├── prepare_data.py                 # Step 1: Data preparation
├── build_memory.py                 # Step 2: Build private memory
├── run_instruction_tuning.py       # Step 3: Instruction tuning
├── run_loss_tuning.py              # Step 4: Multi-objective loss tuning
├── run_ppo_training.py             # Step 5: PPO deferral policy training
├── run_evaluation.py               # Step 6: Evaluation
├── run_inference.py                # Step 7: Inference (interactive/batch)
├── scripts/                        # Shell scripts for each stage
│   ├── 01_prepare_data.sh
│   ├── 02_build_memory.sh
│   ├── 03_instruction_tuning.sh
│   ├── 04_loss_tuning.sh
│   ├── 05_ppo_training.sh
│   ├── 06_evaluation.sh
│   └── 07_inference.sh
├── code/                           # Original prototype code (preserved)
│   ├── privacy_preserving_llm_cascade.py
│   ├── instruction_tuning.py
│   ├── loss_tuning.py
│   ├── prompt_engineering.py
│   └── ...
├── model/                          # Original model code (preserved)
│   └── ...
├── appendix.pdf                    # Paper appendix
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/MatthewKKai/Privacy_preserving_LLM_Cascade.git
cd Privacy_preserving_LLM_Cascade
pip install -r requirements.txt
```

**Dependencies:** PyTorch >= 2.0, Transformers >= 4.36, PEFT >= 0.7, OpenAI >= 1.0, rouge-score, python-Levenshtein.

**Hardware:** The full pipeline (Steps 3-4) requires a GPU with at least 16 GB VRAM for Gemma-2B. Steps 1-2 and 5-7 can run on CPU. For testing, GPT-2 can be used as a drop-in replacement.

## Pipeline: Step-by-Step Guide

The P3Defer pipeline consists of seven stages, each with its own script. Run them sequentially.

### Step 1: Data Preparation

Downloads and prepares datasets with privacy labels. Supports three datasets from the paper: **GSM8K** (math QA), **MedSum** (medical summarization), and **EmailSum** (email summarization).

```bash
# Prepare GSM8K dataset
python prepare_data.py --dataset gsm8k --output_dir ./data

# Prepare MedSum dataset
python prepare_data.py --dataset medsum --output_dir ./data

# Prepare EmailSum dataset
python prepare_data.py --dataset emailsum --output_dir ./data
```

**Output:** `./data/{dataset}/train.json` and `./data/{dataset}/test.json` with fields `{question, answer, privacy}`.

### Step 2: Build Private Memory

Scans the training corpus to extract private tokens (names, locations, medical terms, etc.) and builds the private memory module used for query masking during deferral.

```bash
python build_memory.py \
    --dataset gsm8k \
    --data_dir ./data/gsm8k \
    --output_path ./output/private_memory.json \
    --threshold 0.3
```

**Parameters:**
- `--threshold`: Levenshtein distance threshold for token matching (default: 0.3). Lower values are stricter.

**Output:** `./output/private_memory.json` containing the private token store and detection metrics.

### Step 3: CoT-Enhanced Instruction Tuning

Fine-tunes the local LLM with Chain-of-Thought enhanced prompts using LoRA. The CoT format teaches the model to: (a) detect privacy, (b) solve the task step-by-step, (c) self-critique, and (d) decide on rewriting.

```bash
python run_instruction_tuning.py \
    --dataset gsm8k \
    --data_dir ./data/gsm8k \
    --model_name google/gemma-2-2b-it \
    --output_dir ./output/instruction_tuning \
    --num_epochs 3 \
    --batch_size 4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 2e-4
```

**Parameters:**
- `--model_name`: Local LLM (default: `google/gemma-2-2b-it`). Use `gpt2` for testing.
- `--lora_rank`: LoRA rank (default: 16).
- `--lora_alpha`: LoRA alpha (default: 32).

**Output:** LoRA-tuned model saved to `./output/instruction_tuning/final/`.

### Step 4: Multi-Objective Loss Tuning

Trains the local model with the combined multi-objective loss (Eq. 9-10 in the paper):

> L = L_task + α · L_privacy + β · H(confidence < t) · L_KD

where H(·) is the Heaviside step function that activates knowledge distillation only when the local model's confidence is below threshold t.

```bash
python run_loss_tuning.py \
    --dataset gsm8k \
    --data_dir ./data/gsm8k \
    --local_model google/gemma-2-2b-it \
    --server_model google/gemma-2-9b-it \
    --output_dir ./output/loss_tuning \
    --alpha 0.4 \
    --beta 0.1 \
    --num_epochs 3
```

**Parameters:**
- `--alpha`: Weight for privacy classification loss (default: 0.4).
- `--beta`: Weight for knowledge distillation loss (default: 0.1).
- `--confidence_threshold`: Heaviside threshold (default: 0.6).
- `--server_model`: Set to `None` to skip distillation (CPU-only mode).

**Output:** Tuned local model and privacy head saved to `./output/loss_tuning/`.

### Step 5: PPO Deferral Policy Training

Trains the RL-based deferral policy using PPO (Algorithm 1). The agent learns to select among three actions: local answer, defer, or mask-and-defer.

```bash
python run_ppo_training.py \
    --dataset gsm8k \
    --data_dir ./data/gsm8k \
    --memory_path ./output/private_memory.json \
    --output_dir ./output/ppo_policy \
    --num_iterations 50 \
    --samples_per_iter 100 \
    --lambda_privacy 0.5 \
    --ppo_epochs 4
```

**Parameters:**
- `--lambda_privacy`: Privacy weight in the reward function R_t = P^q + λ · P^p (default: 0.5).
- `--num_iterations`: Number of PPO training iterations (default: 50).
- `--use_model_signals`: Use actual model outputs for privacy/quality signals (requires GPU).

**Output:** Policy, value, and state encoder networks saved to `./output/ppo_policy/`.

### Step 6: Evaluation

Runs the full cascade pipeline on the test set and computes all metrics from the paper.

```bash
python run_evaluation.py \
    --dataset gsm8k \
    --data_dir ./data/gsm8k \
    --policy_dir ./output/ppo_policy \
    --memory_path ./output/private_memory.json \
    --output_dir ./output/eval_results \
    --use_api_server
```

**Parameters:**
- `--use_api_server`: Use OpenAI API as the server model (requires `OPENAI_API_KEY`).
- `--num_samples`: Limit the number of test samples (default: all).

**Metrics computed:**

| Category | Metric | Description |
|----------|--------|-------------|
| Quality | Accuracy (Acc.) | Exact match for QA tasks |
| Quality | ROUGE-1, ROUGE-L | For summarization tasks |
| Efficiency | Coverage Rate (CR) | Fraction of queries answered |
| Efficiency | Server Coverage Rate (SCR) | Fraction deferred to server |
| Privacy | Privacy Precision | Precision of privacy detection |
| Privacy | Privacy Recall | Recall of privacy detection |
| Privacy | Leakage Rate r(leakage) | Private tokens leaked to server |

**Output:** `./output/eval_results/eval_results.json`.

### Step 7: Inference

Run the full P3Defer cascade on individual queries (interactive mode) or a batch of queries.

```bash
# Interactive mode
python run_inference.py \
    --dataset gsm8k \
    --memory_path ./output/private_memory.json \
    --policy_dir ./output/ppo_policy \
    --interactive

# Batch mode
python run_inference.py \
    --dataset gsm8k \
    --memory_path ./output/private_memory.json \
    --policy_dir ./output/ppo_policy \
    --input_file queries.jsonl \
    --output_file ./output/inference_results.json
```

## Quick Start (Testing without GPU)

For testing the full pipeline without a GPU, you can skip the model-dependent stages (Steps 3-4) and use synthetic signals for PPO training:

```bash
# Step 1: Prepare data
python prepare_data.py --dataset gsm8k --output_dir ./data

# Step 2: Build private memory
python build_memory.py --dataset gsm8k --data_dir ./data/gsm8k \
    --output_path ./output/private_memory.json

# Step 5: Train PPO policy (with synthetic signals)
python run_ppo_training.py --dataset gsm8k --data_dir ./data/gsm8k \
    --memory_path ./output/private_memory.json \
    --output_dir ./output/ppo_policy \
    --num_iterations 50 --samples_per_iter 100

# Step 6: Evaluate (with API server)
python run_evaluation.py --dataset gsm8k --data_dir ./data/gsm8k \
    --policy_dir ./output/ppo_policy \
    --memory_path ./output/private_memory.json \
    --output_dir ./output/eval_results \
    --use_api_server --num_samples 50

# Step 7: Interactive inference
python run_inference.py --dataset gsm8k \
    --memory_path ./output/private_memory.json \
    --policy_dir ./output/ppo_policy \
    --interactive
```

## Key Equations

The reward function for the deferral policy (Eq. 2):

> R_t = P^q(y, ŷ) + λ · P^p(x)

The multi-objective training loss (Eq. 9-10):

> L = L_task + α · L_privacy + β · H(logit < t) · L_KD

The PPO policy gradient (Eq. 5):

> ∇_θ J(π_θ) = E[∑ ∇_θ log π_θ(a_t | s_t) · Â_t]

## Reference Results (from the paper)

**Table 1: Cascade Performance (Gemma-2B → Gemma-7B)**

| Dataset | CR (%) | SCR (%) | Local Only | Cascade | Server Only |
|---------|--------|---------|------------|---------|-------------|
| GSM8K | 66.41 | 92.61 | 27.33 | **55.96** | 52.85 |
| MedSum | 69.71 | 88.40 | 35.31 | **63.94** | 61.21 |
| EmailSum | 94.61 | 94.61 | 28.91 | **61.21** | 57.33 |

**Table 2: Privacy Study**

| Dataset | Precision (%) | Recall (%) | Leakage Rate (%) |
|---------|---------------|------------|-------------------|
| GSM8K | 96.31 | 88.79 | 20.11 |
| MedSum | 92.17 | 85.56 | 23.87 |
| EmailSum | 96.91 | 85.77 | 16.34 |

## Citation

```bibtex
@inproceedings{zhang2025p3defer,
  title={Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning},
  author={Zhang, Kai and Wang, Congchao and Peng, Liqian and Go, Alec and Liu, Xiaozhong},
  booktitle={Proceedings of the ACM Web Conference (WWW)},
  year={2025}
}
```

## License

This project is for research purposes only.
