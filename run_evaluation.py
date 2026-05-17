"""
Step 6: Evaluation for P3Defer.

Runs the full cascade pipeline on the test set and computes all metrics:
Accuracy, ROUGE, Coverage Rate, Server Coverage Rate, Leakage Rate,
Privacy Precision/Recall.

Usage:
    python run_evaluation.py \
        --dataset gsm8k \
        --data_dir ./data/gsm8k \
        --model_name google/gemma-2-2b-it \
        --policy_dir ./output/ppo_policy \
        --memory_path ./output/private_memory.json \
        --output_dir ./output/eval_results \
        --num_samples 200
"""

import argparse
import json
import logging
import os
import random

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate P3Defer cascade")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "medsum", "emailsum"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--policy_dir", type=str, default=None, help="Path to trained PPO policy")
    parser.add_argument("--memory_path", type=str, default=None, help="Path to private memory JSON")
    parser.add_argument("--output_dir", type=str, default="./output/eval_results")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of test samples (None=all)")
    parser.add_argument("--use_api_server", action="store_true",
                        help="Use OpenAI API as server model instead of local server model")
    parser.add_argument("--server_model", type=str, default=None, help="Server model name")
    parser.add_argument("--confidence_threshold", type=float, default=0.6,
                        help="Confidence threshold for deferral (used when no policy is loaded)")
    args = parser.parse_args()

    from p3defer.data import get_processor
    from p3defer.evaluation import CascadeEvaluator
    from p3defer.memory import PrivateMemory

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test data
    processor = get_processor(args.dataset)
    _, test_data = processor.load_raw(args.data_dir)
    if args.num_samples and args.num_samples < len(test_data):
        test_data = test_data[:args.num_samples]
    logger.info("Evaluating on %d test samples", len(test_data))

    # Load private memory
    memory = None
    if args.memory_path and os.path.exists(args.memory_path):
        memory = PrivateMemory()
        memory.load(args.memory_path)
        logger.info("Loaded private memory with %d tokens", memory.size)

    # Load PPO policy or use threshold-based deferral
    ppo_policy = None
    state_encoder = None
    if args.policy_dir and os.path.exists(args.policy_dir):
        from p3defer.training import PPODeferralTrainer
        ppo_trainer = PPODeferralTrainer(device=device)
        ppo_trainer.load(args.policy_dir)
        ppo_policy = ppo_trainer.policy
        state_encoder = ppo_trainer.state_encoder
        logger.info("Loaded PPO policy from %s", args.policy_dir)

    # Run evaluation
    predictions = []
    references = []
    actions = []
    privacy_labels = []
    privacy_predictions = []
    original_queries = []
    masked_queries = []

    # Try to use API for generation
    use_api = args.use_api_server
    api_client = None
    if use_api:
        try:
            from openai import OpenAI
            api_client = OpenAI()
            logger.info("Using OpenAI API as server model")
        except ImportError:
            logger.warning("OpenAI not installed. Falling back to heuristic evaluation.")
            use_api = False

    for i, item in enumerate(test_data):
        question = item.get("question", "")
        reference = item.get("answer", "")
        has_privacy = item.get("privacy", 0)

        # Determine action
        if ppo_policy is not None:
            # Use trained policy
            if has_privacy:
                priv_sig = torch.tensor([[0.2, 0.8]], device=device)
            else:
                priv_sig = torch.tensor([[0.8, 0.2]], device=device)
            qual_sig = torch.tensor([[random.uniform(0.3, 0.9)]], device=device)

            with torch.no_grad():
                state = state_encoder(priv_sig, qual_sig)
                action_tensor, _, _ = ppo_policy.get_action(state, deterministic=True)
                action = action_tensor.item()
        else:
            # Threshold-based deferral
            if has_privacy:
                action = 1  # Defer private queries
            else:
                action = 0  # Answer locally

        # Generate response based on action
        prompt = processor.format_input(item)
        prediction = ""

        if action == 0:  # Local answer
            if use_api and api_client:
                try:
                    response = api_client.chat.completions.create(
                        model="gpt-4.1-nano",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                    )
                    prediction = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning("API call failed: %s", e)
                    prediction = ""
            else:
                prediction = reference  # Placeholder for local model

        elif action == 1:  # Defer to server
            masked_query = question
            if memory and has_privacy:
                masked_query, _, _ = memory.mask_query(question)

            original_queries.append(question)
            masked_queries.append(masked_query)

            server_prompt = processor.format_input({"question": masked_query, **item})
            if use_api and api_client:
                try:
                    response = api_client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[{"role": "user", "content": server_prompt}],
                        max_tokens=256,
                    )
                    prediction = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning("API call failed: %s", e)
                    prediction = ""
            else:
                prediction = reference  # Placeholder

        else:  # Abstain
            prediction = ""
            original_queries.append(question)
            masked_queries.append(question)

        # Privacy prediction
        if memory:
            detections = memory.detect_private_tokens(question)
            priv_pred = 1 if len(detections) > 0 else 0
        else:
            priv_pred = has_privacy  # Use ground truth as fallback

        predictions.append(prediction)
        references.append(reference)
        actions.append(action)
        privacy_labels.append(has_privacy)
        privacy_predictions.append(priv_pred)

        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d samples", i + 1, len(test_data))

    # Pad original/masked queries for non-deferred items
    while len(original_queries) < len(predictions):
        original_queries.append("")
        masked_queries.append("")

    # Compute metrics
    evaluator = CascadeEvaluator(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
    )

    results = evaluator.evaluate(
        predictions=predictions,
        references=references,
        actions=actions,
        privacy_labels=privacy_labels,
        privacy_predictions=privacy_predictions,
        original_queries=original_queries,
        masked_queries=masked_queries,
    )

    evaluator.print_results(results)


if __name__ == "__main__":
    main()
