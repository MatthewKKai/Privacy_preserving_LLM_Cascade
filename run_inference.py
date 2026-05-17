"""
Step 7: Inference for P3Defer.

Runs the full P3Defer cascade on individual queries or a batch of queries.
Supports interactive mode and batch mode.

Usage (interactive):
    python run_inference.py \
        --dataset gsm8k \
        --memory_path ./output/private_memory.json \
        --policy_dir ./output/ppo_policy \
        --interactive

Usage (batch):
    python run_inference.py \
        --dataset gsm8k \
        --memory_path ./output/private_memory.json \
        --policy_dir ./output/ppo_policy \
        --input_file queries.jsonl \
        --output_file results.json
"""

import argparse
import json
import logging
import os
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


ACTION_NAMES = {0: "LOCAL", 1: "DEFER", 2: "ABSTAIN"}


def run_single_query(
    query: str,
    processor,
    memory,
    ppo_policy,
    state_encoder,
    api_client,
    device: str = "cpu",
) -> dict:
    """Run the P3Defer cascade on a single query."""
    # Privacy detection
    if memory:
        detections = memory.detect_private_tokens(query)
        has_privacy = len(detections) > 0
    else:
        has_privacy = False

    # Determine action
    if ppo_policy is not None and state_encoder is not None:
        if has_privacy:
            priv_sig = torch.tensor([[0.2, 0.8]], device=device)
        else:
            priv_sig = torch.tensor([[0.8, 0.2]], device=device)
        import random
        qual_sig = torch.tensor([[random.uniform(0.4, 0.8)]], device=device)

        with torch.no_grad():
            state = state_encoder(priv_sig, qual_sig)
            action_tensor, _, _ = ppo_policy.get_action(state, deterministic=True)
            action = action_tensor.item()
    else:
        action = 1 if has_privacy else 0

    # Generate response
    masked_query = query
    if action == 1 and memory and has_privacy:
        masked_query, replacements, num_masked = memory.mask_query(query)
    else:
        replacements = []
        num_masked = 0

    prompt = processor.format_input({"question": masked_query if action == 1 else query})

    response = ""
    model_used = ""
    if action == 0:
        model_used = "local (gpt-4.1-nano)"
        if api_client:
            try:
                resp = api_client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                )
                response = resp.choices[0].message.content.strip()
            except Exception as e:
                response = f"[Error: {e}]"
    elif action == 1:
        model_used = "server (gpt-4.1-mini)"
        if api_client:
            try:
                resp = api_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                )
                response = resp.choices[0].message.content.strip()
            except Exception as e:
                response = f"[Error: {e}]"
    else:
        model_used = "abstain"
        response = "[Query abstained due to high privacy risk and low confidence]"

    return {
        "query": query,
        "action": ACTION_NAMES[action],
        "has_privacy": has_privacy,
        "masked_query": masked_query if action == 1 else None,
        "num_tokens_masked": num_masked,
        "replacements": replacements,
        "model_used": model_used,
        "response": response,
    }


def main():
    parser = argparse.ArgumentParser(description="P3Defer inference")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "medsum", "emailsum"])
    parser.add_argument("--memory_path", type=str, default=None)
    parser.add_argument("--policy_dir", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="./output/inference_results.json")
    args = parser.parse_args()

    from p3defer.data import get_processor
    from p3defer.memory import PrivateMemory

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = get_processor(args.dataset)

    # Load memory
    memory = None
    if args.memory_path and os.path.exists(args.memory_path):
        memory = PrivateMemory()
        memory.load(args.memory_path)
        logger.info("Loaded private memory with %d tokens", memory.size)

    # Load policy
    ppo_policy = None
    state_encoder = None
    if args.policy_dir and os.path.exists(args.policy_dir):
        from p3defer.training import PPODeferralTrainer
        trainer = PPODeferralTrainer(device=device)
        trainer.load(args.policy_dir)
        ppo_policy = trainer.policy
        state_encoder = trainer.state_encoder
        logger.info("Loaded PPO policy from %s", args.policy_dir)

    # API client
    api_client = None
    try:
        from openai import OpenAI
        api_client = OpenAI()
    except ImportError:
        logger.warning("OpenAI not installed. Responses will be empty.")

    if args.interactive:
        print("\n" + "=" * 60)
        print("P3Defer Interactive Inference")
        print("=" * 60)
        print("Enter a query (or 'quit' to exit):\n")

        while True:
            try:
                query = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue

            result = run_single_query(
                query, processor, memory, ppo_policy, state_encoder, api_client, device
            )

            print(f"\n  Action:        {result['action']}")
            print(f"  Has Privacy:   {result['has_privacy']}")
            if result['masked_query']:
                print(f"  Masked Query:  {result['masked_query']}")
                print(f"  Tokens Masked: {result['num_tokens_masked']}")
            print(f"  Model Used:    {result['model_used']}")
            print(f"  Response:      {result['response'][:500]}")
            print()

    elif args.input_file:
        # Batch mode
        queries = []
        with open(args.input_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        queries.append(item.get("question", item.get("query", line)))
                    except json.JSONDecodeError:
                        queries.append(line)

        results = []
        for i, query in enumerate(queries):
            result = run_single_query(
                query, processor, memory, ppo_policy, state_encoder, api_client, device
            )
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info("Processed %d/%d queries", i + 1, len(queries))

        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output_file)

    else:
        parser.print_help()
        print("\nPlease specify --interactive or --input_file")


if __name__ == "__main__":
    main()
