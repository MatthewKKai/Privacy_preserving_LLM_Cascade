"""
Step 5: PPO Deferral Policy Training for P3Defer.

Trains the RL-based deferral policy (Algorithm 1) that decides whether
to answer locally, defer to server, or abstain.

Usage:
    python run_ppo_training.py \
        --dataset gsm8k \
        --data_dir ./data/gsm8k \
        --model_name google/gemma-2-2b-it \
        --memory_path ./output/private_memory.json \
        --output_dir ./output/ppo_policy \
        --num_iterations 50 \
        --lambda_privacy 0.5
"""

import argparse
import json
import logging
import os

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PPO deferral policy training")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "medsum", "emailsum"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--memory_path", type=str, default=None, help="Path to private memory JSON")
    parser.add_argument("--output_dir", type=str, default="./output/ppo_policy")
    parser.add_argument("--num_iterations", type=int, default=50, help="Number of PPO training iterations")
    parser.add_argument("--samples_per_iter", type=int, default=100, help="Samples per rollout iteration")
    parser.add_argument("--lambda_privacy", type=float, default=0.5, help="Privacy weight in reward")
    parser.add_argument("--lr_policy", type=float, default=3e-4)
    parser.add_argument("--lr_value", type=float, default=1e-3)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--state_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--use_model_signals", action="store_true",
                        help="Use actual model for privacy/quality signals (requires GPU)")
    args = parser.parse_args()

    from p3defer.data import get_processor
    from p3defer.training import PPODeferralTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    processor = get_processor(args.dataset)
    train_data, _ = processor.load_raw(args.data_dir)

    # Initialize PPO trainer
    trainer = PPODeferralTrainer(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        lambda_privacy=args.lambda_privacy,
        device=device,
    )

    # Prepare signals
    if args.use_model_signals:
        # Use actual model to generate privacy/quality signals
        logger.info("Loading model for signal generation: %s", args.model_name)
        from p3defer.models.cascade_model import CascadeModel
        cascade = CascadeModel(
            local_model_name=args.model_name,
            load_server=False,
            device=device,
        )
        cascade.to(device)
        cascade.eval()

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    training_log = []

    for iteration in range(args.num_iterations):
        # Sample batch
        import random
        batch_size = min(args.samples_per_iter, len(train_data))
        batch = random.sample(train_data, batch_size)

        privacy_signals = []
        quality_signals = []
        privacy_labels = []

        for item in batch:
            has_privacy = item.get("privacy", 0) == 1
            privacy_labels.append(has_privacy)

            if args.use_model_signals:
                # Get actual model signals
                import transformers
                inputs = cascade.tokenizer(
                    item["question"], return_tensors="pt",
                    truncation=True, max_length=256,
                ).to(device)
                with torch.no_grad():
                    priv_sig = cascade.get_privacy_signal(
                        inputs["input_ids"], inputs["attention_mask"]
                    )
                    qual_sig = cascade.get_quality_signal(
                        inputs["input_ids"], inputs["attention_mask"]
                    )
                privacy_signals.append(priv_sig.squeeze(0).cpu())
                quality_signals.append(qual_sig.squeeze(0).cpu())
            else:
                # Use synthetic signals based on labels
                if has_privacy:
                    priv_sig = torch.tensor([0.2, 0.8])  # High privacy probability
                else:
                    priv_sig = torch.tensor([0.8, 0.2])  # Low privacy probability
                # Random quality signal
                qual_sig = torch.tensor([random.uniform(0.3, 0.9)])
                privacy_signals.append(priv_sig)
                quality_signals.append(qual_sig)

        # Collect rollout
        rollout_stats = trainer.collect_rollout(
            privacy_signals=privacy_signals,
            quality_signals=quality_signals,
            privacy_labels=privacy_labels,
        )

        # PPO update
        update_stats = trainer.update()

        # Log
        log_entry = {
            "iteration": iteration + 1,
            **rollout_stats,
            **update_stats,
        }
        training_log.append(log_entry)

        if (iteration + 1) % 5 == 0:
            logger.info(
                "Iter %d/%d: reward=%.3f, policy_loss=%.4f, "
                "local=%d, defer=%d, abstain=%d",
                iteration + 1, args.num_iterations,
                rollout_stats["mean_reward"],
                update_stats["policy_loss"],
                rollout_stats["action_local"],
                rollout_stats["action_defer"],
                rollout_stats["action_abstain"],
            )

    # Save
    trainer.save(args.output_dir)
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info("PPO training complete. Models saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
