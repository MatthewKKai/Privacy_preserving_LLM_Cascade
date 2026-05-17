"""
PPO Trainer for the P3Defer deferral policy.

Implements Algorithm 1 from the paper: PPO-based training of the deferral
policy network pi_theta that decides whether to answer locally (a1),
defer to server (a2), or abstain (a3).

The reward function is:
    R_t = P^q(y, y_hat) + lambda * P^p(x)

where P^q is the quality score and P^p is the privacy preservation score.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.policy_network import PolicyNetwork, ValueNetwork, StateEncoder

logger = logging.getLogger(__name__)


class RewardFunction:
    """Computes the reward R_t = P^q(y, y_hat) + lambda * P^p(x).

    P^q: Quality score - measures how good the answer is.
    P^p: Privacy score - measures how well privacy is preserved.
    """

    def __init__(self, lambda_privacy: float = 0.5):
        """Initialize the reward function.

        Args:
            lambda_privacy: Weight for the privacy component.
        """
        self.lambda_privacy = lambda_privacy

    def compute(
        self,
        action: int,
        quality_score: float,
        has_privacy: bool,
        privacy_masked: bool = False,
    ) -> float:
        """Compute the reward for a single step.

        Args:
            action: The action taken (0=local, 1=defer, 2=abstain).
            quality_score: Quality of the generated answer (0-1).
            has_privacy: Whether the query contains private info.
            privacy_masked: Whether private tokens were masked before deferral.

        Returns:
            Scalar reward value.
        """
        # Quality component
        p_q = quality_score

        # Privacy component
        if action == 0:  # Local: no privacy risk
            p_p = 1.0
        elif action == 1:  # Defer to server
            if has_privacy:
                p_p = 0.8 if privacy_masked else 0.0  # Partial credit if masked
            else:
                p_p = 1.0  # No privacy concern
        else:  # Abstain: safe but no answer
            p_p = 1.0
            p_q = 0.0  # No quality since no answer

        reward = p_q + self.lambda_privacy * p_p
        return reward


class PPODeferralTrainer:
    """PPO trainer for the deferral policy.

    Implements the full PPO training loop (Algorithm 1) with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        state_dim: int = 128,
        hidden_dim: int = 128,
        num_actions: int = 3,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 32,
        lambda_privacy: float = 0.5,
        device: str = "cpu",
    ):
        """Initialize the PPO trainer.

        Args:
            state_dim: Dimension of the state vector.
            hidden_dim: Hidden layer dimension for networks.
            num_actions: Number of possible actions.
            lr_policy: Learning rate for the policy network.
            lr_value: Learning rate for the value network.
            gamma: Discount factor.
            gae_lambda: GAE lambda for advantage estimation.
            clip_epsilon: PPO clipping parameter.
            entropy_coeff: Entropy bonus coefficient.
            value_loss_coeff: Value loss coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            ppo_epochs: Number of PPO optimization epochs per update.
            mini_batch_size: Mini-batch size for PPO updates.
            lambda_privacy: Privacy weight in the reward function.
            device: Device to train on.
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # Networks
        self.state_encoder = StateEncoder(
            privacy_dim=2, quality_dim=1,
            hidden_dim=hidden_dim, state_dim=state_dim,
        ).to(device)
        self.policy = PolicyNetwork(state_dim, hidden_dim, num_actions).to(device)
        self.value = ValueNetwork(state_dim, hidden_dim).to(device)

        # Optimizers
        policy_params = list(self.state_encoder.parameters()) + list(self.policy.parameters())
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)

        # Reward function
        self.reward_fn = RewardFunction(lambda_privacy)

        # Rollout buffer
        self._buffer = RolloutBuffer()

    def collect_rollout(
        self,
        privacy_signals: List[torch.Tensor],
        quality_signals: List[torch.Tensor],
        privacy_labels: List[bool],
        quality_evaluator=None,
    ) -> Dict[str, float]:
        """Collect a rollout of experience.

        For each query, the agent observes the state, takes an action,
        and receives a reward based on the quality and privacy outcomes.

        Args:
            privacy_signals: List of [2] privacy logit tensors.
            quality_signals: List of [1] quality score tensors.
            privacy_labels: List of boolean privacy labels.
            quality_evaluator: Optional callable(action, idx) -> quality_score.

        Returns:
            Dictionary with rollout statistics.
        """
        self._buffer.clear()
        total_reward = 0.0
        action_counts = {0: 0, 1: 0, 2: 0}

        for i in range(len(privacy_signals)):
            priv_sig = privacy_signals[i].unsqueeze(0).to(self.device)
            qual_sig = quality_signals[i].unsqueeze(0).to(self.device)

            # Encode state
            with torch.no_grad():
                state = self.state_encoder(priv_sig, qual_sig)
                action, log_prob, _ = self.policy.get_action(state)
                value = self.value(state)

            action_idx = action.item()
            action_counts[action_idx] += 1

            # Compute quality score
            if quality_evaluator is not None:
                q_score = quality_evaluator(action_idx, i)
            else:
                # Default: local gets moderate quality, server gets high, abstain gets 0
                q_score = {0: 0.6, 1: 0.8, 2: 0.0}[action_idx]

            # Compute reward
            reward = self.reward_fn.compute(
                action=action_idx,
                quality_score=q_score,
                has_privacy=privacy_labels[i],
                privacy_masked=(action_idx == 1),  # Assume masking when deferring
            )
            total_reward += reward

            self._buffer.add(
                state=state.squeeze(0),
                action=action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                value=value.squeeze(0),
                reward=reward,
            )

        stats = {
            "mean_reward": total_reward / max(len(privacy_signals), 1),
            "total_reward": total_reward,
            "action_local": action_counts[0],
            "action_defer": action_counts[1],
            "action_abstain": action_counts[2],
            "num_steps": len(privacy_signals),
        }
        return stats

    def update(self) -> Dict[str, float]:
        """Perform PPO update using collected rollout data.

        Returns:
            Dictionary with training statistics.
        """
        if len(self._buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        # Compute returns and advantages using GAE
        states, actions, old_log_probs, values, rewards = self._buffer.get_all()
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        values = values.to(self.device)

        returns, advantages = self._compute_gae(rewards, values)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(states))
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # Policy loss
                new_log_probs, entropy = self.policy.evaluate_action(mb_states, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # Value loss
                new_values = self.value(mb_states).squeeze(-1)
                value_loss = F.mse_loss(new_values, mb_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                # Update
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def _compute_gae(
        self, rewards: List[float], values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: List of scalar rewards.
            values: [T] tensor of value estimates.

        Returns:
            Tuple of (returns, advantages) tensors.
        """
        T = len(rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)

        last_gae = 0.0
        last_value = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * last_value - values[t].item()
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t].item()
            last_value = values[t].item()

        return returns, advantages

    def save(self, output_dir: str) -> None:
        """Save the policy, value, and state encoder networks."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(output_dir, "policy.pt"))
        torch.save(self.value.state_dict(), os.path.join(output_dir, "value.pt"))
        torch.save(self.state_encoder.state_dict(), os.path.join(output_dir, "state_encoder.pt"))
        logger.info("Saved PPO models to %s", output_dir)

    def load(self, output_dir: str) -> None:
        """Load the policy, value, and state encoder networks."""
        self.policy.load_state_dict(
            torch.load(os.path.join(output_dir, "policy.pt"), map_location=self.device)
        )
        self.value.load_state_dict(
            torch.load(os.path.join(output_dir, "value.pt"), map_location=self.device)
        )
        self.state_encoder.load_state_dict(
            torch.load(os.path.join(output_dir, "state_encoder.pt"), map_location=self.device)
        )
        logger.info("Loaded PPO models from %s", output_dir)


class RolloutBuffer:
    """Simple rollout buffer for PPO."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
    ):
        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.values.append(value.detach().cpu())
        self.rewards.append(reward)

    def get_all(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.cat(self.values).squeeze(-1) if self.values[0].dim() > 0 else torch.tensor(
                [v.item() for v in self.values]
            ),
            self.rewards,
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()

    def __len__(self):
        return len(self.states)
