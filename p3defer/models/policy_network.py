"""
Policy Network, Value Network, and State Encoder for P3Defer.

Implements the RL-based deferral decision mechanism from Section 2.2 of the paper.

The agent observes state s_t = [e^p; e^q] (privacy embedding concatenated with
quality embedding) and selects an action from:
  a1: Answer directly using the local LLM
  a2: Defer to the server LLM (with privacy masking)
  a3: Abstain from answering

The policy is trained with PPO to maximize:
  R_t = P^q(y, y_hat) + lambda * P^p(x)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StateEncoder(nn.Module):
    """Encodes privacy and quality signals into a state representation.

    The state s_t = [e^p; e^q] is formed by concatenating:
    - e^p: Privacy embedding from the privacy detection head
    - e^q: Quality embedding from the local model's confidence/logits

    Both are projected to a common dimension and concatenated.
    """

    def __init__(
        self,
        privacy_dim: int = 2,
        quality_dim: int = 1,
        hidden_dim: int = 64,
        state_dim: int = 128,
    ):
        """Initialize the state encoder.

        Args:
            privacy_dim: Dimension of privacy signal (2 for binary classification logits).
            quality_dim: Dimension of quality signal (1 for confidence score).
            hidden_dim: Hidden layer dimension.
            state_dim: Output state dimension.
        """
        super().__init__()
        self.privacy_encoder = nn.Sequential(
            nn.Linear(privacy_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim // 2),
        )
        self.quality_encoder = nn.Sequential(
            nn.Linear(quality_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim // 2),
        )
        self.state_dim = state_dim

    def forward(
        self, privacy_signal: torch.Tensor, quality_signal: torch.Tensor
    ) -> torch.Tensor:
        """Encode privacy and quality signals into a state vector.

        Args:
            privacy_signal: [batch, privacy_dim] privacy detection logits.
            quality_signal: [batch, quality_dim] quality/confidence score.

        Returns:
            [batch, state_dim] state vector.
        """
        e_p = self.privacy_encoder(privacy_signal)
        e_q = self.quality_encoder(quality_signal)
        return torch.cat([e_p, e_q], dim=-1)


class PolicyNetwork(nn.Module):
    """Policy network pi_theta for the deferral decision.

    Takes the state s_t and outputs a probability distribution over
    three actions: {a1=local, a2=defer, a3=abstain}.
    """

    def __init__(self, state_dim: int = 128, hidden_dim: int = 128, num_actions: int = 3):
        """Initialize the policy network.

        Args:
            state_dim: Dimension of the input state.
            hidden_dim: Hidden layer dimension.
            num_actions: Number of possible actions (default 3).
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.num_actions = num_actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute action logits.

        Args:
            state: [batch, state_dim] state vector.

        Returns:
            [batch, num_actions] action logits.
        """
        return self.network(state)

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities.

        Args:
            state: [batch, state_dim] state vector.

        Returns:
            [batch, num_actions] action probabilities.
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy.

        Args:
            state: [batch, state_dim] state vector.
            deterministic: If True, select the action with highest probability.

        Returns:
            Tuple of (action, log_prob, entropy).
        """
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_action(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the log probability and entropy of a given action.

        Args:
            state: [batch, state_dim] state vector.
            action: [batch] action indices.

        Returns:
            Tuple of (log_prob, entropy).
        """
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Value network V_phi for PPO advantage estimation.

    Estimates the expected return from a given state, used to compute
    the generalized advantage estimate (GAE) for PPO training.
    """

    def __init__(self, state_dim: int = 128, hidden_dim: int = 128):
        """Initialize the value network.

        Args:
            state_dim: Dimension of the input state.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate the value of a state.

        Args:
            state: [batch, state_dim] state vector.

        Returns:
            [batch, 1] estimated value.
        """
        return self.network(state)
