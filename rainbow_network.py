"""
Rainbow Network

Combines three key improvements over vanilla DQN:

1. DUELING ARCHITECTURE
   Instead of: Q(s,a) directly
   We compute: Q(s,a) = V(s) + A(s,a) - mean(A)

   - V(s) = "How good is this state overall?"
   - A(s,a) = "How much better is action a than average?"

   This helps because many states have similar values regardless of action
   (e.g., when no ghosts are nearby, all moves are roughly equal).

2. DISTRIBUTIONAL RL (C51)
   Instead of predicting a single expected return, we predict a DISTRIBUTION
   over possible returns using 51 "atoms" (discrete buckets).

   Why? Returns are uncertain! "Expected value = 100" could mean:
   - Always get 100 (low variance)
   - Sometimes 0, sometimes 200 (high variance)

   Knowing the distribution helps make better decisions.

3. NOISY LAYERS
   Replace final linear layers with noisy versions for learned exploration.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from noisy_linear import NoisyLinear


class RainbowNetwork(nn.Module):
    """
    Rainbow DQN Network: Dueling + Distributional + Noisy

    Instead of outputting Q-values directly, outputs a distribution
    over returns for each action.
    """

    def __init__(
        self,
        num_actions: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        """
        Args:
            num_actions: Number of possible actions
            num_atoms: Number of atoms for distributional RL (C51 uses 51)
            v_min: Minimum possible return
            v_max: Maximum possible return
        """
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support: the discrete values our distribution is over
        # e.g., [-10, -9.6, -9.2, ..., 9.6, 10] for 51 atoms
        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, num_atoms)
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Convolutional layers (same as before)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = 64 * 7 * 7  # 3136

        # DUELING: Two separate streams after conv layers
        # Value stream: V(s) - how good is this state?
        self.value_stream = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_atoms),  # Output: distribution over returns
        )

        # Advantage stream: A(s,a) - how much better is each action?
        self.advantage_stream = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * num_atoms),  # Distribution per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Batch of states, shape (batch, 4, 84, 84)

        Returns:
            Log probabilities over atoms for each action
            Shape: (batch, num_actions, num_atoms)
        """
        batch_size = x.shape[0]

        # Shared conv features
        features = self.conv(x)
        features = features.view(batch_size, -1)

        # Dueling streams
        value = self.value_stream(features)  # (batch, num_atoms)
        advantage = self.advantage_stream(features)  # (batch, num_actions * num_atoms)

        # Reshape advantage
        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        # Combine: Q = V + A - mean(A)
        # This is the dueling combination, applied to distributions
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Softmax to get probabilities (over atoms, for each action)
        q_dist = F.log_softmax(q_dist, dim=2)

        return q_dist

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values by computing expected value of each distribution.

        Args:
            x: Batch of states

        Returns:
            Q-values, shape (batch, num_actions)
        """
        log_probs = self.forward(x)
        probs = log_probs.exp()
        # Q = sum(prob * support) = expected value
        q_values = (probs * self.support).sum(dim=2)
        return q_values

    def reset_noise(self):
        """Reset noise in all noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# Test
if __name__ == "__main__":
    net = RainbowNetwork(num_actions=9)

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    x = torch.randn(4, 4, 84, 84)  # Batch of 4

    # Get distribution
    log_probs = net(x)
    print(f"Log probs shape: {log_probs.shape}")  # (4, 9, 51)

    # Get Q-values
    q_values = net.get_q_values(x)
    print(f"Q-values shape: {q_values.shape}")  # (4, 9)
    print(f"Best actions: {q_values.argmax(dim=1)}")

    # Verify it's a valid probability distribution
    probs = log_probs.exp()
    print(f"Probs sum to 1: {probs.sum(dim=2).allclose(torch.ones(4, 9))}")
