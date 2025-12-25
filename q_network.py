"""
Q-Network for DQN

A convolutional neural network that takes game frames as input
and outputs Q-values for each possible action.

Architecture (based on the original DQN paper):
- Input: 4 stacked grayscale frames (84x84x4)
- Conv layers extract visual features (walls, ghosts, pellets)
- Fully connected layers compute Q-values for each action
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, num_actions: int):
        """
        Initialize the Q-Network.

        Args:
            num_actions: Number of possible actions (9 for Ms. Pac-Man)
        """
        super().__init__()

        # Convolutional layers - extract features from the image
        # Input shape: (batch, 4 frames, 84 height, 84 width)
        self.conv = nn.Sequential(
            # Layer 1: 32 filters, 8x8 kernel, stride 4
            # Output: (batch, 32, 20, 20)
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Layer 2: 64 filters, 4x4 kernel, stride 2
            # Output: (batch, 64, 9, 9)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Layer 3: 64 filters, 3x3 kernel, stride 1
            # Output: (batch, 64, 7, 7)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Fully connected layers - compute Q-values from features
        # 64 * 7 * 7 = 3136 flattened features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),  # Output: Q-value for each action
        )

    def forward(self, x):
        """
        Forward pass: image → Q-values

        Args:
            x: Batch of stacked frames, shape (batch, 4, 84, 84)
               Values should be floats in range [0, 1]

        Returns:
            Q-values for each action, shape (batch, num_actions)
        """
        features = self.conv(x)
        q_values = self.fc(features)
        return q_values