"""
Replay Buffer for DQN

Stores experiences (state, action, reward, next_state, done) and allows
random sampling for training. This breaks correlation between consecutive
experiences and allows learning from past experiences multiple times.
"""
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of experiences to store.
                      When full, oldest experiences are removed.
        """
        # deque automatically removes oldest items when capacity is exceeded
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store an experience in the buffer.

        Args:
            state: The screen/observation before the action
            action: The action taken (0-8 for Ms. Pac-Man)
            reward: The reward received
            next_state: The screen/observation after the action
            done: Whether the game ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Randomly sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Lists of (states, actions, rewards, next_states, dones)
        """
        # Randomly pick batch_size experiences from the buffer
        batch = random.sample(self.buffer, batch_size)

        # Unzip into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current number of experiences stored."""
        return len(self.buffer)