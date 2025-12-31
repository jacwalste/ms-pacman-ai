"""
DQN Agent

Combines the Q-Network and Replay Buffer to create an agent that can:
1. Select actions (epsilon-greedy exploration)
2. Store experiences
3. Learn from batches of experiences
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer
from q_network import QNetwork


class DQNAgent:
    def __init__(
        self,
        num_actions: int,
        buffer_capacity: int = 100_000,
        batch_size: int = 32,
        gamma: float = 0.99,
        lr: float = 1e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 100_000,
        target_update_freq: int = 1000,
        device: str = None,
    ):
        """
        Initialize the DQN Agent.

        Args:
            num_actions: Number of possible actions
            buffer_capacity: Max experiences in replay buffer
            batch_size: Number of experiences per training batch
            gamma: Discount factor (how much to value future rewards)
            lr: Learning rate for optimizer
            epsilon_start: Initial exploration rate (100% random)
            epsilon_end: Final exploration rate (10% random)
            epsilon_decay: Steps to decay epsilon over
            target_update_freq: Steps between target network updates
            device: 'cuda', 'mps', or 'cpu'
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        # Epsilon schedule for exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Networks: policy (main) and target (stable copy)
        self.policy_net = QNetwork(num_actions).to(self.device)
        self.target_net = QNetwork(num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss - less sensitive to outliers

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        # Track total episodes for model comparison
        self.episodes_done = 0

    @property
    def epsilon(self):
        """Current exploration rate (decays over time)."""
        # Linear decay from epsilon_start to epsilon_end
        progress = min(1.0, self.steps_done / self.epsilon_decay)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current game state (4 stacked frames)
            training: If False, always use best action (no exploration)

        Returns:
            Action index (0 to num_actions-1)
        """
        # Explore: random action
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        # Exploit: use Q-network to pick best action
        with torch.no_grad():
            # Add batch dimension, convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store an experience in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Sample a batch from the buffer and perform one gradient update.

        Returns:
            Loss value, or None if buffer doesn't have enough samples
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) for the actions we took
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: policy net picks best action, target net evaluates it
        # This reduces overestimation bias from regular DQN
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Update step counter
        self.steps_done += 1

        # Periodically update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str, episodes: int = None):
        """Save the model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': episodes if episodes else self.episodes_done,
            'model_type': 'dqn',
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint.get('episodes_done', 0)


# Quick test
if __name__ == "__main__":
    agent = DQNAgent(num_actions=9)

    # Fake state (4 frames of 84x84)
    fake_state = np.random.rand(4, 84, 84).astype(np.float32)

    # Test action selection
    action = agent.select_action(fake_state)
    print(f"Selected action: {action}")
    print(f"Current epsilon: {agent.epsilon:.3f}")

    # Store some fake experiences
    for _ in range(100):
        fake_next = np.random.rand(4, 84, 84).astype(np.float32)
        agent.store_experience(fake_state, action, 1.0, fake_next, False)

    # Test learning
    loss = agent.learn()
    print(f"Loss: {loss:.4f}")
    print(f"Steps done: {agent.steps_done}")
