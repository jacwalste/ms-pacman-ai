"""
Rainbow DQN Agent

Combines ALL the improvements:
1. Double DQN - policy selects, target evaluates (reduces overestimation)
2. Prioritized Experience Replay - learn more from surprising experiences
3. Dueling Networks - separate value and advantage streams
4. Distributional RL (C51) - predict distribution, not just mean
5. Noisy Networks - learned exploration (no epsilon-greedy!)
6. N-step Returns - look further ahead for better credit assignment

This is the full Rainbow algorithm from DeepMind's 2017 paper.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from prioritized_replay_buffer import PrioritizedReplayBuffer
from rainbow_network import RainbowNetwork


class NStepBuffer:
    """
    Accumulates N transitions before adding to replay buffer.

    N-step return: r_1 + γr_2 + γ²r_3 + ... + γⁿ⁻¹r_n + γⁿV(s_n)

    Instead of learning from immediate reward only, we use the actual
    rewards from N steps, which gives clearer learning signal.
    """

    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=n_steps)

    def add(self, state, action, reward, next_state, done) -> tuple:
        """
        Add transition. Returns complete n-step transition if ready.
        """
        self.buffer.append((state, action, reward, next_state, done))

        # Not enough transitions yet
        if len(self.buffer) < self.n_steps:
            return None

        # Compute n-step return
        n_step_return = 0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_step_return += (self.gamma ** i) * r
            if d:  # Episode ended early
                break

        # Get first state/action and last next_state
        first_state, first_action, _, _, _ = self.buffer[0]
        _, _, _, last_next_state, last_done = self.buffer[-1]

        return (first_state, first_action, n_step_return, last_next_state, last_done)

    def flush(self):
        """Flush remaining transitions at episode end."""
        transitions = []
        while len(self.buffer) > 0:
            # Compute partial n-step return with remaining transitions
            n_step_return = 0
            for i, (_, _, r, _, d) in enumerate(self.buffer):
                n_step_return += (self.gamma ** i) * r
                if d:
                    break

            first_state, first_action, _, _, _ = self.buffer[0]
            _, _, _, last_next_state, last_done = list(self.buffer)[-1]

            transitions.append(
                (first_state, first_action, n_step_return, last_next_state, last_done)
            )
            self.buffer.popleft()

        return transitions

    def reset(self):
        self.buffer.clear()


class RainbowAgent:
    """
    Full Rainbow DQN Agent.
    """

    def __init__(
        self,
        num_actions: int,
        # Distributional
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        # Replay
        buffer_capacity: int = 100_000,
        batch_size: int = 32,
        # Learning
        gamma: float = 0.99,
        lr: float = 6.25e-5,  # Rainbow uses smaller LR
        n_steps: int = 3,
        # Updates
        target_update_freq: int = 1000,
        # Priority replay
        alpha: float = 0.5,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        # Device
        device: str = None,
    ):
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
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_steps = n_steps
        self.target_update_freq = target_update_freq

        # Support for distributional RL
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Networks
        self.policy_net = RainbowNetwork(
            num_actions, num_atoms, v_min, v_max
        ).to(self.device)
        self.target_net = RainbowNetwork(
            num_actions, num_atoms, v_min, v_max
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1.5e-4)

        # Prioritized replay buffer
        self.buffer = PrioritizedReplayBuffer(
            buffer_capacity, alpha=alpha, beta_start=beta_start, beta_frames=beta_frames
        )

        # N-step buffer
        self.n_step_buffer = NStepBuffer(n_steps, gamma)

        self.steps_done = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using noisy network (no epsilon-greedy!).

        The noise in the network provides exploration during training.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net.get_q_values(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience with n-step processing."""
        # Add to n-step buffer
        transition = self.n_step_buffer.add(state, action, reward, next_state, done)

        if transition is not None:
            self.buffer.push(*transition)

        # Flush remaining at episode end
        if done:
            for trans in self.n_step_buffer.flush():
                self.buffer.push(*trans)
            self.n_step_buffer.reset()

    def learn(self):
        """
        Sample prioritized batch and perform distributional RL update.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample from prioritized buffer
        (states, actions, rewards, next_states, dones,
         indices, weights) = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Get current distribution
        log_probs = self.policy_net(states)
        log_probs_a = log_probs[range(self.batch_size), actions]  # (batch, atoms)

        with torch.no_grad():
            # Double DQN: policy net selects action
            next_q_values = self.policy_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=1)

            # Target net evaluates
            next_log_probs = self.target_net(next_states)
            next_probs = next_log_probs.exp()
            next_probs_a = next_probs[range(self.batch_size), next_actions]

            # Compute projected distribution (Bellman update for distributions)
            # T_z = r + γⁿ * z  (shifted support)
            gamma_n = self.gamma ** self.n_steps
            t_z = rewards.unsqueeze(1) + gamma_n * self.support * (1 - dones.unsqueeze(1))
            t_z = t_z.clamp(self.v_min, self.v_max)

            # Project onto fixed support
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Handle edge case where l == u
            l[(u > 0) & (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) & (l == u)] += 1

            # Distribute probability
            target_probs = torch.zeros_like(next_probs_a)
            offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms,
                                    self.batch_size).long().unsqueeze(1).to(self.device)

            target_probs.view(-1).index_add_(
                0, (l + offset).view(-1), (next_probs_a * (u.float() - b)).view(-1)
            )
            target_probs.view(-1).index_add_(
                0, (u + offset).view(-1), (next_probs_a * (b - l.float())).view(-1)
            )

        # Cross-entropy loss (KL divergence)
        loss = -(target_probs * log_probs_a).sum(dim=1)

        # TD error for priority update (use loss as proxy)
        td_errors = loss.detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        # Weighted loss for importance sampling correction
        loss = (loss * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Reset noise for next step
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        self.steps_done += 1

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']


# Test
if __name__ == "__main__":
    agent = RainbowAgent(num_actions=9)

    # Fake state
    fake_state = np.random.rand(4, 84, 84).astype(np.float32)

    # Test action selection (no epsilon!)
    action = agent.select_action(fake_state)
    print(f"Selected action: {action}")

    # Store some experiences
    for i in range(100):
        fake_next = np.random.rand(4, 84, 84).astype(np.float32)
        done = (i == 99)  # End episode at 100
        agent.store_experience(fake_state, action, 1.0, fake_next, done)
        fake_state = fake_next

    print(f"Buffer size: {len(agent.buffer)}")

    # Test learning
    loss = agent.learn()
    print(f"Loss: {loss:.4f}" if loss else "Not enough samples yet")
