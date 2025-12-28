"""
Prioritized Experience Replay Buffer

Instead of sampling experiences uniformly at random, we sample based on
how "surprising" each experience was (measured by TD error).

Key idea: Learn more from experiences where we were very wrong.

Uses a Sum Tree data structure for efficient O(log n) sampling.
"""
import numpy as np


class SumTree:
    """
    Binary tree where each node is the sum of its children.
    Leaves hold priorities, internal nodes hold sums.

    Allows O(log n) sampling proportional to priority.

    Example tree (capacity=4):
                    [42]           <- root = sum of all priorities
                   /    \\
                [13]    [29]       <- internal nodes
                /  \\    /  \\
              [3] [10] [12] [17]   <- leaves (priorities)

    To sample: pick random number 0-42, traverse down
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree array
        self.data = np.zeros(capacity, dtype=object)  # Experience storage
        self.write_idx = 0  # Where to write next
        self.size = 0  # Current number of experiences

    def _propagate_up(self, idx: int, change: float):
        """Update parent nodes when a leaf changes."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate_up(parent, change)

    def _find_leaf(self, value: float) -> int:
        """Find leaf node for a given cumulative value."""
        idx = 0  # Start at root
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):  # Reached leaf level
                return idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

    def add(self, priority: float, data):
        """Add experience with given priority."""
        tree_idx = self.write_idx + self.capacity - 1  # Leaf index in tree
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        """Update priority of a leaf node."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate_up(tree_idx, change)

    def get(self, value: float):
        """Sample a leaf based on cumulative priority value."""
        leaf_idx = self._find_leaf(value)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        return self.tree[0]  # Root holds sum of all priorities


class PrioritizedReplayBuffer:
    """
    Replay buffer that samples experiences proportional to their TD error.

    Hyperparameters:
        alpha: How much prioritization to use (0 = uniform, 1 = full priority)
        beta: Importance sampling correction (annealed from beta_start to 1)
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.max_priority = 1.0  # Track max for new experiences

    @property
    def beta(self) -> float:
        """Anneal beta from beta_start to 1 over beta_frames."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def push(self, state, action, reward, next_state, done):
        """Add experience with max priority (will be updated after learning)."""
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size: int):
        """
        Sample batch proportional to priorities.

        Returns:
            states, actions, rewards, next_states, dones,
            indices (for updating), weights (for importance sampling)
        """
        batch = []
        indices = []
        priorities = []

        # Divide priority range into segments for stratified sampling
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            idx, priority, data = self.tree.get(value)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        probs = np.array(priorities) / self.tree.total_priority
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= weights.max()  # Normalize to max weight = 1

        self.frame += 1

        # Unzip experiences
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            states, actions, rewards, next_states, dones,
            indices, weights
        )

    def update_priorities(self, indices: list, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha  # Small epsilon to avoid 0
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.size


# Test
if __name__ == "__main__":
    buffer = PrioritizedReplayBuffer(capacity=1000)

    # Add experiences
    for i in range(100):
        state = np.random.rand(4, 84, 84)
        buffer.push(state, i % 9, float(i), state, False)

    print(f"Buffer size: {len(buffer)}")
    print(f"Beta: {buffer.beta:.3f}")

    # Sample
    states, actions, rewards, next_states, dones, indices, weights = buffer.sample(32)
    print(f"Sampled {len(states)} experiences")
    print(f"Weights range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Update priorities
    fake_td_errors = np.random.rand(32)
    buffer.update_priorities(indices, fake_td_errors)
    print("Priorities updated!")
