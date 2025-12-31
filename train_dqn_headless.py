"""
Headless DQN Training (no demo windows)

Used by train_both.py for background training.
"""
import os
import numpy as np
from collections import deque

from wrappers import make_env
from dqn_agent import DQNAgent


def train(num_episodes: int = 10000):
    """Train DQN without demo windows."""
    env = make_env(render_mode=None)

    agent = DQNAgent(
        num_actions=env.action_space.n,
        buffer_capacity=100_000,
        batch_size=32,
        gamma=0.99,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=100_000,
        target_update_freq=1000,
    )

    recent_rewards = deque(maxlen=100)
    best_avg_reward = float('-inf')
    milestones = {500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000}

    os.makedirs('models', exist_ok=True)

    print("DQN: Filling replay buffer...")
    state, _ = env.reset()
    for _ in range(10_000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_experience(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]

    print(f"DQN: Buffer filled. Starting training for {num_episodes} episodes!")
    print("-" * 60)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
            state = next_state
            episode_reward += reward

        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)
        agent.episodes_done = episode

        avg_loss = np.mean(episode_loss) if episode_loss else 0
        print(
            f"Episode {episode:5d} | "
            f"Reward: {episode_reward:6.0f} | "
            f"Avg(100): {avg_reward:7.1f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Loss: {avg_loss:.4f}"
        )

        if avg_reward > best_avg_reward and episode >= 100:
            best_avg_reward = avg_reward
            agent.save('models/dqn_best.pth', episodes=episode)
            print(f"  -> New best! Saved dqn_best.pth")

        if episode in milestones:
            agent.save(f'models/dqn_ep{episode}.pth', episodes=episode)
            print(f"  -> Milestone! Saved dqn_ep{episode}.pth")

    env.close()
    agent.save('models/dqn_final.pth', episodes=episode)
    print(f"DQN: Training complete! Total episodes: {episode}")


if __name__ == "__main__":
    episodes = int(os.environ.get('TRAIN_EPISODES', 10000))
    train(episodes)
