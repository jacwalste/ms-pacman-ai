"""
Training Loop for Rainbow DQN

Same structure as train.py but uses the Rainbow agent with all improvements.
No epsilon-greedy needed - noisy networks handle exploration!
"""
import os
import time
import glob
import numpy as np
from collections import deque

from wrappers import make_env
from rainbow_agent import RainbowAgent


def train(
    num_episodes: int = 10000,
    demo_every: int = 50,
    save_every: int = 100,
    min_buffer_size: int = 10_000,
    resume_from: str = None,
):
    """Train the Rainbow agent."""
    # Create environments
    env = make_env(render_mode=None)
    demo_env = make_env(render_mode='human')

    # Create Rainbow agent (no epsilon parameters - noisy nets handle exploration!)
    agent = RainbowAgent(
        num_actions=env.action_space.n,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        buffer_capacity=100_000,
        batch_size=32,
        gamma=0.99,
        lr=6.25e-5,
        n_steps=3,
        target_update_freq=1000,
        alpha=0.5,
        beta_start=0.4,
        beta_frames=100_000,
    )

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        agent.load(resume_from)
        print(f"Resumed from {resume_from} (steps: {agent.steps_done})")

    # Track progress
    recent_rewards = deque(maxlen=100)
    best_avg_reward = float('-inf')

    # Create models directory
    os.makedirs('models', exist_ok=True)

    print("=" * 60)
    print("RAINBOW DQN - Training with ALL the improvements!")
    print("=" * 60)
    print("Components active:")
    print("  [x] Double DQN (reduce overestimation)")
    print("  [x] Prioritized Replay (learn from surprises)")
    print("  [x] Dueling Network (separate value/advantage)")
    print("  [x] Distributional RL (predict return distribution)")
    print("  [x] Noisy Networks (learned exploration)")
    print("  [x] N-step Returns (look 3 steps ahead)")
    print("=" * 60)

    print("\nFilling replay buffer...")
    state, _ = env.reset()
    for step in range(min_buffer_size):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_experience(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]
        if (step + 1) % 2000 == 0:
            print(f"  Buffer: {step + 1}/{min_buffer_size}")

    print(f"Buffer filled with {len(agent.buffer)} experiences. Starting training!")
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

        avg_loss = np.mean(episode_loss) if episode_loss else 0
        print(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:6.0f} | "
            f"Avg(100): {avg_reward:6.1f} | "
            f"Loss: {avg_loss:.4f}"
        )

        if avg_reward > best_avg_reward and episode >= 100:
            best_avg_reward = avg_reward
            agent.save('models/rainbow_best.pth')
            print(f"  -> New best average! Saved model.")

        if episode % save_every == 0:
            agent.save(f'models/rainbow_checkpoint_ep{episode}.pth')

        if episode % demo_every == 0:
            print("\nDemo time! Watch Rainbow play...")
            demo_reward = play_demo(demo_env, agent)
            print(f"Demo reward: {demo_reward}\n")

    env.close()
    demo_env.close()
    agent.save('models/rainbow_final.pth')
    print("Training complete!")


def play_demo(env, agent):
    """Play one episode with rendering."""
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(0.01)

    return total_reward


def find_latest_rainbow_model():
    """Find most recent rainbow model."""
    models = glob.glob('models/rainbow_*.pth')
    if not models:
        return None
    return max(models, key=os.path.getmtime)


if __name__ == "__main__":
    import sys

    resume_path = None
    if '--resume' in sys.argv:
        resume_path = find_latest_rainbow_model()
        if resume_path:
            print(f"Found most recent model: {resume_path}")
        else:
            print("No Rainbow models found, starting fresh.")

    train(
        num_episodes=10000,
        demo_every=50,
        save_every=100,
        resume_from=resume_path,
    )
