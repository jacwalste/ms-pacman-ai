"""
Training Loop for DQN Ms. Pac-Man

Trains the agent by playing episodes and learning from experience.
Periodically shows a visual demo so you can watch progress.
"""
import os
import time
import numpy as np
from collections import deque

from wrappers import make_env
from dqn_agent import DQNAgent


def train(
    num_episodes: int = 1000,
    demo_every: int = 50,
    save_every: int = 100,
    min_buffer_size: int = 10_000,
    resume_from: str = None,
):
    """
    Train the DQN agent.

    Args:
        num_episodes: Total episodes to train
        demo_every: Play a visible demo every N episodes
        save_every: Save model every N episodes
        min_buffer_size: Fill buffer this much before learning
        resume_from: Path to model checkpoint to resume from
    """
    # Create environments
    env = make_env(render_mode=None)  # Fast, no rendering
    demo_env = make_env(render_mode='human')  # For watching

    # Create agent
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

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        agent.load(resume_from)
        print(f"Resumed from {resume_from} (steps: {agent.steps_done}, epsilon: {agent.epsilon:.3f})")

    # Track progress
    recent_rewards = deque(maxlen=100)
    best_avg_reward = float('-inf')

    # Create models directory
    os.makedirs('models', exist_ok=True)

    print("Filling replay buffer...")
    state, _ = env.reset()
    for _ in range(min_buffer_size):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_experience(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]

    print(f"Buffer filled with {len(agent.buffer)} experiences. Starting training!")
    print("-" * 60)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False

        while not done:
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store and learn
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            episode_reward += reward

        # Track progress
        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)

        # Print progress
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        print(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:6.0f} | "
            f"Avg(100): {avg_reward:6.1f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Loss: {avg_loss:.4f}"
        )

        # Save best model
        if avg_reward > best_avg_reward and episode >= 100:
            best_avg_reward = avg_reward
            agent.save('models/best_model.pth')
            print(f"  ↳ New best average! Saved model.")

        # Save periodic checkpoint
        if episode % save_every == 0:
            agent.save(f'models/checkpoint_ep{episode}.pth')

        # Play demo
        if episode % demo_every == 0:
            print("\n🎮 Demo time! Watch the AI play...")
            demo_reward = play_demo(demo_env, agent)
            print(f"Demo reward: {demo_reward}\n")

    env.close()
    demo_env.close()
    agent.save('models/final_model.pth')
    print("Training complete!")


def play_demo(env, agent):
    """Play one episode with rendering."""
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state, training=False)  # No exploration
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(0.01)  # Slow down slightly for visibility

    return total_reward


if __name__ == "__main__":
    import sys

    # Check for --resume flag
    resume_path = None
    if '--resume' in sys.argv:
        resume_path = 'models/final_model.pth'
        # Or use a specific checkpoint: models/checkpoint_ep500.pth

    train(
        num_episodes=500,  # Start with 500, increase for better results
        demo_every=50,     # Watch every 50 episodes
        save_every=100,
        resume_from=resume_path,
    )
