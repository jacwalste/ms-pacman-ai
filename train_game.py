#!/usr/bin/env python3
"""
Train Rainbow on any Atari game.

Usage:
    python train_game.py --game Pong --episodes 5000
    python train_game.py --game Breakout --episodes 5000 --resume
    python train_game.py --game MsPacman --episodes 10000 --resume
"""
import os
import sys
import argparse
import numpy as np
from collections import deque

from wrappers import make_env
from rainbow_agent import RainbowAgent


def safe_save(agent, path, episodes, game):
    """Save model with error handling - don't crash if save fails."""
    try:
        agent.save(path, episodes=episodes, game=game)
        return True
    except Exception as e:
        print(f"  WARNING: Failed to save {path}: {e}", flush=True)
        return False


def train(game: str, num_episodes: int = 10000, resume: bool = False):
    """Train Rainbow on specified game."""
    print(f"=" * 60)
    print(f"TRAINING RAINBOW ON {game.upper()}")
    print(f"=" * 60)

    env = make_env(render_mode=None, game=game)
    model_dir = f'models/{game.lower()}'
    os.makedirs(model_dir, exist_ok=True)

    print(f"Action space: {env.action_space.n} actions")
    print(f"Training for {num_episodes} episodes")

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

    start_episode = 1
    best_avg_reward = float('-inf')

    # Resume from checkpoint if requested
    checkpoint_path = f'{model_dir}/checkpoint.pth'
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}...")
        agent.load(checkpoint_path)
        start_episode = agent.episodes_done + 1
        print(f"Resuming from episode {start_episode}")
    elif resume:
        print(f"No checkpoint found at {checkpoint_path}, starting fresh")

    print()

    recent_rewards = deque(maxlen=100)
    milestones = {500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000}

    print(f"{game}: Filling replay buffer...")
    state, _ = env.reset()
    for step in range(10_000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_experience(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]
        if (step + 1) % 2000 == 0:
            print(f"  Buffer: {step + 1}/10000", flush=True)

    print(f"{game}: Buffer filled. Starting training!")
    print("-" * 60, flush=True)

    for episode in range(start_episode, num_episodes + 1):
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
            f"Loss: {avg_loss:.4f}",
            flush=True
        )

        if avg_reward > best_avg_reward and episode >= 100:
            best_avg_reward = avg_reward
            if safe_save(agent, f'{model_dir}/best.pth', episode, game):
                print(f"  -> New best! Saved {model_dir}/best.pth", flush=True)

        if episode in milestones:
            if safe_save(agent, f'{model_dir}/ep{episode}.pth', episode, game):
                print(f"  -> Milestone! Saved {model_dir}/ep{episode}.pth", flush=True)

        # Save checkpoint every 500 episodes
        if episode % 500 == 0:
            safe_save(agent, f'{model_dir}/checkpoint.pth', episode, game)

    env.close()
    safe_save(agent, f'{model_dir}/final.pth', episode, game)
    print(f"{game}: Training complete! Total episodes: {episode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Rainbow on Atari games')
    parser.add_argument('--game', type=str, default='MsPacman',
                        help='Game to train on (e.g., MsPacman, Pong, Breakout)')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of episodes to train')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    args = parser.parse_args()

    train(args.game, args.episodes, args.resume)
