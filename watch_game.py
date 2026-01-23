#!/usr/bin/env python3
"""
Watch a trained Rainbow agent play any Atari game.

Usage:
    python watch_game.py --game Pong
    python watch_game.py --game Breakout --model models/breakout/ep1000.pth
    python watch_game.py --game MsPacman --games 5
"""
import os
import sys
import argparse
import time
import torch
import numpy as np

from wrappers import make_env
from rainbow_agent import RainbowAgent


def find_best_model(game: str) -> str:
    """Find the best available model for a game."""
    model_dir = f'models/{game.lower()}'

    # Priority order
    candidates = ['best.pth', 'final.pth', 'checkpoint.pth']

    for candidate in candidates:
        path = os.path.join(model_dir, candidate)
        if os.path.exists(path):
            return path

    # Look for milestone models
    milestones = [20000, 15000, 10000, 7500, 5000, 3000, 2000, 1000, 500]
    for ep in milestones:
        path = os.path.join(model_dir, f'ep{ep}.pth')
        if os.path.exists(path):
            return path

    return None


def watch(game: str, model_path: str = None, num_games: int = 3, delay: float = 0.02):
    """Watch the agent play."""
    # Find model if not specified
    if model_path is None:
        model_path = find_best_model(game)
        if model_path is None:
            print(f"No trained model found for {game}!")
            print(f"Train one with: python train_game.py --game {game}")
            sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    # Load checkpoint to get info
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    episodes_trained = checkpoint.get('episodes_done', '?')
    saved_game = checkpoint.get('game', game)
    num_actions = checkpoint.get('num_actions')

    print(f"=" * 50)
    print(f"WATCHING: {game}")
    print(f"=" * 50)
    print(f"Model: {model_path}")
    print(f"Trained for: {episodes_trained} episodes")
    if saved_game and saved_game != game:
        print(f"WARNING: Model was trained on {saved_game}, playing {game}")
    print(f"=" * 50)
    print()

    # Create environment
    env = make_env(render_mode='human', game=game)

    # Use num_actions from checkpoint if available, else from env
    if num_actions is None:
        num_actions = env.action_space.n

    # Create and load agent
    agent = RainbowAgent(num_actions=num_actions)
    agent.load(model_path)

    total_rewards = []

    for game_num in range(1, num_games + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        print(f"Game {game_num}/{num_games}...", end=' ', flush=True)

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            time.sleep(delay)

        total_rewards.append(episode_reward)
        print(f"Score: {episode_reward:.0f} ({steps} steps)")

    env.close()

    print()
    print(f"=" * 50)
    print(f"RESULTS: {num_games} games")
    print(f"=" * 50)
    print(f"Average: {np.mean(total_rewards):.1f}")
    print(f"Best:    {max(total_rewards):.0f}")
    print(f"Worst:   {min(total_rewards):.0f}")


def list_models():
    """List all available trained models."""
    print("Available models:")
    print("-" * 40)

    if not os.path.exists('models'):
        print("No models directory found.")
        return

    for game_dir in sorted(os.listdir('models')):
        game_path = os.path.join('models', game_dir)
        if os.path.isdir(game_path):
            models = [f for f in os.listdir(game_path) if f.endswith('.pth')]
            if models:
                print(f"\n{game_dir.upper()}:")
                for model in sorted(models):
                    path = os.path.join(game_path, model)
                    try:
                        ckpt = torch.load(path, map_location='cpu', weights_only=False)
                        eps = ckpt.get('episodes_done', '?')
                        print(f"  {model} ({eps} episodes)")
                    except:
                        print(f"  {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Watch Rainbow play Atari')
    parser.add_argument('--game', type=str, default='MsPacman',
                        help='Game to play (e.g., MsPacman, Pong, Breakout)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (auto-detects best if not specified)')
    parser.add_argument('--games', type=int, default=3,
                        help='Number of games to play')
    parser.add_argument('--list', action='store_true',
                        help='List all available models')
    parser.add_argument('--delay', type=float, default=0.02,
                        help='Delay between frames (seconds)')
    args = parser.parse_args()

    if args.list:
        list_models()
    else:
        watch(args.game, args.model, args.games, args.delay)
