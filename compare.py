"""
Compare Models

Watch and compare DQN vs Rainbow at the same training milestone.

Usage:
  ./venv/bin/python compare.py --episodes 1000
  ./venv/bin/python compare.py --episodes 500 --games 5
  ./venv/bin/python compare.py --list  # Show available models
"""
import os
import sys
import time
import glob
import torch
from wrappers import make_env


def list_models():
    """List all available milestone models."""
    print("\nAvailable models:")
    print("-" * 50)

    # Find DQN models
    dqn_models = sorted(glob.glob('models/dqn_ep*.pth'))
    print("\nDQN Models:")
    if dqn_models:
        for m in dqn_models:
            info = get_model_info(m)
            print(f"  {m} - {info}")
    else:
        print("  (none found)")

    # Find Rainbow models
    rainbow_models = sorted(glob.glob('models/rainbow_ep*.pth'))
    print("\nRainbow Models:")
    if rainbow_models:
        for m in rainbow_models:
            info = get_model_info(m)
            print(f"  {m} - {info}")
    else:
        print("  (none found)")

    print()


def get_model_info(path):
    """Get info about a model file."""
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        episodes = checkpoint.get('episodes_done', 'unknown')
        model_type = checkpoint.get('model_type', 'unknown')
        return f"episodes: {episodes}, type: {model_type}"
    except Exception as e:
        return f"error: {e}"


def load_agent(model_path):
    """Load the appropriate agent type based on model file."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_type = checkpoint.get('model_type', 'dqn')

    if model_type == 'rainbow':
        from rainbow_agent import RainbowAgent
        agent = RainbowAgent(num_actions=9)
    else:
        from dqn_agent import DQNAgent
        agent = DQNAgent(num_actions=9)

    agent.load(model_path)
    return agent, model_type


def play_game(env, agent):
    """Play one game, return score."""
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(0.015)

    return total_reward


def compare(episodes: int, num_games: int = 3):
    """Compare DQN and Rainbow at the same episode count."""
    dqn_path = f'models/dqn_ep{episodes}.pth'
    rainbow_path = f'models/rainbow_ep{episodes}.pth'

    # Check what's available
    has_dqn = os.path.exists(dqn_path)
    has_rainbow = os.path.exists(rainbow_path)

    if not has_dqn and not has_rainbow:
        print(f"No models found at {episodes} episodes.")
        print("Run with --list to see available models.")
        return

    env = make_env(render_mode='human')

    results = {}

    # Test DQN
    if has_dqn:
        print(f"\n{'='*50}")
        print(f"DQN @ {episodes} episodes")
        print('='*50)
        agent, _ = load_agent(dqn_path)
        dqn_scores = []
        for game in range(1, num_games + 1):
            print(f"\nGame {game}/{num_games}...")
            score = play_game(env, agent)
            dqn_scores.append(score)
            print(f"Score: {score}")
        results['dqn'] = dqn_scores
        print(f"\nDQN Average: {sum(dqn_scores)/len(dqn_scores):.0f}")

    # Test Rainbow
    if has_rainbow:
        print(f"\n{'='*50}")
        print(f"Rainbow @ {episodes} episodes")
        print('='*50)
        agent, _ = load_agent(rainbow_path)
        rainbow_scores = []
        for game in range(1, num_games + 1):
            print(f"\nGame {game}/{num_games}...")
            score = play_game(env, agent)
            rainbow_scores.append(score)
            print(f"Score: {score}")
        results['rainbow'] = rainbow_scores
        print(f"\nRainbow Average: {sum(rainbow_scores)/len(rainbow_scores):.0f}")

    env.close()

    # Summary
    if has_dqn and has_rainbow:
        print(f"\n{'='*50}")
        print(f"COMPARISON @ {episodes} episodes")
        print('='*50)
        dqn_avg = sum(results['dqn']) / len(results['dqn'])
        rainbow_avg = sum(results['rainbow']) / len(results['rainbow'])
        print(f"DQN Average:     {dqn_avg:>6.0f}")
        print(f"Rainbow Average: {rainbow_avg:>6.0f}")
        diff = rainbow_avg - dqn_avg
        pct = (diff / dqn_avg * 100) if dqn_avg > 0 else 0
        if diff > 0:
            print(f"Rainbow is {diff:.0f} points better ({pct:+.1f}%)")
        elif diff < 0:
            print(f"DQN is {-diff:.0f} points better ({-pct:+.1f}%)")
        else:
            print("It's a tie!")


def watch_single(model_path: str, num_games: int = 3):
    """Watch a specific model play."""
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    env = make_env(render_mode='human')
    agent, model_type = load_agent(model_path)

    info = get_model_info(model_path)
    print(f"\nWatching: {model_path}")
    print(f"Info: {info}")
    print("-" * 40)

    scores = []
    for game in range(1, num_games + 1):
        print(f"\nGame {game}/{num_games}...")
        score = play_game(env, agent)
        scores.append(score)
        print(f"Score: {score}")

    print(f"\nAverage: {sum(scores)/len(scores):.0f}")
    env.close()


if __name__ == "__main__":
    if '--list' in sys.argv:
        list_models()
    elif '--episodes' in sys.argv:
        idx = sys.argv.index('--episodes')
        episodes = int(sys.argv[idx + 1])
        num_games = 3
        if '--games' in sys.argv:
            idx = sys.argv.index('--games')
            num_games = int(sys.argv[idx + 1])
        compare(episodes, num_games)
    elif '--model' in sys.argv:
        idx = sys.argv.index('--model')
        model_path = sys.argv[idx + 1]
        num_games = 3
        if '--games' in sys.argv:
            idx = sys.argv.index('--games')
            num_games = int(sys.argv[idx + 1])
        watch_single(model_path, num_games)
    else:
        print(__doc__)
        print("\nExamples:")
        print("  ./venv/bin/python compare.py --list")
        print("  ./venv/bin/python compare.py --episodes 1000")
        print("  ./venv/bin/python compare.py --episodes 1000 --games 5")
        print("  ./venv/bin/python compare.py --model models/dqn_ep500.pth")
