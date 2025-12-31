#!/usr/bin/env python3
"""
Train Both DQN and Rainbow Concurrently

Runs both training scripts in parallel so you can compare them fairly.
Both will train for the same number of episodes.

Usage:
  ./venv/bin/python train_both.py
  ./venv/bin/python train_both.py --episodes 5000
"""
import subprocess
import sys
import os
import time
from datetime import datetime


def train_both(num_episodes: int = 10000):
    """Run both DQN and Rainbow training in parallel."""

    print("=" * 60)
    print("TRAINING BOTH DQN AND RAINBOW")
    print("=" * 60)
    print(f"Episodes: {num_episodes} each")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    print("This will run both trainings in parallel.")
    print("Each will save milestones at: 500, 1000, 2000, 3000, 5000, 7500, 10000")
    print()
    print("Output is logged to:")
    print("  - logs/dqn_training.log")
    print("  - logs/rainbow_training.log")
    print()
    print("You can monitor progress with:")
    print("  tail -f logs/dqn_training.log")
    print("  tail -f logs/rainbow_training.log")
    print()

    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Update num_episodes in both training scripts temporarily
    # We'll pass it via environment variable instead
    env = os.environ.copy()
    env['TRAIN_EPISODES'] = str(num_episodes)

    # Start DQN training
    print("Starting DQN training...")
    with open('logs/dqn_training.log', 'w') as dqn_log:
        dqn_process = subprocess.Popen(
            [sys.executable, 'train_dqn_headless.py'],
            stdout=dqn_log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    # Start Rainbow training
    print("Starting Rainbow training...")
    with open('logs/rainbow_training.log', 'w') as rainbow_log:
        rainbow_process = subprocess.Popen(
            [sys.executable, 'train_rainbow_headless.py'],
            stdout=rainbow_log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print()
    print("Both trainings running in background!")
    print(f"DQN PID: {dqn_process.pid}")
    print(f"Rainbow PID: {rainbow_process.pid}")
    print()
    print("Waiting for both to complete...")
    print("(You can safely close this terminal - training will continue)")
    print()

    # Wait for both to complete
    try:
        while dqn_process.poll() is None or rainbow_process.poll() is None:
            dqn_status = "running" if dqn_process.poll() is None else "done"
            rainbow_status = "running" if rainbow_process.poll() is None else "done"

            # Check latest episode from logs
            dqn_ep = get_latest_episode('logs/dqn_training.log')
            rainbow_ep = get_latest_episode('logs/rainbow_training.log')

            print(f"\r  DQN: {dqn_status} (ep {dqn_ep}) | Rainbow: {rainbow_status} (ep {rainbow_ep})    ", end='', flush=True)
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Training processes will continue in background.")
        print(f"To stop them: kill {dqn_process.pid} {rainbow_process.pid}")
        return

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Compare results with:")
    print("  ./venv/bin/python compare.py --list")
    print("  ./venv/bin/python compare.py --episodes 1000")


def get_latest_episode(log_file):
    """Parse log file to get latest episode number."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 'Episode' in line and '|' in line:
                    # Parse "Episode  123 | ..."
                    parts = line.split('|')[0]
                    ep = parts.replace('Episode', '').strip()
                    return ep
    except:
        pass
    return "0"


if __name__ == "__main__":
    episodes = 10000
    if '--episodes' in sys.argv:
        idx = sys.argv.index('--episodes')
        episodes = int(sys.argv[idx + 1])

    train_both(episodes)
