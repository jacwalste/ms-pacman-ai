"""Watch a trained Rainbow agent play Ms. Pac-Man."""
import sys
import time
import glob
import os
from wrappers import make_env
from rainbow_agent import RainbowAgent


def watch(model_path: str = None, num_games: int = 3):
    """Watch the trained Rainbow agent play."""
    if model_path is None:
        # Find best rainbow model
        if os.path.exists('models/rainbow_best.pth'):
            model_path = 'models/rainbow_best.pth'
        else:
            models = glob.glob('models/rainbow_*.pth')
            if models:
                model_path = max(models, key=os.path.getmtime)
            else:
                print("No Rainbow models found. Train first with train_rainbow.py!")
                return

    env = make_env(render_mode='human')
    agent = RainbowAgent(num_actions=env.action_space.n)

    try:
        agent.load(model_path)
        print(f"Loaded Rainbow model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        return

    for game in range(1, num_games + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        print(f"\nGame {game}/{num_games}")

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.02)

        print(f"Score: {total_reward}")

    env.close()


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else None
    watch(model)
