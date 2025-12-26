"""Watch a trained agent play Ms. Pac-Man."""
import sys
import time
from wrappers import make_env
from dqn_agent import DQNAgent


def watch(model_path: str = 'models/best_model.pth', num_games: int = 3):
    """Watch the trained agent play."""
    env = make_env(render_mode='human')
    agent = DQNAgent(num_actions=env.action_space.n)

    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No model found at {model_path}. Train first!")
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
    model = sys.argv[1] if len(sys.argv) > 1 else 'models/best_model.pth'
    watch(model)
