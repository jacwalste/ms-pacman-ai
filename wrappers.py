"""
Environment Wrappers

These modify the raw game environment to make it easier for the AI to learn:
1. Grayscale - Color doesn't matter much, reduces computation
2. Resize - Shrink from 210x160 to 84x84
3. Frame stack - Stack 4 frames so AI can see motion
4. Frame skip - Repeat each action for 4 frames (faster training)
"""
import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'  # Disable SDL audio

import gymnasium as gym
import numpy as np
from collections import deque


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    """Convert to grayscale and resize to 84x84."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        # Convert RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])

        # Resize from 210x160 to 84x84 using simple slicing/averaging
        # Quick and dirty resize - take every nth pixel and crop
        resized = gray[::2, ::2]  # 105x80
        resized = resized[10:94, :]  # 84x80 (crop top)

        # Pad width to 84
        padded = np.zeros((84, 84), dtype=np.uint8)
        padded[:, 2:82] = resized[:, :80]

        return padded.astype(np.uint8)


class FrameStackWrapper(gym.Wrapper):
    """Stack the last n frames together."""

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # Update observation space for stacked frames
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_frames, old_space.shape[0], old_space.shape[1]),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill all frames with initial observation
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked(), reward, terminated, truncated, info

    def _get_stacked(self):
        return np.array(self.frames)


class NormalizeWrapper(gym.ObservationWrapper):
    """Normalize pixel values from [0, 255] to [0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class FireResetWrapper(gym.Wrapper):
    """Press FIRE at the start of the game (required for some Atari games)."""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Action 1 is usually FIRE
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


def make_env(render_mode=None):
    """Create a wrapped Ms. Pac-Man environment."""
    import ale_py
    gym.register_envs(ale_py)

    env = gym.make(
        'ALE/MsPacman-v5',
        render_mode=render_mode,
        frameskip=4,  # Repeat each action 4 times
    )
    # Disable sound
    if hasattr(env.unwrapped, 'ale'):
        env.unwrapped.ale.setBool('sound', False)
    env = GrayscaleResizeWrapper(env)
    env = FrameStackWrapper(env, n_frames=4)
    env = NormalizeWrapper(env)
    return env


# Test
if __name__ == "__main__":
    env = make_env()

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")

    # Take a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i+1}: reward={reward}, done={term or trunc}")

    env.close()
    print("Wrappers working!")
