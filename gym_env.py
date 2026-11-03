"""Gymnasium wrapper around AlertEnv so SB3 algorithms can train on the raw scores."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env import AlertEnv, EnvConfig


class AlertGymEnv(gym.Env):
    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        super().__init__()
        self.env = AlertEnv(config, seed=seed)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.env.window,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        return self.env.reset(), {}

    def step(self, action):
        obs, r, done, info = self.env.step(int(action))
        return obs, r, done, False, info
