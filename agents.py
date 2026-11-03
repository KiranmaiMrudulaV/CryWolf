"""Agents: a fixed-threshold baseline and a tabular Q-learning agent."""
from __future__ import annotations

import numpy as np

from env import ALERT, WAIT


class ThresholdAgent:
    """Baseline: alert as soon as the mean of the last k scores crosses a threshold."""

    def __init__(self, threshold: float = 0.5, k: int = 3):
        self.threshold, self.k = threshold, k

    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        return ALERT if obs[-self.k:].mean() >= self.threshold else WAIT


class QLearningAgent:
    """Tabular Q-learning over a small discretized state.

    State = (bucket of the latest score, bucket of the 5-frame mean,
             number of the last 5 frames above 0.45).
    That keeps the table small (10 x 10 x 6 x 2 actions) so it trains in
    seconds with plain NumPy, while still letting the agent tell a
    single spike apart from a sustained rise.
    """

    def __init__(self, alpha=0.1, gamma=0.98, eps_start=1.0, eps_end=0.02,
                 eps_decay_episodes=30_000, seed=0):
        self.q = np.zeros((10, 10, 6, 2))
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_episodes
        self.eps = eps_start
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def state(obs: np.ndarray) -> tuple[int, int, int]:
        last = min(int(obs[-1] * 10), 9)
        mean = min(int(obs.mean() * 10), 9)
        high = int((obs > 0.45).sum())
        return last, mean, high

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(2))
        return int(np.argmax(self.q[self.state(obs)]))

    def update(self, obs, action, reward, next_obs, done):
        s, s2 = self.state(obs), self.state(next_obs)
        target = reward if done else reward + self.gamma * self.q[s2].max()
        self.q[s + (action,)] += self.alpha * (target - self.q[s + (action,)])

    def decay(self, episode: int):
        frac = min(1.0, episode / self.eps_decay)
        self.eps = self.eps_start + frac * (self.eps_end - self.eps_start)
