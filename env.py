"""Simulated noisy threat detector and an alert-decision environment.

Each episode is a stream of detector confidence scores (one per frame).
In some episodes a threat appears partway through, and from then on the
detector's scores drift upward. In the rest, the scores are background
noise, which sometimes spikes (a phone, a can, a controller...).

At every frame the agent chooses:
    0 = WAIT   (keep watching)
    1 = ALERT  (raise the alarm; ends the episode)

The agent has to learn when evidence is strong enough to alert:
too eager and it raises false alarms, too cautious and it alerts late
or misses the threat entirely.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WAIT, ALERT = 0, 1


@dataclass
class EnvConfig:
    episode_len: int = 60          # frames per episode
    threat_prob: float = 0.5       # chance an episode contains a threat
    noise_std: float = 0.12        # detector noise
    spike_prob: float = 0.04       # chance of a look-alike spike per frame
    threat_strength: float = 0.35  # how far a threat lifts the mean score
    # rewards
    r_detect: float = 10.0         # correct alert
    r_delay: float = 0.3           # cost per frame of delay after threat appears
    r_false_alarm: float = -10.0   # alert with no threat (or before it appears)
    r_miss: float = -20.0          # threat present, never alerted
    r_clear: float = 2.0           # no threat, correctly stayed quiet
    # harder-environment options (off by default, so the base env is unchanged)
    threat_strengths: tuple | None = None  # if set, each threat draws its strength from this tuple
    drift_std: float = 0.0         # per-frame random-walk step of the background level


class AlertEnv:
    """Minimal gym-style environment: reset() -> obs, step(a) -> obs, r, done, info."""

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.window = 5

    # ------------------------------------------------------------------ #
    def _score(self, t: int) -> float:
        base = 0.25
        if self.cfg.drift_std > 0:  # slowly wandering background level
            self.drift = float(np.clip(self.drift + self.rng.normal(0, self.cfg.drift_std), -0.1, 0.15))
            base += self.drift
        if self.has_threat and t >= self.threat_start:
            ramp = min(1.0, (t - self.threat_start + 1) / 6)  # signal builds up
            base += self.strength * ramp
        s = base + self.rng.normal(0, self.cfg.noise_std)
        if self.rng.random() < self.cfg.spike_prob:  # look-alike object
            s += self.rng.uniform(0.3, 0.6)
        return float(np.clip(s, 0.0, 1.0))

    def _obs(self) -> np.ndarray:
        recent = self.scores[-self.window:]
        pad = [0.0] * (self.window - len(recent))
        return np.array(pad + recent, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        self.t = 0
        self.has_threat = self.rng.random() < self.cfg.threat_prob
        self.threat_start = int(self.rng.integers(10, self.cfg.episode_len - 15))
        self.strength = self.cfg.threat_strength
        if self.cfg.threat_strengths:
            self.strength = float(self.rng.choice(self.cfg.threat_strengths))
        self.drift = 0.0
        self.scores = [self._score(0)]
        return self._obs()

    def step(self, action: int):
        c = self.cfg
        threat_active = self.has_threat and self.t >= self.threat_start
        info = {"t": self.t, "has_threat": self.has_threat, "threat_start": self.threat_start}

        if action == ALERT:
            if threat_active:
                delay = self.t - self.threat_start
                info.update(outcome="detected", delay=delay)
                return self._obs(), c.r_detect - c.r_delay * delay, True, info
            info.update(outcome="false_alarm")
            return self._obs(), c.r_false_alarm, True, info

        # WAIT
        self.t += 1
        if self.t >= c.episode_len:
            if self.has_threat:
                info.update(outcome="missed")
                return self._obs(), c.r_miss, True, info
            info.update(outcome="clear")
            return self._obs(), c.r_clear, True, info

        self.scores.append(self._score(self.t))
        return self._obs(), 0.0, False, info
