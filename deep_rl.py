"""Compare tabular Q-learning, PPO, DQN and fixed thresholds.

Usage:
    python deep_rl.py                    # base environment
    python deep_rl.py --config hard      # multiple threat strengths + drifting noise
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO

from agents import ThresholdAgent
from env import EnvConfig
from gym_env import AlertGymEnv
from train import evaluate, train_q

CONFIGS = {
    "base": EnvConfig(),
    "hard": EnvConfig(threat_strengths=(0.15, 0.25, 0.35, 0.45), drift_std=0.02),
}


class SB3Agent:
    """Adapter so an SB3 model plugs into train.evaluate (act(obs, greedy))."""

    def __init__(self, model):
        self.model = model

    def act(self, obs, greedy=True):
        a, _ = self.model.predict(obs, deterministic=True)
        return int(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, default="base")
    ap.add_argument("--steps", type=int, default=300_000, help="env steps for PPO/DQN")
    ap.add_argument("--q-episodes", type=int, default=40_000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    results = {}
    print("training Q-learning ...")
    results["q_learning"] = evaluate(train_q(cfg, args.q_episodes), config=cfg)

    print("training PPO ...")
    ppo = PPO("MlpPolicy", AlertGymEnv(cfg, seed=0), seed=0, verbose=0,
              n_steps=2048, batch_size=256, learning_rate=3e-4, gamma=0.98, ent_coef=0.01)
    ppo.learn(args.steps)
    results["ppo"] = evaluate(SB3Agent(ppo), config=cfg)

    print("training DQN ...")
    dqn = DQN("MlpPolicy", AlertGymEnv(cfg, seed=0), seed=0, verbose=0, gamma=0.98,
              learning_rate=1e-3, buffer_size=100_000, learning_starts=2000, batch_size=128,
              exploration_fraction=0.5, exploration_final_eps=0.02, train_freq=4,
              target_update_interval=1000)
    dqn.learn(args.steps)
    results["dqn"] = evaluate(SB3Agent(dqn), config=cfg)

    for th in np.arange(0.35, 0.651, 0.05):
        results[f"threshold_{th:.2f}"] = evaluate(ThresholdAgent(threshold=float(th)), config=cfg)

    print(f"\n[{args.config}] {'agent':<18}{'reward':>8}{'detect':>8}{'false':>8}{'miss':>8}{'delay':>8}")
    for name, m in results.items():
        print(f"{name:<24}{m['mean_reward']:>8}{m['detection_rate']:>8}"
              f"{m['false_alarm_rate']:>8}{m['miss_rate']:>8}{str(m['mean_delay_frames']):>8}")
    (out / f"deep_rl_{args.config}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
