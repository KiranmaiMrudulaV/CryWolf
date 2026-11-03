"""Train the Q-learning agent, compare it with fixed-threshold baselines, and plot results.

Usage:
    python train.py                 # default: 40,000 training episodes
    python train.py --episodes 80000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agents import QLearningAgent, ThresholdAgent
from env import AlertEnv


def run_episode(env, agent, learn=False):
    obs, total = env.reset(), 0.0
    while True:
        a = agent.act(obs, greedy=not learn)
        nxt, r, done, info = env.step(a)
        if learn:
            agent.update(obs, a, r, nxt, done)
        total += r
        obs = nxt
        if done:
            return total, info


def train_q(config=None, episodes=40_000, seed=0):
    """Train a Q-learning agent on `config` and return it."""
    env = AlertEnv(config, seed=seed)
    agent = QLearningAgent(eps_decay_episodes=int(episodes * 0.75), seed=seed)
    for ep in range(episodes):
        agent.decay(ep)
        run_episode(env, agent, learn=True)
    return agent


def evaluate(agent, episodes=5000, seed=123, config=None):
    env = AlertEnv(config, seed=seed)
    outcomes, delays, rewards = [], [], []
    for _ in range(episodes):
        r, info = run_episode(env, agent)
        rewards.append(r)
        outcomes.append(info["outcome"])
        if info["outcome"] == "detected":
            delays.append(info["delay"])
    o = np.array(outcomes)
    n_threat_eps = sum(1 for x in outcomes if x in ("detected", "missed"))
    return {
        "mean_reward": round(float(np.mean(rewards)), 2),
        "detection_rate": round(float((o == "detected").sum() / max(n_threat_eps, 1)), 3),
        "false_alarm_rate": round(float((o == "false_alarm").sum() / episodes), 3),
        "miss_rate": round(float((o == "missed").sum() / max(n_threat_eps, 1)), 3),
        "mean_delay_frames": round(float(np.mean(delays)), 2) if delays else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40_000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    env = AlertEnv(seed=0)
    agent = QLearningAgent(eps_decay_episodes=int(args.episodes * 0.75))
    checkpoints, curve = [], []
    for ep in range(args.episodes):
        agent.decay(ep)
        run_episode(env, agent, learn=True)
        if (ep + 1) % 1000 == 0:  # greedy check on fresh episodes, no exploration
            checkpoints.append(ep + 1)
            curve.append(evaluate(agent, episodes=500, seed=ep)["mean_reward"])

    results = {"q_learning": evaluate(agent)}
    for th in (0.40, 0.45, 0.50, 0.55, 0.60):
        results[f"threshold_{th:.2f}"] = evaluate(ThresholdAgent(threshold=th))

    print(f"{'agent':<18}{'reward':>8}{'detect':>8}{'false':>8}{'miss':>8}{'delay':>8}")
    for name, m in results.items():
        print(f"{name:<18}{m['mean_reward']:>8}{m['detection_rate']:>8}"
              f"{m['false_alarm_rate']:>8}{m['miss_rate']:>8}{str(m['mean_delay_frames']):>8}")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    np.save(out / "q_table.npy", agent.q)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(checkpoints, curve, color="#2a6fdb", marker="o", ms=3, label="Q-learning agent")
        best_th = max((k for k in results if k.startswith("threshold")),
                      key=lambda k: results[k]["mean_reward"])
        plt.axhline(results[best_th]["mean_reward"], ls="--", color="#999",
                    label=f"best fixed threshold ({best_th.split('_')[1]})")
        plt.xlabel("Training episode")
        plt.ylabel("Mean reward per episode (greedy evaluation)")
        plt.title("Q-learning alert agent: learning curve")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out / "learning_curve.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
