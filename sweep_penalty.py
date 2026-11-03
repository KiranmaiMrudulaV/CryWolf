"""Reward sensitivity: sweep the false-alarm penalty and plot the delay / false-alarm Pareto curve.

Usage:
    python sweep_penalty.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from agents import ThresholdAgent
from env import EnvConfig
from train import evaluate, train_q

PENALTIES = [-1, -2, -5, -10, -20, -40, -80]


def main():
    out = Path("results")
    out.mkdir(exist_ok=True)
    base = EnvConfig()
    # Evaluate everything on the SAME (base) environment so the policies are comparable;
    # only the training-time penalty changes.
    rows = []
    for p in PENALTIES:
        agent = train_q(replace(base, r_false_alarm=float(p)), episodes=40_000)
        m = evaluate(agent, config=base)
        rows.append({"false_alarm_penalty": p, **m})
        print(f"penalty {p:>4}: false={m['false_alarm_rate']:.3f} delay={m['mean_delay_frames']} "
              f"detect={m['detection_rate']}")

    thr = []
    for th in np.arange(0.35, 0.701, 0.025):
        m = evaluate(ThresholdAgent(threshold=float(th)), config=base)
        thr.append({"threshold": round(float(th), 3), **m})

    (out / "penalty_sweep.json").write_text(json.dumps({"q_learning": rows, "thresholds": thr}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot([r["false_alarm_rate"] * 100 for r in rows],
             [r["mean_delay_frames"] for r in rows], "o-", color="#2a6fdb", label="Q-learning (varying penalty)")
    for r in rows:
        plt.annotate(str(r["false_alarm_penalty"]), (r["false_alarm_rate"] * 100, r["mean_delay_frames"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8, color="#2a6fdb")
    plt.plot([t["false_alarm_rate"] * 100 for t in thr],
             [t["mean_delay_frames"] for t in thr], "s--", color="#999", ms=4, label="Fixed threshold (varying threshold)")
    plt.xlabel("False-alarm rate (% of episodes)")
    plt.ylabel("Mean detection delay (frames)")
    plt.title("Detection delay vs false-alarm rate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out / "pareto.png", dpi=150)


if __name__ == "__main__":
    main()
