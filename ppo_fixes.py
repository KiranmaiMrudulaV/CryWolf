"""Try fixes for PPO collapsing to 'never alert' in the hard environment.

Usage:
    python ppo_fixes.py <variant>      # one of VARIANTS; writes results/ppo_fix_<variant>.json
Every variant is EVALUATED on the original (unshaped) hard config.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from deep_rl import CONFIGS, SB3Agent
from gym_env import AlertGymEnv
from train import evaluate

HARD = CONFIGS["hard"]
# name: (train_config, ppo kwargs, timesteps, vecnormalize)
VARIANTS = {
    "baseline": (HARD, dict(ent_coef=0.01), 300_000, False),
    "ent_0.05": (HARD, dict(ent_coef=0.05), 300_000, False),
    "ent_0.10": (HARD, dict(ent_coef=0.10), 300_000, False),
    "vecnorm": (HARD, dict(ent_coef=0.01), 300_000, True),
    "steps_1M": (HARD, dict(ent_coef=0.01), 1_000_000, False),
    "step_cost": (replace(HARD, r_step_threat=0.5), dict(ent_coef=0.01), 300_000, False),
    "combo": (replace(HARD, r_step_threat=0.5), dict(ent_coef=0.05), 300_000, True),
}


def main(name: str):
    cfg, kw, steps, vecnorm = VARIANTS[name]
    make = lambda: AlertGymEnv(cfg, seed=0)
    env = VecNormalize(DummyVecEnv([make]), norm_obs=False, norm_reward=True, gamma=0.98) if vecnorm else make()
    model = PPO("MlpPolicy", env, seed=0, verbose=0, n_steps=2048, batch_size=256,
                learning_rate=3e-4, gamma=0.98, **kw)
    model.learn(steps)
    res = evaluate(SB3Agent(model), config=HARD)  # fair: unshaped reward, fresh seed
    print(name, res)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/ppo_fix_{name}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
