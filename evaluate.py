"""Evaluate PPO checkpoints on the default (unmodified) robosuite Lift env.

Usage:
    python evaluate.py logs/stage_0/ppo_final --episodes 50
    python evaluate.py logs/stage_0/ppo_final logs/stage_1/ppo_final --episodes 50
    python evaluate.py logs/stage_*/ppo_final --episodes 50
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from enjoy import _find_config, _find_vec_normalize
from robosuite_env import RobosuiteGymEnv
from train_ppo import load_config


def _make_default_lift_env(cfg: dict, n_envs: int = 1):
    """Create default Lift env(s) wrapped in Monitor + VecEnv."""
    def _make():
        env = RobosuiteGymEnv("Lift", robots=cfg["robots"], **cfg["env_kwargs"])
        return Monitor(env)
    factories = [_make for _ in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="forkserver")


def evaluate_checkpoint(
    checkpoint: str,
    n_episodes: int = 50,
    config_override: str | None = None,
    seed: int = 42,
    n_envs: int | None = None,
) -> dict:
    """Run n_episodes on default Lift and return results dict."""
    config_path = _find_config(checkpoint, config_override)
    cfg = load_config(config_path)

    if n_envs is None:
        n_envs = min(n_episodes, os.cpu_count() or 1)
    n_envs = min(n_envs, n_episodes)

    print(f"\n{'='*60}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Config:     {config_path}")
    print(f"Parallel:   {n_envs} envs")

    vec_env = _make_default_lift_env(cfg, n_envs=n_envs)

    norm_file = _find_vec_normalize(checkpoint)
    vec_env = VecNormalize.load(str(norm_file), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    print(f"VecNorm:    {norm_file}")

    model = PPO.load(checkpoint)

    vec_env.seed([seed + i for i in range(n_envs)])
    obs = vec_env.reset()
    ep_rewards = np.zeros(n_envs)
    ep_steps = np.zeros(n_envs, dtype=int)
    rewards = []
    successes = []

    while len(rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        ep_rewards += reward
        ep_steps += 1
        for i in range(n_envs):
            if dones[i] and len(rewards) < n_episodes:
                success = infos[i].get("is_success", False)
                rewards.append(ep_rewards[i])
                successes.append(success)
                ep_num = len(rewards)
                print(f"  Ep {ep_num:3d}/{n_episodes}  [{'SUCCESS' if success else 'FAIL'}]  steps={ep_steps[i]:4d}  reward={ep_rewards[i]:.2f}")
                ep_rewards[i] = 0.0
                ep_steps[i] = 0

    vec_env.close()

    results = {
        "checkpoint": checkpoint,
        "episodes": n_episodes,
        "success_rate": np.mean(successes),
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "rewards": rewards,
        "successes": successes,
    }

    print(f"\n  Summary: success_rate={results['success_rate']:.2%}  "
          f"reward={results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints on default Lift")
    parser.add_argument("checkpoints", nargs="+", help="Path(s) to saved model(s)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--config", default=None, help="Config override (auto-detected if omitted)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Parallel eval envs (default: min(episodes, cpu_count))")
    args = parser.parse_args()

    all_results = []
    for ckpt in args.checkpoints:
        results = evaluate_checkpoint(ckpt, args.episodes, args.config, args.seed, args.n_envs)
        all_results.append(results)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Comparison")
        print(f"{'='*60}")
        print(f"{'Checkpoint':<50} {'Success':>8} {'Reward':>12}")
        print(f"{'-'*50} {'-'*8} {'-'*12}")
        for r in all_results:
            ckpt_short = r["checkpoint"]
            if len(ckpt_short) > 49:
                ckpt_short = "..." + ckpt_short[-46:]
            print(f"{ckpt_short:<50} {r['success_rate']:>7.1%} {r['mean_reward']:>7.2f} +/- {r['std_reward']:.2f}")


if __name__ == "__main__":
    main()
