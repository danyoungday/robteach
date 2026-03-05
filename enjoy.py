"""Visualize a trained PPO checkpoint in the robosuite GUI.

Usage:
    mjpython enjoy.py logs/ppo_final          # .zip extension optional
    mjpython enjoy.py logs/checkpoints/ppo_50000_steps --episodes 5
"""

import argparse
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from train_ppo import load_config, make_vec_env


def _find_vec_normalize(checkpoint: str) -> Path:
    """Auto-detect VecNormalize stats next to the checkpoint or in logs root."""
    ckpt = Path(checkpoint).with_suffix("")  # strip .zip if present
    # CheckpointCallback saves as ppo_vecnormalize_<steps>_steps.pkl
    sibling = ckpt.parent / ckpt.name.replace("ppo_", "ppo_vecnormalize_", 1)
    candidates = [
        sibling.with_suffix(".pkl"),
        ckpt.parent / "vec_normalize.pkl",
        ckpt.parent.parent / "vec_normalize.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No VecNormalize stats found for checkpoint {checkpoint}. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def _get_robosuite_env(vec_env):
    """Unwrap VecNormalize -> DummyVecEnv -> Monitor -> GymEnv."""
    return vec_env.venv.envs[0].env


def _find_config(checkpoint: str, cli_config: str | None = None) -> str:
    """Locate config.yaml saved alongside the checkpoint, or fall back to CLI arg."""
    ckpt = Path(checkpoint).with_suffix("")
    candidates = [
        ckpt.parent / "config.yaml",
        ckpt.parent.parent / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    if cli_config is not None:
        return cli_config
    raise FileNotFoundError(
        f"No config.yaml found for checkpoint {checkpoint}. "
        f"Searched: {[str(c) for c in candidates]}. "
        "Provide one with --config."
    )


def run(checkpoint: str, n_episodes: int = 10, config_override: str | None = None):
    config_path = _find_config(checkpoint, config_override)
    cfg = load_config(config_path)
    print(f"Loaded config from {config_path}")

    model = PPO.load(checkpoint)

    # Create env headless (same as training) to avoid robosuite viewer issues.
    vec_env = make_vec_env(cfg, n_envs=1, normalize=False)

    norm_file = _find_vec_normalize(checkpoint)
    vec_env = VecNormalize.load(str(norm_file), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    print(f"Loaded normalization stats from {norm_file}")

    # Launch a single MuJoCo viewer ourselves — survives resets.
    robo_env = _get_robosuite_env(vec_env)
    viewer = mujoco.viewer.launch_passive(
        robo_env.env.sim.model._model,
        robo_env.env.sim.data._data,
    )

    for ep in range(1, n_episodes + 1):
        obs = vec_env.reset()
        episode_reward = 0.0
        step = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _infos = vec_env.step(action)
            viewer.sync()

            episode_reward += reward[0]
            step += 1
            done = dones[0]

        success = _infos[0].get("is_success", False)
        status = "SUCCESS" if success else "FAIL"
        print(f"Episode {ep} [{status}] — {step} steps, total reward: {episode_reward:.4f}")

    viewer.close()
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to saved model (.zip or without extension)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--config", default=None, help="Path to YAML config (auto-detected from checkpoint dir if omitted)")
    args = parser.parse_args()

    run(args.checkpoint, args.episodes, args.config)
