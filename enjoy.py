"""Visualize a trained SAC checkpoint in the robosuite GUI.

Usage:
    mjpython enjoy.py logs/sac_final          # .zip extension optional
    mjpython enjoy.py logs/checkpoints/sac_25000_steps --episodes 5
"""

import argparse
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

from train_sac import CONFIG, make_vec_env


def _find_vec_normalize(checkpoint: str) -> Path:
    """Auto-detect VecNormalize stats next to the checkpoint or in logs root."""
    ckpt = Path(checkpoint).with_suffix("")  # strip .zip if present
    # CheckpointCallback saves as sac_vecnormalize_<steps>_steps.pkl
    sibling = ckpt.parent / ckpt.name.replace("sac_", "sac_vecnormalize_", 1)
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


def run(checkpoint: str, n_episodes: int = 10):
    model = SAC.load(checkpoint)

    # Create env headless (same as training) to avoid robosuite viewer issues.
    vec_env = make_vec_env(CONFIG, n_envs=1, normalize=False)

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
    args = parser.parse_args()

    run(args.checkpoint, args.episodes)
