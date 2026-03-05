"""Train a PPO policy on a robosuite environment.

Usage:
    conda activate robosuite
    python train_ppo.py [configs/basic.yaml]
"""

import os
import shutil
import sys

import torch
torch.set_num_threads(1)  # prevent PyTorch from fighting SubprocVecEnv workers for cores

from pathlib import Path
from typing import Callable

import yaml
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from fixed_lift_env import FixedCubeLiftEnv

# ── Helpers ──────────────────────────────────────────────────────────────────


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear decay from initial_value → 0 over training."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


ACTIVATION_MAP = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
}


def load_config(path: str) -> dict:
    """Load a YAML config and apply non-serializable transformations."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # n_envs: "auto" → computed from CPU count
    if cfg["n_envs"] == "auto":
        cfg["n_envs"] = max(2, ((os.cpu_count() or 4) - 2) // 2 * 2)

    # ranges: lists → tuples
    cfg["cube_x_range"] = tuple(cfg["cube_x_range"])
    cfg["cube_y_range"] = tuple(cfg["cube_y_range"])

    # learning_rate → linear schedule
    cfg["ppo_kwargs"]["learning_rate"] = linear_schedule(cfg["ppo_kwargs"]["learning_rate"])

    # activation_fn: string → nn.Module class
    pk = cfg["ppo_kwargs"]["policy_kwargs"]
    pk["activation_fn"] = ACTIVATION_MAP[pk["activation_fn"]]

    return cfg


def _env_factory(cfg: dict):
    """Returns a callable that creates a single monitored env (picklable for SubprocVecEnv)."""
    def _make():
        env = FixedCubeLiftEnv(
            robots=cfg["robots"],
            cube_x_range=cfg["cube_x_range"],
            cube_y_range=cfg["cube_y_range"],
            start_gripped=cfg.get("start_gripped", False),
            **cfg["env_kwargs"],
        )
        return Monitor(env)
    return _make


def make_vec_env(cfg: dict, n_envs: int, normalize: bool = True) -> DummyVecEnv | SubprocVecEnv | VecNormalize:
    factories = [_env_factory(cfg) for _ in range(n_envs)]
    if n_envs == 1:
        vec_env = DummyVecEnv(factories)
    else:
        vec_env = SubprocVecEnv(factories, start_method="forkserver")
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def train(cfg: dict, config_path: str | None = None) -> PPO:
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy source YAML for reproducibility
    if config_path is not None:
        shutil.copy2(config_path, save_dir / "config.yaml")

    # ── Optional wandb setup ─────────────────────────────────────────────
    wandb_active = False
    wandb_cfg = cfg.get("wandb") or {}
    if wandb_cfg.get("project"):
        import wandb
        from wandb.integration.sb3 import WandbCallback

        # Log the raw YAML config (before load_config transforms)
        log_config = {}
        if config_path is not None:
            with open(config_path) as f:
                log_config = yaml.safe_load(f)

        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            config=log_config,
            sync_tensorboard=True,
            save_code=True,
        )
        wandb_active = True

    n_envs = cfg["n_envs"]
    env = make_vec_env(cfg, n_envs)
    model = PPO(
        "MlpPolicy", env, device=cfg["device"],
        tensorboard_log=str(save_dir / "tb"),
        **cfg["ppo_kwargs"],
    )

    print(f"device={cfg['device']}  n_envs={n_envs}  "
          f"cores_detected={os.cpu_count()}  "
          f"steps/update={n_envs * cfg['ppo_kwargs']['n_steps']}")

    callbacks = []

    if wandb_active:
        callbacks.append(WandbCallback(verbose=2))

    if cfg["checkpoint_freq"] > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(cfg["checkpoint_freq"] // n_envs, 1),
                save_path=str(save_dir / "checkpoints"),
                name_prefix="ppo",
                save_vecnormalize=True,
            )
        )

    if cfg["eval_freq"] > 0:
        eval_env = make_vec_env(cfg, n_envs=1)
        # Freeze eval normalization stats (use training stats, don't update)
        eval_env.training = False
        eval_env.norm_reward = False
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(save_dir / "best"),
                log_path=str(save_dir / "eval"),
                eval_freq=max(cfg["eval_freq"] // n_envs, 1),
                deterministic=True,
                render=False,
            )
        )

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=callbacks or None,
    )

    out_path = save_dir / "ppo_final"
    model.save(out_path)
    env.save(str(save_dir / "vec_normalize.pkl"))
    print(f"Saved → {out_path}.zip + vec_normalize.pkl")

    env.close()

    if wandb_active:
        wandb.finish()

    return model


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/basic.yaml"
    cfg = load_config(config_path)
    train(cfg, config_path)
