"""Train a PPO policy on a robosuite environment.

Usage:
    conda activate robosuite
    python train_ppo.py
"""

import torch
torch.set_num_threads(1)  # prevent PyTorch from fighting SubprocVecEnv workers for cores

from pathlib import Path
from typing import Callable

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


# ── Configuration ────────────────────────────────────────────────────────────

CONFIG = dict(
    robots="Panda",              # e.g. "UR5e", ["Panda", "Panda"]
    cube_x_range=(0, 0), # (min, max) x offset from table center; use (0, 0) to fix
    cube_y_range=(0, 0), # (min, max) y offset from table center; use (0, 0) to fix

    env_kwargs=dict(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        # reward_scale=1.0,        # normalize max per-step reward to 1.0
        control_freq=20,
        horizon=500,             # canonical benchmark setting
        hard_reset=False,        # soft reset — much faster
    ),

    # Parallelism
    n_envs=4,                # parallel envs (SubprocVecEnv); set to 1 for single-process
    device="mps",            # "mps" | "cpu" | "cuda"

    # PPO hyperparameters
    ppo_kwargs=dict(
        learning_rate=linear_schedule(3e-4),
        n_steps=512,         # steps per env per rollout (total = n_envs * n_steps)
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=1e-3,       # entropy bonus — prevents premature convergence
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=dict(
            log_std_init=-2,         # smaller initial std → less random actions
            ortho_init=False,        # works better with ReLU
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    ),

    total_timesteps=1_000_000,
    save_dir="./logs",
    checkpoint_freq=50_000,  # 0 to disable
    eval_freq=50_000,        # 0 to disable
)

# ─────────────────────────────────────────────────────────────────────────────


def _env_factory(cfg: dict):
    """Returns a callable that creates a single monitored env (picklable for SubprocVecEnv)."""
    def _make():
        env = FixedCubeLiftEnv(
            robots=cfg["robots"],
            cube_x_range=cfg["cube_x_range"],
            cube_y_range=cfg["cube_y_range"],
            **cfg["env_kwargs"],
        )
        return Monitor(env)
    return _make


def make_vec_env(cfg: dict, n_envs: int, normalize: bool = True) -> DummyVecEnv | SubprocVecEnv | VecNormalize:
    factories = [_env_factory(cfg) for _ in range(n_envs)]
    if n_envs == 1:
        vec_env = DummyVecEnv(factories)
    else:
        vec_env = SubprocVecEnv(factories, start_method="fork")
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def train(cfg: dict = CONFIG) -> PPO:
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    n_envs = cfg["n_envs"]
    env = make_vec_env(cfg, n_envs)
    model = PPO("MlpPolicy", env, device=cfg["device"], **cfg["ppo_kwargs"])

    print(f"device={cfg['device']}  n_envs={n_envs}  "
          f"steps/update={n_envs * cfg['ppo_kwargs']['n_steps']}")

    callbacks = []

    if cfg["checkpoint_freq"] > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=cfg["checkpoint_freq"],
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
                eval_freq=cfg["eval_freq"],
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
    return model


if __name__ == "__main__":
    train()
