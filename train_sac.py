"""Train a SAC policy on a robosuite environment.

Usage:
    conda activate robosuite
    python train_sac.py
"""

import torch
torch.set_num_threads(1)  # prevent PyTorch from fighting SubprocVecEnv workers for cores

from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from fixed_lift_env import FixedCubeLiftEnv

# ── Configuration ────────────────────────────────────────────────────────────

CONFIG = dict(
    robots="Panda",              # e.g. "UR5e", ["Panda", "Panda"]
    cube_x_range=(0, 0), # (min, max) x offset from table center; use (0, 0) to fix
    cube_y_range=(0, 0), # (min, max) y offset from table center; use (0, 0) to fix
    start_gripped=False,

    env_kwargs=dict(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        horizon=500,             # canonical benchmark setting
        hard_reset=False,        # soft reset — much faster
    ),

    # Parallelism
    n_envs=1,                # SAC doesn't benefit from parallel envs
    device="mps",            # "mps" | "cpu" | "cuda"

    # SAC hyperparameters
    sac_kwargs=dict(
        learning_rate=7.3e-4,
        buffer_size=300_000,
        batch_size=256,
        ent_coef="auto",        # auto-tuned entropy
        gamma=0.98,
        tau=0.02,
        train_freq=8,
        gradient_steps=8,
        learning_starts=10_000,
        use_sde=True,           # state-dependent exploration
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
        verbose=1,
    ),

    total_timesteps=300_000,     # SAC converges faster
    save_dir="./logs",
    checkpoint_freq=25_000,      # more frequent for shorter run
    eval_freq=25_000,
)

# ─────────────────────────────────────────────────────────────────────────────


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
        vec_env = SubprocVecEnv(factories, start_method="fork")
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def train(cfg: dict = CONFIG) -> SAC:
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    n_envs = cfg["n_envs"]
    env = make_vec_env(cfg, n_envs)
    model = SAC("MlpPolicy", env, device=cfg["device"], **cfg["sac_kwargs"])

    print(f"device={cfg['device']}  n_envs={n_envs}  "
          f"buffer_size={cfg['sac_kwargs']['buffer_size']}  "
          f"train_freq={cfg['sac_kwargs']['train_freq']}")

    callbacks = []

    if cfg["checkpoint_freq"] > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=cfg["checkpoint_freq"],
                save_path=str(save_dir / "checkpoints"),
                name_prefix="sac",
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

    out_path = save_dir / "sac_final"
    model.save(out_path)
    env.save(str(save_dir / "vec_normalize.pkl"))
    print(f"Saved → {out_path}.zip + vec_normalize.pkl")

    env.close()
    return model


if __name__ == "__main__":
    train()
