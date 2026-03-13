"""Train a PPO policy on a robosuite environment.

Usage:
    conda activate robosuite
    python train_ppo.py [configs/basic.yaml]
"""

import importlib.util
import os
import shutil
import sys

import torch
torch.set_num_threads(1)  # prevent PyTorch from fighting SubprocVecEnv workers for cores

from pathlib import Path

import numpy as np
import yaml
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import wandb
from wandb.integration.sb3 import WandbCallback

# ── Callbacks ────────────────────────────────────────────────────────────────


class StopOnSuccessOrStagnation(BaseCallback):
    """Stop training on success-rate threshold OR reward stagnation."""

    def __init__(self, success_threshold: float = 1.0, max_no_improvement_evals: int = 5,
                 min_evals: int = 0, stagnation_success_threshold: float = 0.7, verbose: int = 0):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.stagnation_success_threshold = stagnation_success_threshold
        self.stop_reason = None
        self.n_evals = 0
        self.no_improvement_evals = 0
        self.last_best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        buf = self.parent._is_success_buffer
        success_rate = np.mean(buf) if len(buf) > 0 else 0.0

        # Check hard success threshold
        if success_rate >= self.success_threshold:
            if self.verbose:
                print(f"Early stopping: success rate {success_rate:.2f} >= {self.success_threshold}")
            self.stop_reason = "success"
            return False

        # Check stagnation (based on mean reward)
        self.n_evals += 1
        if self.n_evals > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    if success_rate >= self.stagnation_success_threshold:
                        print(f"Early stopping: reward stagnated but success rate {success_rate:.2f} "
                                f">= {self.stagnation_success_threshold} — treating as success")
                        self.stop_reason = "success"
                    else:
                        print(f"Early stopping: no reward improvement for "
                                f"{self.no_improvement_evals} evals "
                                f"(success rate {success_rate:.2f})")
                        self.stop_reason = "stagnated"
                    return False
        self.last_best_mean_reward = self.parent.best_mean_reward
        return True


class SaveVecNormalizeOnBest(BaseCallback):
    """Save VecNormalize stats whenever EvalCallback finds a new best model."""

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        env = self.parent.eval_env
        env.save(os.path.join(self.save_path, "vec_normalize.pkl"))
        if self.verbose:
            print(f"Saved VecNormalize stats to {self.save_path}/vec_normalize.pkl")
        return True


# ── Helpers ──────────────────────────────────────────────────────────────────


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

    # activation_fn: string → nn.Module class
    pk = cfg["ppo_kwargs"]["policy_kwargs"]
    pk["activation_fn"] = ACTIVATION_MAP[pk["activation_fn"]]

    return cfg


def _env_factory(cfg: dict, env_cls_path: str):
    """Returns a callable that creates a single monitored env (picklable for SubprocVecEnv)."""
    def _make():
        spec = importlib.util.spec_from_file_location("curriculum_env", env_cls_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        EnvCls = mod.CurriculumEnv
        env = EnvCls(robots=cfg["robots"], **cfg["env_kwargs"])
        return Monitor(env)
    return _make


def make_vec_env(cfg: dict, n_envs: int, normalize: bool = True, env_cls_path: str = "") -> DummyVecEnv | SubprocVecEnv | VecNormalize:
    factories = [_env_factory(cfg, env_cls_path=env_cls_path) for _ in range(n_envs)]
    if n_envs == 1:
        vec_env = DummyVecEnv(factories)
    else:
        vec_env = SubprocVecEnv(factories, start_method="forkserver")
    if normalize:
        gamma = cfg.get("ppo_kwargs", {}).get("gamma", 0.99)
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=gamma)
    return vec_env


def resume_ppo(
    cfg: dict,
    ppo_path: str,
    vec_norm_path: str,
    env_cls_path: str = "",
    tb_log_dir: str | None = None,
) -> tuple[PPO, VecNormalize]:
    """Load a PPO model and VecNormalize stats from explicit checkpoint paths."""
    n_envs = cfg["n_envs"]
    env = make_vec_env(cfg, n_envs, env_cls_path=env_cls_path)
    env = VecNormalize.load(vec_norm_path, env.venv if isinstance(env, VecNormalize) else env)
    env.training = True
    env.norm_reward = True

    ppo_kwargs = {k: v for k, v in cfg["ppo_kwargs"].items() if k != "policy_kwargs"}
    model = PPO.load(
        ppo_path, env=env, device=cfg["device"],
        tensorboard_log=tb_log_dir,
        **ppo_kwargs,
    )
    print(f"Resumed model from {ppo_path} with norm stats from {vec_norm_path}")
    return model, env


def train(cfg: dict, config_path: str | None = None, resume_from: dict | None = None, env_cls_path: str = "") -> dict:

    print("Running training with config:")
    print(yaml.dump(cfg, indent=4))

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy source YAML for reproducibility
    if config_path is not None:
        shutil.copy2(config_path, save_dir / "config.yaml")

    # ── Optional wandb setup ─────────────────────────────────────────────
    wandb_active = False
    wandb_cfg = cfg.get("wandb") or {}
    if wandb_cfg.get("project"):

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

    if resume_from is not None:
        model, env = resume_ppo(
            cfg,
            ppo_path=resume_from["ppo_path"],
            vec_norm_path=resume_from["vec_norm_path"],
            env_cls_path=env_cls_path,
            tb_log_dir=str(save_dir / "tb"),
        )
        n_envs = cfg["n_envs"]
    else:
        n_envs = cfg["n_envs"]
        env = make_vec_env(cfg, n_envs, env_cls_path=env_cls_path)
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

    eval_env = None
    stop_callback = None
    if cfg["eval_freq"] > 0:
        n_eval_envs = min(cfg.get("n_eval_episodes", 10), n_envs)
        eval_env = make_vec_env(cfg, n_envs=n_eval_envs, env_cls_path=env_cls_path)
        # Freeze eval normalization stats (use training stats, don't update)
        eval_env.training = False
        eval_env.norm_reward = False
        if cfg.get("early_stop", True):
            stop_callback = StopOnSuccessOrStagnation(
                success_threshold=cfg.get("success_stop_threshold", 1.0),
                max_no_improvement_evals=cfg.get("max_no_improvement_evals", 10),
                min_evals=cfg.get("min_evals_before_stagnation", 0),
                stagnation_success_threshold=cfg.get("stagnation_success_threshold", 0.75),
                verbose=1,
            )
        save_vec_norm_callback = SaveVecNormalizeOnBest(
            save_path=str(save_dir / "best"), verbose=1
        )
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(save_dir / "best"),
                log_path=str(save_dir / "eval"),
                eval_freq=max(cfg["eval_freq"] // n_envs, 1),
                n_eval_episodes=cfg.get("n_eval_episodes", 10),
                deterministic=True,
                render=False,
                callback_on_new_best=save_vec_norm_callback,
                callback_after_eval=stop_callback,
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
    if eval_env is not None:
        eval_env.close()

    if wandb_active:
        wandb.finish()

    stop_reason = "completed"
    if stop_callback is not None and stop_callback.stop_reason is not None:
        stop_reason = stop_callback.stop_reason

    return {"stop_reason": stop_reason}


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/basic.yaml"
    cfg = load_config(config_path)
    train(cfg, config_path, env_cls_path=cfg["env_cls_path"])
