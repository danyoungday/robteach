import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import robosuite

from robosuite.wrappers import Wrapper, GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

import wandb
from wandb.integration.sb3 import WandbCallback

import yaml

from agent import VideoCurriculumAgent
from callback import VideoCurriculumCallback, BaselineCurriculumCallback
from wrapper import RewardShapingWrapper


class SuccessWrapper(Wrapper):
    """
    Show success rate in robosuite info dict
    """
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["is_success"] = bool(self.env._check_success())
        return obs, reward, done, info


def env_factory(reward_shaping_wrapper: bool = False):
    """
    Builds a single training/eval env.
      - `reward_shaping_wrapper=True`: disable robosuite's built-in reward shaping and chain
        RewardShapingWrapper so the LLM can control reward weights online (video path).
    """
    def _make():
        env = robosuite.make(
            "Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=(not reward_shaping_wrapper),
            control_freq=20,
            hard_reset=False
        )
        if reward_shaping_wrapper:
            env = RewardShapingWrapper(env)
        env = SuccessWrapper(env)
        env = GymWrapper(env)
        info_keywords = ("is_success",) + (("ep_base_return",) if reward_shaping_wrapper else ())
        return Monitor(env, info_keywords=info_keywords)
    return _make


def make_vec_env(
    evaluate: bool,
    n_envs: int,
    reward_shaping_wrapper: bool = False,
    vecnormalize_path: str = None
):
    """
    Creates a vectorized environment for training or evaluation.
    if reward_shaping_wrapper is true, the env wraps the default reward with a way for us to change weights online.
    """
    factories = [
        env_factory(reward_shaping_wrapper=reward_shaping_wrapper)
        for _ in range(n_envs)
    ]
    env = SubprocVecEnv(factories, start_method="forkserver") if n_envs > 1 else DummyVecEnv(factories)

    # We can load from checkpoint if we want
    if vecnormalize_path is None:
        env = VecNormalize(env, norm_obs=True, norm_reward=(not reward_shaping_wrapper), training=(not evaluate))
    else:
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = not evaluate
        env.norm_reward = (not reward_shaping_wrapper)

    return env


def train(cfg: dict):
    """
    Train PPO.
    """
    curriculum_mode = cfg["curriculum_mode"]
    ppo_params = cfg["ppo_kwargs"]
    save_dir = cfg["save_dir"]
    total_timesteps = cfg["total_timesteps"]
    plateau_steps = cfg["plateau_steps"]

    if os.path.exists(save_dir):
        inp = input("Save directory already exists. Replace it? (y/n)")
        if inp.lower() != "y":
            print("Exiting without training.")
            return

    os.mkdir(save_dir)
    with open(f"{save_dir}/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    if curriculum_mode not in ("video", "baseline", 'default'):
        raise ValueError(f"curriculum_mode must be 'video' 'baseline', or 'default' not {curriculum_mode!r}")

    run = wandb.init(
        project="robosuite",
        config=cfg,
        tags=["simplify", curriculum_mode],
        sync_tensorboard=True
    )

    llm_log_path = f"{save_dir}/curriculum_log.txt"
    open(llm_log_path, "a", encoding="utf-8").close()
    wandb.save(llm_log_path, policy="live")
    wandb.save("sysprompts/curriculum_video_system.txt", policy="live")

    # We wrap in the reward_shaping_wrapper if we're doing curriculum learning, otherwise we don't need it and just use
    # the default reward.
    n_envs = cfg["n_envs"]
    vecnormalize_path = cfg.get("vecnormalize_path")
    env = make_vec_env(
        evaluate=False,
        n_envs=n_envs,
        reward_shaping_wrapper=(curriculum_mode != "default"),
        vecnormalize_path=vecnormalize_path
    )

    # Create model or load checkpoint
    checkpoint_path = cfg.get("checkpoint_path")
    if checkpoint_path is None:
        policy = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", **ppo_params)
    else:
        policy = PPO.load(checkpoint_path, env=env, verbose=1, tensorboard_log=f"runs/{run.id}", **ppo_params)

    n_eval_envs = cfg["n_eval_envs"]
    eval_freq = cfg["eval_freq"]
    base_env = make_vec_env(
        evaluate=True,
        n_envs=n_eval_envs,
        reward_shaping_wrapper=False,  # eval always uses robosuite's default reward
        vecnormalize_path=vecnormalize_path  # don't think this is technically needed but just to be safe
    )
    n_eval_episodes = cfg["n_eval_episodes"]
    eval_callback = EvalCallback(
        base_env,
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=n_eval_episodes,
        log_path=save_dir,
        best_model_save_path=save_dir,
        verbose=1
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=f"{save_dir}/checkpoints",
        name_prefix="ppo",
        save_vecnormalize=True
    )
    wandb_callback = WandbCallback()
    callbacks = [eval_callback, checkpoint_callback, wandb_callback]

    # If we're doing curriculum learning, add the appropriate callback to allow us to modify the reward online.
    if curriculum_mode == "video":
        llm_video_log_dir = f"{save_dir}/curriculum_videos"
        curriculum_agent = VideoCurriculumAgent(
            log_path=llm_log_path,
            video_log_dir=llm_video_log_dir,
        )
        curriculum_callback = VideoCurriculumCallback(
            curriculum_agent, plateau_steps=plateau_steps, eval_callback=eval_callback
        )
        callbacks.append(curriculum_callback)
    elif curriculum_mode == "baseline":
        curriculum_callback = BaselineCurriculumCallback(
            plateau_steps=plateau_steps, eval_callback=eval_callback
        )
        callbacks.append(curriculum_callback)

    policy.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    run.finish()


def run_default():
    with open("configs/simplify.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["curriculum_mode"] = "default"
    config["save_dir"] = "results/default-newparams"
    train(config)


if __name__ == "__main__":

    run_default()

    config_path = "configs/simplify.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config)
