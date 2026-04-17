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
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

import wandb
from wandb.integration.sb3 import WandbCallback

import yaml

from agent import CurriculumAgent, VideoCurriculumAgent
from callback import TextCurriculumCallback, VideoCurriculumCallback
from wrapper import DictCurriculumWrapper, RewardShapingWrapper


class SuccessWrapper(Wrapper):
    """
    Show success rate in robosuite info dict
    """
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["is_success"] = bool(self.env._check_success())
        return obs, reward, done, info


def env_factory(spawn_curriculum: bool, reward_shaping_wrapper: bool = False):
    """
    Builds a single training/eval env.
      - `spawn_curriculum=True`: chain DictCurriculumWrapper so the cube's spawn distribution can
        be updated online (used by the text curriculum path).
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
            control_freq=20
        )
        if spawn_curriculum:
            env = DictCurriculumWrapper(env)
        if reward_shaping_wrapper:
            env = RewardShapingWrapper(env)
        env = SuccessWrapper(env)
        env = GymWrapper(env)
        return Monitor(env, info_keywords=("is_success",))
    return _make


def make_vec_env(
    evaluate: bool,
    n_envs: int,
    spawn_curriculum: bool = False,
    reward_shaping_wrapper: bool = False,
    vecnormalize_path: str = None
):
    # Spawn curriculum only applies during training — eval always uses the default full spawn range.
    use_spawn = spawn_curriculum and not evaluate
    factories = [
        env_factory(spawn_curriculum=use_spawn, reward_shaping_wrapper=reward_shaping_wrapper)
        for _ in range(n_envs)
    ]
    env = SubprocVecEnv(factories) if n_envs > 1 else DummyVecEnv(factories)

    # When RewardShapingWrapper is in use the reward becomes non-stationary (weights change
    # online), so disable VecNormalize reward normalization to avoid fighting the LLM.
    norm_reward = (not evaluate) and (not reward_shaping_wrapper)

    # We can load from checkpoint if we want
    if vecnormalize_path is None:
        env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward)
    else:
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = not evaluate
        env.norm_reward = norm_reward

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

    if curriculum_mode not in ("text", "video", "baseline"):
        raise ValueError(f"curriculum_mode must be 'text', or 'video', got {curriculum_mode!r}")

    run = wandb.init(
        project="robosuite",
        config=cfg,
        tags=["simplify", curriculum_mode],
        sync_tensorboard=True
    )
    wandb.save(f"{save_dir}/curriculum_log.txt", policy="live")

    use_reward_wrapper = (curriculum_mode == "video")
    use_spawn_curriculum = (curriculum_mode == "text")

    n_envs = cfg["n_envs"]
    vecnormalize_path = cfg.get("vecnormalize_path")
    env = make_vec_env(
        evaluate=False, n_envs=n_envs,
        spawn_curriculum=use_spawn_curriculum,
        reward_shaping_wrapper=use_reward_wrapper,
        vecnormalize_path=vecnormalize_path
    )

    # Create model or load checkpoint
    checkpoint_path = cfg.get("checkpoint_path")
    if checkpoint_path is None:
        policy = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", **ppo_params)
    else:
        policy = PPO.load(checkpoint_path, env=env, verbose=1, tensorboard_log=f"runs/{run.id}", **ppo_params)

    base_env = make_vec_env(
        evaluate=True, n_envs=10,
        spawn_curriculum=False,  # eval always uses the full spawn distribution
        reward_shaping_wrapper=False,  # eval always uses robosuite's default reward
        vecnormalize_path=vecnormalize_path  # don't think this is technically needed but just to be safe
    )
    eval_callback = EvalCallback(
        base_env,
        eval_freq=50_000 // n_envs,
        n_eval_episodes=10,
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

    llm_log_path = f"{save_dir}/curriculum_log.txt"
    llm_video_log_dir = f"{save_dir}/curriculum_videos"
    if curriculum_mode == "text":
        curriculum_agent = CurriculumAgent(log_path=llm_log_path)
        curriculum_callback = TextCurriculumCallback(curriculum_agent, plateau_steps=plateau_steps)
    else:
        curriculum_agent = VideoCurriculumAgent(
            log_path=llm_log_path,
            video_log_dir=llm_video_log_dir,
        )
        curriculum_callback = VideoCurriculumCallback(curriculum_agent, plateau_steps=plateau_steps)

    policy.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_callback, eval_callback, checkpoint_callback, wandb_callback],
        progress_bar=True
    )

    run.finish()

if __name__ == "__main__":
    config_path = "configs/simplify.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train(config)
