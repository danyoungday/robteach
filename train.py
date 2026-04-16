

import robosuite

from robosuite.wrappers import Wrapper, GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import wandb
from wandb.integration.sb3 import WandbCallback

import yaml

from agent import CurriculumAgent
from callback import TextCurriculumCallback
from wrapper import DictCurriculumWrapper


class SuccessWrapper(Wrapper):
    """
    Show success rate in robosuite info dict
    """
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["is_success"] = bool(self.env._check_success())
        return obs, reward, done, info


def env_factory(curriculum: bool):
    def _make():
        env = robosuite.make(
            "Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20
        )
        if curriculum:
            env = DictCurriculumWrapper(env)
        env = SuccessWrapper(env)
        env = GymWrapper(env)
        return Monitor(env, info_keywords=("is_success",))
    return _make


def make_vec_env(evaluate: bool, n_envs: int):
    if n_envs > 1:
        env = SubprocVecEnv([env_factory(not evaluate) for _ in range(n_envs)])
    else:
        env = DummyVecEnv([env_factory(not evaluate)])

    env = VecNormalize(env, norm_obs=True, norm_reward=(not evaluate))

    return env


def train():

    total_timesteps = 10_000_000
    plateau_steps = 400_000

    run = wandb.init(
        project="robosuite",
        config={
            "total_timesteps": total_timesteps,
            "plateau_steps": plateau_steps
        },
        tags=["simplify"],
        sync_tensorboard=True
    )
    wandb.save("curriculum_log.txt", policy="live") 

    n_envs = 50
    env = make_vec_env(evaluate=False, n_envs=n_envs)

    with open("configs/simplify.yaml", "r", encoding="utf-8") as f:
        curriculum_agent_cfg = yaml.safe_load(f)["ppo_kwargs"]
    policy = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", **curriculum_agent_cfg)

    base_env = make_vec_env(evaluate=True, n_envs=10)
    eval_callback = EvalCallback(
        base_env,
        eval_freq=50_000 // n_envs,
        n_eval_episodes=10,
        log_path="results/test",
        best_model_save_path="results/test",
        verbose=1
    )

    wandb_callback = WandbCallback()

    curriculum_agent = CurriculumAgent(log_path="curriculum_log.txt")
    curriculum_callback = TextCurriculumCallback(curriculum_agent, plateau_steps=plateau_steps)

    policy.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_callback, eval_callback, wandb_callback],
        progress_bar=True
    )

    run.finish()

if __name__ == "__main__":
    train()