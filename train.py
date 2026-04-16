import numpy as np

import robosuite
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler
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


class CurriculumWrapper(Wrapper):
    """
    Gym wrapper that allows updating the curriculum of the env online.
    When apply_curriculum is called, changes the sampler that reset() uses to sample the cube's spawn position and
    orientation according to the curriculum dict.
    """
    def create_placement_initializer(self, xrange, yrange, rotation):
        """
        From Robosuite Lift: creates the placement initializer for the cube object.
        """
        table_offset = np.array((0, 0, 0.8))

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood
        )

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=cube,
            x_range=xrange,
            y_range=yrange,
            rotation=rotation,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=table_offset,
            z_offset=0.01
        )

        return placement_initializer

    def update_curriculum(self, curriculum_dict: dict):
        """
        Updates the curriculum of the env by changing the placement sampler to a one defined by the curriculum dict.
        """
        xrange = (float(curriculum_dict["cube_x_range"][0]), float(curriculum_dict["cube_x_range"][1]))
        yrange = (float(curriculum_dict["cube_y_range"][0]), float(curriculum_dict["cube_y_range"][1]))
        rotation = (float(curriculum_dict["cube_rotation_range"][0]), float(curriculum_dict["cube_rotation_range"][1]))
        placement_initializer = self.create_placement_initializer(xrange, yrange, rotation)

        # Update the underlying env's placement initializer
        self.env.placement_initializer = placement_initializer


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
            env = CurriculumWrapper(env)
        env = SuccessWrapper(env)
        env = GymWrapper(env)
        return Monitor(env)
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