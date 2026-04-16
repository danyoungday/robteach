import numpy as np

import robosuite
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.wrappers import Wrapper, GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import wandb
from wandb.integration.sb3 import WandbCallback

import yaml

from agent import CurriculumAgent


class CurriculumUpdateCallback(BaseCallback):
    """
    Custom callback to update the curriculum based on training metrics at the end of rollout.
    """
    def __init__(self, curriculum_agent: CurriculumAgent):
        super().__init__()
        self.reward_buffer = []
        self.success_buffer = []
        self.n_plateau = None

        self.curriculum_agent = curriculum_agent

        self._last_entropy_loss = 0.0
        self._last_value_loss = 0.0

        # Generate the initial curriculum; application is deferred to _on_training_start
        # because self.training_env is not available until SB3 wires up self.model.
        initial_curriculum = self.curriculum_agent.generate_curriculum(training_metrics=None, past_curriculum=None)
        self.training_history = []
        self.curriculum_history = [initial_curriculum]

    def _on_training_start(self) -> None:
        """
        Set the number of plateau steps based on number of envs and steps per rollout.
        Apply the initial curriculum once the training env is attached.
        """
        plateau_steps = 200_000
        steps_per_rollout = self.model.n_envs * self.model.n_steps
        self.n_plateau = max(plateau_steps // steps_per_rollout, 3)

        initial_curriculum_dict = self.curriculum_agent.parse_curriculum_dict(self.curriculum_history[0])
        self.training_env.env_method("update_curriculum", initial_curriculum_dict)

    def _on_rollout_start(self) -> None:
        """
        Snapshot train/* losses here — they were recorded by the previous iteration's
        train() call and are cleared by logger.dump() before _on_rollout_end runs.
        """
        self._last_entropy_loss = float(self.logger.name_to_value["train/entropy_loss"])
        self._last_value_loss = float(self.logger.name_to_value["train/value_loss"])
        self.success_buffer.append(float(self.logger.name_to_value["rollout/success_rate"]))
        self.reward_buffer.append(float(self.logger.name_to_value["rollout/ep_rew_mean"]))

    def _on_step(self) -> bool:
        """
        Don't update anything.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        If we hit a good success rate, get a new curriculum.
        If our reward plateaus, also get a new curriculum.
        """
        # Skip the first rollout — no train() has run yet, so train/* metrics are still 0.0
        if len(self.success_buffer) < 2 and len(self.training_history) == 0:
            return

        update = False
        stop_reason = None
        # If we achieve 3 straight rollouts with >90% success rate, consider the curriculum solved
        if len(self.success_buffer) >= 3 and all(rate > 0.9 for rate in self.success_buffer[-3:]):
            update = True
            stop_reason = "success"

        # If the best reward in the last n_plateau rollouts hasn't improved, consider the curriculum plateaued
        best_reward = max(self.reward_buffer[-self.n_plateau:])
        best_reward_idx = self.reward_buffer.index(best_reward)
        if len(self.reward_buffer) >= self.n_plateau and best_reward_idx < len(self.reward_buffer) - self.n_plateau:
            update = True
            stop_reason = "plateau"

        # If we beat the curriculum or plateau, update the curriculum
        if update:
            entropy = self._last_entropy_loss
            value_loss = self._last_value_loss
            training_metrics = {
                "stop_reason": stop_reason,
                "success_rate": self.success_buffer[-1],
                "entropy_loss": entropy,
                "value_loss": value_loss
            }
            self.training_history.append(training_metrics)

            # Call curriculum agent to get new curriculum
            curriculum_str = self.curriculum_agent.generate_curriculum(self.training_history, self.curriculum_history)
            self.curriculum_history.append(curriculum_str)

            # Update the underlying envs with the new curriculum
            curriculum_dict = self.curriculum_agent.parse_curriculum_dict(curriculum_str)
            self.training_env.env_method("update_curriculum", curriculum_dict)

            # Reset buffers
            self.success_buffer = []
            self.reward_buffer = []


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


def make_vec_env(eval: bool, n_envs: int):
    if n_envs > 1:
        env = SubprocVecEnv([env_factory(not eval) for _ in range(n_envs)])
    else:
        env = DummyVecEnv([env_factory(not eval)])

    env = VecNormalize(env, norm_obs=True, norm_reward=(not eval))

    return env


def train():

    run = wandb.init(
        project="robosuite",
        tags=["simplify"]
    )

    n_envs = 60
    env = make_vec_env(eval=False, n_envs=n_envs)

    with open("configs/simplify.yaml", "r", encoding="utf-8") as f:
        curriculum_agent_cfg = yaml.safe_load(f)["ppo_kwargs"]
    policy = PPO("MlpPolicy", env, verbose=1, **curriculum_agent_cfg)

    base_env = make_vec_env(eval=True, n_envs=n_envs)
    eval_callback = EvalCallback(
        base_env,
        eval_freq=50_000 // n_envs,
        n_eval_episodes=n_envs,
        log_path="results/test",
        best_model_save_path="results/test",
        verbose=1
    )

    wandb_callback = WandbCallback()

    curriculum_agent = CurriculumAgent(log_path="curriculum_log.txt")
    curriculum_callback = CurriculumUpdateCallback(curriculum_agent)

    policy.learn(total_timesteps=1_000_000, callback=[curriculum_callback, eval_callback, wandb_callback], progress_bar=True)

    run.finish()

if __name__ == "__main__":
    train()