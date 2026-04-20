from abc import ABC, abstractmethod

import numpy as np
import robosuite
import wandb
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

from agent import VideoCurriculumAgent
from wrapper import RewardShapingWrapper


# Plateau fires when between-window relative improvement drops below this threshold.
# Tuned for a qualitative curriculum demo: eval data from the reach-only run showed
# between-window improvements of 10-16% per 1M steps, so 15% catches natural dips.
PLATEAU_RELATIVE_IMPROVEMENT_THRESHOLD = 0.15

# Time-cap safety net: force a curriculum update if nothing has fired in this many env steps.
PLATEAU_MAX_STEPS_BETWEEN_UPDATES = 1_500_000


class HeuristicCurriculumCallback(BaseCallback, ABC):
    """
    Our fixed-heuristic curriculum. If the reward doesn't improve for n rollouts, trigger the curriculum planner.
    If we hit 90% success rate 3 rollouts in a row, also trigger the curriculum planner.
    """
    def __init__(self, plateau_steps: int):
        super().__init__()

        self.reward_buffer = []
        self.success_buffer = []
        self.n_plateau = None
        self.plateau_steps = plateau_steps

        self.n_updates = 0
        self._steps_since_last_update = 0

    def _on_training_start(self) -> None:
        """
        Computes the number of rollouts that corresponds to a number of plateau steps.
        """
        steps_per_rollout = self.model.n_envs * self.model.n_steps
        self.n_plateau = max(self.plateau_steps // steps_per_rollout, 3)

    def _on_step(self) -> bool:
        """
        Don't update anything.
        """
        return True

    @abstractmethod
    def generate_and_set_curriculum(self, stop_reason: str):
        """
        Generates a new curriculum to be passed into the underlying env.
        """
        raise NotImplementedError("Must implement generate_and_set_curriculum()")

    def _on_rollout_end(self) -> None:
        """
        If we hit a good success rate or plateau, trigger the curriculum planner.
        """
        # Pull rollout metrics from ep_info_buffer — logger.name_to_value is cleared by dump()
        # in between rollouts, so reading rollout/* from the logger silently returns 0.0.
        ep_info_buffer = self.model.ep_info_buffer
        if ep_info_buffer is None or len(ep_info_buffer) == 0:
            return
        self.reward_buffer.append(safe_mean([ep["r"] for ep in ep_info_buffer]))
        self.success_buffer.append(safe_mean([ep.get("is_success", 0.0) for ep in ep_info_buffer]))
        self._steps_since_last_update += self.model.n_envs * self.model.n_steps
        if "ep_base_return" in ep_info_buffer[0]:
            self.logger.record(
                "rollout/ep_base_return_mean",
                safe_mean([ep["ep_base_return"] for ep in ep_info_buffer]),
            )

        # If we're solving the task, let the policy keep stabilizing — no curriculum update.
        if len(self.success_buffer) >= 3 and all(rate > 0.9 for rate in self.success_buffer[-3:]):
            return

        update = False
        stop_reason = None
        # Primary trigger: between-window relative improvement below threshold.
        if len(self.reward_buffer) > self.n_plateau * 2:
            recent_window = self.reward_buffer[-self.n_plateau:]
            previous_window = self.reward_buffer[-self.n_plateau*2:-self.n_plateau]
            recent_mean = safe_mean(recent_window)
            previous_mean = safe_mean(previous_window)
            relative = (recent_mean - previous_mean) / max(abs(previous_mean), 1e-6)

            if relative < PLATEAU_RELATIVE_IMPROVEMENT_THRESHOLD:
                update = True
                stop_reason = "plateau"

        # Fallback trigger: force a curriculum update if it's been too long since the last one.
        if not update and self._steps_since_last_update >= PLATEAU_MAX_STEPS_BETWEEN_UPDATES:
            update = True
            stop_reason = "timecap"

        # If we beat the curriculum or plateau, update the curriculum
        if update:
            # Log for wandb
            self.n_updates += 1
            self.logger.record("curriculum/stage", self.n_updates)
            self.logger.record("curriculum/stop_reason", stop_reason)
            self.logger.record("curriculum/success_rate", self.success_buffer[-1])

            # Don't actually update if we hit n_updates > 10 for API safety with the LLM
            if self.n_updates <= 10:
                # TODO: Generate and set new curriculum
                self.generate_and_set_curriculum(stop_reason)

            # Reset buffers and time-cap counter
            self.success_buffer = []
            self.reward_buffer = []
            self._steps_since_last_update = 0


class VideoCurriculumCallback(HeuristicCurriculumCallback):
    """
    Curriculum callback that captures video of the current policy at plateau/success and passes
    the frames (plus the history of previously-set reward weights — no training statistics) to
    a VideoCurriculumAgent. The agent returns a reward-weights dict which is applied to all
    envs in the training vec env. Cube spawn distribution is NOT under LLM control in this
    callback — the LLM only tunes reward shaping.

    A dedicated single-process robosuite env with offscreen rendering is built lazily on first
    curriculum update. Observations from this env are normalized manually using stats synced
    from the training VecNormalize before being passed to the policy.
    """

    def __init__(
        self,
        curriculum_agent: VideoCurriculumAgent,
        plateau_steps: int,
        n_episodes: int = 2,
        frames_per_episode: int = 6,
        render_height: int = 256,
        render_width: int = 256,
    ):
        super().__init__(plateau_steps)
        self.curriculum_agent = curriculum_agent
        self.n_episodes = n_episodes
        self.frames_per_episode = frames_per_episode
        self.render_height = render_height
        self.render_width = render_width

        # Initial curriculum is deferred to _on_training_start so the LLM sees a video of
        # the untrained policy on its very first call instead of a blank prompt.
        self.curriculum_history: list[str] = []

        # Built lazily — now also used by the initial capture in _on_training_start.
        self._render_env = None
        self._reward_wrapper = None

        # Seed with wrapper defaults so _capture_frames can mirror the training env state
        # before any LLM call has returned weights.
        self._current_weights = dict(RewardShapingWrapper.DEFAULT_WEIGHTS)

    def _on_training_start(self) -> None:
        super()._on_training_start()
        # Capture behavior of the untrained policy and ask the LLM for the first curriculum
        # so it's grounded in what the agent actually looks like at step 0.
        self._build_render_env()
        frames = self._capture_frames()
        self._log_video_to_wandb(frames, tag="initial")
        context = {"stop_reason": "initial", "stage": 0, "remaining": 10}
        response = self.curriculum_agent.generate_curriculum(
            past_curriculum=None, frames=frames, context=context
        )
        self.curriculum_history.append(response)

        weights = self.curriculum_agent.parse_response(response)
        self._current_weights = weights
        self._log_weights(weights)
        self.training_env.env_method("update_reward_weights", weights)

    def _log_weights(self, weights: dict) -> None:
        for k, v in weights.items():
            self.logger.record(f"curriculum/weight/{k}", float(v))

    def _log_video_to_wandb(self, frames: list[np.ndarray], tag: str) -> None:
        if wandb.run is None or not frames:
            return
        video = np.stack(frames).transpose(0, 3, 1, 2)  # (T, H, W, C) → (T, C, H, W)
        wandb.log({f"curriculum/video/{tag}": wandb.Video(video, fps=2)})

    def _build_render_env(self) -> None:
        """Build a single render env with the same wrapper stack as training, minus Monitor/VecEnv."""
        env = robosuite.make(
            "Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,  # policy expects flat obs, not images
            reward_shaping=False,  # we replace shaping via RewardShapingWrapper
            control_freq=20,
        )
        env = RewardShapingWrapper(env)
        self._reward_wrapper = env
        env = GymWrapper(env)
        self._render_env = env

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply the training VecNormalize obs stats (if any) to a raw obs."""
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is None:
            return obs
        return vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32))

    def _capture_frames(self) -> list[np.ndarray]:
        """Roll out the current policy deterministically for n_episodes and return subsampled frames."""
        # Mirror current training-env reward weights on the render env.
        self._reward_wrapper.update_reward_weights(self._current_weights)

        all_frames: list[np.ndarray] = []
        for _ in range(self.n_episodes):
            obs, _ = self._render_env.reset()
            # Re-grab sim after reset: robosuite's hard_reset=True rebuilds self.sim each
            # reset(), so a handle captured before reset() renders a dead, un-stepped sim.
            sim = self._reward_wrapper.env.sim
            ep_frames = []
            # Lift's default horizon is 500; bail early on done.
            for _ in range(500):
                norm_obs = self._normalize_obs(np.asarray(obs))
                action, _ = self.model.predict(norm_obs, deterministic=True)
                obs, _, terminated, truncated, _ = self._render_env.step(action)
                frame = sim.render(
                    camera_name="agentview",
                    width=self.render_width,
                    height=self.render_height,
                )
                ep_frames.append(np.flipud(frame).astype(np.uint8))
                if terminated or truncated:
                    break

            if not ep_frames:
                continue
            # Subsample evenly to self.frames_per_episode
            n = min(self.frames_per_episode, len(ep_frames))
            indices = np.linspace(0, len(ep_frames) - 1, n).astype(int)
            all_frames.extend(ep_frames[i] for i in indices)

        return all_frames

    def generate_and_set_curriculum(self, stop_reason: str):
        """
        Capture behavior video, call the LLM with video + past reward-weight history (no
        training metrics), apply the new reward weights to the training envs.
        """
        if self._render_env is None:
            self._build_render_env()

        # Record video
        frames = self._capture_frames()
        self._log_video_to_wandb(frames, tag=stop_reason)

        # n_updates was incremented in _on_rollout_end before this is called, so stage=n_updates
        # is 1 for the first plateau call, 10 for the last.
        context = {
            "stop_reason": stop_reason,
            "stage": self.n_updates,
            "remaining": 10 - self.n_updates,
        }
        response = self.curriculum_agent.generate_curriculum(
            past_curriculum=self.curriculum_history,
            frames=frames,
            context=context,
        )
        self.curriculum_history.append(response)

        # Get weights
        weights = self.curriculum_agent.parse_response(response)
        self._current_weights = weights
        self._log_weights(weights)
        self.training_env.env_method("update_reward_weights", weights)


class BaselineCurriculumCallback(HeuristicCurriculumCallback):
    """
    Predefines a set of weights as curriculum, and at each generate_and_set_curriculum, applies the next set of weights
    in the curriculum. This is a baseline to compare against the LLM-generated curriculum.
    """
    def __init__(self, plateau_steps: int):
        super().__init__(plateau_steps)

        self.curriculum = [
            {"blah": 1, "blah2": 2, "blah3": 3},
            {"blah": 0.5, "blah2": 1, "blah3": 1.5}
        ]
        self.curriculum_idx = 0

    def generate_and_set_curriculum(self, stop_reason: str):
        if self.curriculum_idx >= len(self.curriculum):
            print("Baseline curriculum exhausted. No new curriculum to apply.")
            return

        new_weights = self.curriculum[self.curriculum_idx]
        self.training_env.env_method("update_reward_weights", new_weights)
        self.curriculum_idx += 1
