from abc import ABC, abstractmethod
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

from agent import CurriculumAgent


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

        update = False
        stop_reason = None
        # If we achieve 3 straight rollouts with >90% success rate, consider the curriculum solved
        if len(self.success_buffer) >= 3 and all(rate > 0.9 for rate in self.success_buffer[-3:]):
            update = True
            stop_reason = "success"

        # Compare the most recent n_plateau steps to the last n_plateau steps. If the average is worse, we're stagnating
        if len(self.reward_buffer) > self.n_plateau * 2:
            recent_window = self.reward_buffer[-self.n_plateau:]
            previous_window = self.reward_buffer[-self.n_plateau*2:-self.n_plateau]
            recent_best = max(recent_window)
            previous_best = max(previous_window)

            if recent_best <= previous_best:
                update = True
                stop_reason = "plateau"

        # If we beat the curriculum or plateau, update the curriculum
        if update:
            # Log for wandb
            self.n_updates += 1
            assert self.n_updates <= 10, "Hit max number of curriculum updates. Killing training."
            self.logger.record("curriculum/stage", self.n_updates)
            self.logger.record("curriculum/stop_reason", stop_reason)
            self.logger.record("curriculum/success_rate", self.success_buffer[-1])

            # TODO: Generate and set new curriculum
            self.generate_and_set_curriculum(stop_reason)

            # Reset buffers
            self.success_buffer = []
            self.reward_buffer = []


class TextCurriculumCallback(HeuristicCurriculumCallback):
    """
    Custom callback to update the curriculum based on training metrics at the end of rollout.
    """
    def __init__(self, curriculum_agent: CurriculumAgent, plateau_steps: int):
        super().__init__(plateau_steps)
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
        Apply the initial curriculum once the training env is attached.
        """
        super()._on_training_start()

        initial_curriculum_dict = self.curriculum_agent.parse_curriculum_dict(self.curriculum_history[0])
        self.training_env.env_method("update_curriculum", initial_curriculum_dict)

    def _on_rollout_start(self) -> None:
        """
        Snapshot train/* losses here — they were recorded by the previous iteration's
        train() call and are cleared by logger.dump() before _on_rollout_end runs.
        """
        self._last_entropy_loss = float(self.logger.name_to_value["train/entropy_loss"])
        self._last_value_loss = float(self.logger.name_to_value["train/value_loss"])

    def generate_and_set_curriculum(self, stop_reason: str):
        """
        Generate new curriculum using LLM based on previous curriculum and training metrics.
        """
        training_metrics = {
            "stop_reason": stop_reason,
            "success_rate": self.success_buffer[-1],
            "avg_reward": self.reward_buffer[-1],
            "entropy_loss": self._last_entropy_loss,
            "value_loss": self._last_value_loss
        }
        self.training_history.append(training_metrics)

        curriculum = self.curriculum_agent.generate_curriculum(
            training_metrics=self.training_history,
            past_curriculum=self.curriculum_history
        )
        self.curriculum_history.append(curriculum)

        # Set curriculum on env
        curriculum_dict = self.curriculum_agent.parse_curriculum_dict(curriculum)
        self.training_env.env_method("update_curriculum", curriculum_dict)

