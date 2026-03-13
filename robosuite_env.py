import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite


# Keys present in robosuite 1.5 Lift obs that aggregate all sub-observations.
# Use these by default to avoid duplicate data in the observation vector.
DEFAULT_OBS_KEYS = ("robot0_proprio-state", "object-state")


class RobosuiteGymEnv(gym.Env):
    """Gymnasium wrapper for robosuite environments.

    Flattens the dict observation into a single float32 vector using
    `obs_keys`. Override `obs_keys` or subclass to change what is observed.

    Example:
        env = RobosuiteGymEnv(
            "Lift", "Panda",
            has_renderer=False, has_offscreen_renderer=False,
            use_object_obs=True, use_camera_obs=False,
            reward_shaping=True, control_freq=20,
        )
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env_name: str,
        robots: str | list[str],
        obs_keys: tuple[str, ...] = DEFAULT_OBS_KEYS,
        **kwargs,
    ):
        super().__init__()
        self.obs_keys = obs_keys
        self.env = suite.make(env_name, robots=robots, **kwargs)

        # Build spaces from a dummy reset
        raw_obs = self.env.reset()
        flat_obs = self._flatten_obs(raw_obs)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32
        )
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        arrays = [np.asarray(obs_dict[k]).flatten() for k in self.obs_keys]
        return np.concatenate(arrays).astype(np.float32)

    # ------------------------------------------------------------------
    def _seed_sampler(self, sampler, rng):
        """Recursively set rng on a placement sampler and any sub-samplers."""
        sampler.rng = rng
        if hasattr(sampler, "samplers"):  # SequentialCompositeSampler
            for sub in sampler.samplers.values():
                self._seed_sampler(sub, rng)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.env.rng = rng
            if hasattr(self.env, "placement_initializer") and self.env.placement_initializer is not None:
                self._seed_sampler(self.env.placement_initializer, rng)
        obs = self.env.reset()
        return self._flatten_obs(obs), {}

    def step(self, action):
        # robosuite returns (obs, reward, done, info) — old Gym API
        obs, reward, done, info = self.env.step(action)
        if hasattr(self.env, "_check_success"):
            info["is_success"] = self.env._check_success()
        return self._flatten_obs(obs), float(reward), False, bool(done), info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        super().close()
