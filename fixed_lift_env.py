"""Lift environment with configurable cube spawn region on the table."""

import warnings

import numpy as np
from robosuite.utils.placement_samplers import UniformRandomSampler

from robosuite_env import RobosuiteGymEnv, DEFAULT_OBS_KEYS

TABLE_OFFSET = (0, 0, 0.8)


class FixedCubeLiftEnv(RobosuiteGymEnv):
    """Lift env with a configurable cube spawn region.

    Args:
        robots: Robot string or list, e.g. "Panda".
        cube_x_range: (min, max) x offset from table center.
        cube_y_range: (min, max) y offset from table center.
        cube_rotation: Rotation in radians, or None for random.
        obs_keys: Observation keys to flatten.
        **kwargs: Forwarded to robosuite.make("Lift", ...).
    """

    def __init__(
        self,
        robots: str | list[str] = "Panda",
        cube_x_range: tuple[float, float] = (-0.03, 0.03),
        cube_y_range: tuple[float, float] = (-0.03, 0.03),
        cube_rotation: float | None = 0.0,
        start_gripped: bool = False,
        obs_keys: tuple[str, ...] = DEFAULT_OBS_KEYS,
        **kwargs,
    ):
        self.start_gripped = start_gripped
        sampler = UniformRandomSampler(
            name="CubeSampler",
            x_range=list(cube_x_range),
            y_range=list(cube_y_range),
            rotation=cube_rotation,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=TABLE_OFFSET,
            z_offset=0.01,
        )

        super().__init__(
            env_name="Lift",
            robots=robots,
            obs_keys=obs_keys,
            placement_initializer=sampler,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Scripted grasp for start_gripped mode
    # ------------------------------------------------------------------
    def _scripted_grasp(self):
        """Move EEF to cube and close gripper using the OSC_POSE controller."""
        robot = self.env.robots[0]
        cube_body_id = self.env.sim.model.body_name2id(self.env.cube.root_body)

        # Phase 1: move EEF to cube position
        for _ in range(200):
            cube_pos = self.env.sim.data.body_xpos[cube_body_id]
            eef_pos = self.env.sim.data.site_xpos[robot.eef_site_id["right"]]
            delta = cube_pos - eef_pos
            if np.linalg.norm(delta) < 0.01:
                break
            # Scale delta for controller, gripper open = -1
            action = np.zeros(7)
            action[:3] = np.clip(delta * 5.0, -1.0, 1.0)
            action[6] = -1  # gripper open
            self.env.step(action)

        # Phase 2: close gripper until grasp detected
        grasped = False
        for _ in range(50):
            action = np.zeros(7)
            action[6] = 1  # gripper close
            self.env.step(action)
            if self.env._check_grasp(robot.gripper, self.env.cube):
                grasped = True
                break

        if not grasped:
            warnings.warn("start_gripped: scripted grasp failed to acquire cube")

        # Reset counters so episode horizon is unaffected
        self.env.timestep = 0
        self.env.cur_time = 0
        self.env.done = False

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.start_gripped:
            self._scripted_grasp()
            raw_obs = self.env._get_observations(force_update=True)
            obs = self._flatten_obs(raw_obs)
        return obs, info


CurriculumEnv = FixedCubeLiftEnv
