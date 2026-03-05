"""Lift environment with configurable cube spawn region on the table."""

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
        obs_keys: tuple[str, ...] = DEFAULT_OBS_KEYS,
        **kwargs,
    ):
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
