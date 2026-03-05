"""Test that FixedCubeLiftEnv produces deterministic observations when the cube spawn area is fixed."""

import numpy as np
import pytest

from fixed_lift_env import FixedCubeLiftEnv

ENV_KWARGS = dict(
    has_renderer=False,
    has_offscreen_renderer=False,
    use_object_obs=True,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
    horizon=500,
    hard_reset=False,
)


@pytest.fixture(scope="module")
def env():
    e = FixedCubeLiftEnv(
        robots="Panda",
        cube_x_range=(0, 0),
        cube_y_range=(0, 0),
        cube_rotation=0.0,
        **ENV_KWARGS,
    )
    yield e
    e.close()


def _get_object_obs(env):
    """Return only the object-state slice from the raw robosuite obs dict."""
    raw = env.env._get_observations()
    return np.asarray(raw["object-state"]).flatten().astype(np.float32)


def test_fixed_spawn_gives_consistent_cube_pos(env):
    """Resetting multiple times with a zero-area spawn region should place the cube identically."""
    obs_list = []
    for _ in range(5):
        env.reset()
        obs_list.append(_get_object_obs(env))

    # First 7 elements are cube absolute pose (pos xyz + quat xyzw).
    # Remaining elements are cube-to-gripper relative pos, which varies
    # due to robot arm settling differently each reset.
    for i in range(1, len(obs_list)):
        np.testing.assert_array_equal(
            obs_list[0][:7], obs_list[i][:7],
            err_msg=f"Reset {i} produced a different cube pose than reset 0",
        )
