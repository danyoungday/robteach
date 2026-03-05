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


@pytest.fixture(scope="module")
def gripped_env():
    e = FixedCubeLiftEnv(
        robots="Panda",
        cube_x_range=(0, 0),
        cube_y_range=(0, 0),
        cube_rotation=0.0,
        start_gripped=True,
        **ENV_KWARGS,
    )
    yield e
    e.close()


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


# ── start_gripped tests ──────────────────────────────────────────────────


def test_start_gripped_grasp_acquired(gripped_env):
    """After reset with start_gripped=True the robot should be grasping the cube."""
    gripped_env.reset()
    robot = gripped_env.env.robots[0]
    assert gripped_env.env._check_grasp(robot.gripper, gripped_env.env.cube)


def test_start_gripped_timestep_reset(gripped_env):
    """Episode counters should be zero after a start_gripped reset."""
    gripped_env.reset()
    assert gripped_env.env.timestep == 0
    assert gripped_env.env.done is False


def test_start_gripped_obs_shape(gripped_env):
    """Observation shape should match observation_space even with start_gripped."""
    obs, info = gripped_env.reset()
    assert obs.shape == gripped_env.observation_space.shape
    assert obs.dtype == np.float32


def test_start_gripped_false_no_grasp(env):
    """With start_gripped=False (default), the cube should NOT be grasped after reset."""
    env.reset()
    robot = env.env.robots[0]
    assert not env.env._check_grasp(robot.gripper, env.env.cube)


def test_start_gripped_full_episode_runs(gripped_env):
    """Can step through a full episode after a start_gripped reset without errors."""
    gripped_env.reset()
    for _ in range(10):
        action = gripped_env.action_space.sample()
        obs, reward, terminated, truncated, info = gripped_env.step(action)
        assert obs.shape == gripped_env.observation_space.shape
        if terminated or truncated:
            break
