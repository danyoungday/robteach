import imageio
import json

import numpy as np
import pytest
import robosuite

from callback import VideoCurriculumCallback
from wrapper import RewardShapingWrapper


# def test_base_curriculum():
#     curriculum_agent = CurriculumAgent(log_path="test_log.txt")
#     curriculum = curriculum_agent.generate_curriculum(training_metrics=None, past_curriculum=None)
#     print(curriculum)


# def test_curriculum_with_metrics():
#     curriculum_agent = CurriculumAgent(log_path="test_log.txt")
#     training_history = [
#         {"success_rate": 0.1, "entropy_loss": 0.5, "value_loss": 1.0, "stop_reason": "plateau"},
#         {"success_rate": 0.3, "entropy_loss": 0.4, "value_loss": 0.8, "stop_reason": "success"},
#         {"success_rate": 0.5, "entropy_loss": 0.3, "value_loss": 0.6, "stop_reason": "success"},
#     ]
#     curriculum_history = [
#         "first let's set the spawn range of the cube to 0 and the rotation to 0",
#         "now let's increase the spawn range to 0.05 but add no rotation",
#         "now let's add rotation"
#     ]
#     curriculum = curriculum_agent.generate_curriculum(
#         training_metrics=training_history,
#         past_curriculum=curriculum_history
#     )
#     print(curriculum)


def test_update_placement():
    """
    Tests the CurriculumWrapper's ability to update the cube's spawn position and rotation according to the curriculum
    dict online.
    """
    env = robosuite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20
    )

    curriculum_wrapper = DictCurriculumWrapper(env)

    default_spawn_ranges = [-0.03, 0.03]

    # Phase 1: fixed spawn point, zero rotation
    fixed_ranges = [0.0, 0.0]
    fixed_rotation = [0.0, 0.0]
    curriculum_wrapper.update_curriculum({
        "cube_x_range": fixed_ranges,
        "cube_y_range": fixed_ranges,
        "cube_rotation_range": fixed_rotation,
    })
    for _ in range(100):
        obs = curriculum_wrapper.reset()
        cube_pos = obs["cube_pos"]
        cube_quat = obs["cube_quat"]
        angle = 2 * np.arctan2(cube_quat[2], cube_quat[3])
        assert abs(cube_pos[0]) <= 1e-6
        assert abs(cube_pos[1]) <= 1e-6
        assert abs(angle) <= 1e-6

    # Phase 2: slightly increased spawn range
    small_ranges = [-0.01, 0.01]
    small_rotation = [-0.1, 0.1]
    curriculum_wrapper.update_curriculum({
        "cube_x_range": small_ranges,
        "cube_y_range": small_ranges,
        "cube_rotation_range": small_rotation,
    })
    xs, ys, angles = [], [], []
    for _ in range(100):
        obs = curriculum_wrapper.reset()
        cube_pos = obs["cube_pos"]
        cube_quat = obs["cube_quat"]
        angle = 2 * np.arctan2(cube_quat[2], cube_quat[3])
        assert small_ranges[0] <= cube_pos[0] <= small_ranges[1]
        assert small_ranges[0] <= cube_pos[1] <= small_ranges[1]
        assert small_rotation[0] <= angle <= small_rotation[1]
        xs.append(cube_pos[0])
        ys.append(cube_pos[1])
        angles.append(angle)
    assert max(xs) - min(xs) > 0
    assert max(ys) - min(ys) > 0
    assert max(angles) - min(angles) > 0

    # Phase 3: spawn range larger than the default
    large_ranges = [-0.1, 0.1]
    large_rotation = [-np.pi, np.pi]
    curriculum_wrapper.update_curriculum({
        "cube_x_range": large_ranges,
        "cube_y_range": large_ranges,
        "cube_rotation_range": large_rotation,
    })
    exceeded_default = False
    for _ in range(100):
        obs = curriculum_wrapper.reset()
        cube_pos = obs["cube_pos"]
        cube_quat = obs["cube_quat"]
        angle = 2 * np.arctan2(cube_quat[2], cube_quat[3])
        assert large_ranges[0] <= cube_pos[0] <= large_ranges[1]
        assert large_ranges[0] <= cube_pos[1] <= large_ranges[1]
        assert large_rotation[0] <= angle <= large_rotation[1]
        if abs(cube_pos[0]) > default_spawn_ranges[1] or abs(cube_pos[1]) > default_spawn_ranges[1]:
            exceeded_default = True
    assert exceeded_default


class _StubVideoAgent:
    """Stand-in for VideoCurriculumAgent — no Claude calls, fixed reward weights."""
    _WEIGHTS_JSON = (
        '{"reach_weight": 1.0, "grasp_weight": 0.25, "success_weight": 2.25, '
        '"lift_height_weight": 0.0, "vertical_align_weight": 0.0, '
        '"grasp_when_near_weight": 0.0, "action_penalty": 0.0}'
    )

    def generate_curriculum(self, past_curriculum=None, frames=None):
        return self._WEIGHTS_JSON

    def parse_response(self, response):
        return json.loads(response)


class _StubModel:
    """Random-action policy so _capture_frames has something that actually moves the arm."""
    def __init__(self, action_space):
        self._action_space = action_space

    def predict(self, obs, deterministic=True):
        return self._action_space.sample(), None

    def get_vec_normalize_env(self):
        return None


def test_capture_frames():
    """
    VideoCurriculumCallback._capture_frames should return frames that actually change over
    time. A static sequence means the arm isn't moving or we're rendering from a stale sim.
    """
    callback = VideoCurriculumCallback(
        _StubVideoAgent(), plateau_steps=1000, n_episodes=1, frames_per_episode=6,
    )
    callback._build_render_env()
    callback._current_weights = callback.curriculum_agent.parse_response(
        _StubVideoAgent._WEIGHTS_JSON
    )
    callback.model = _StubModel(callback._render_env.action_space)

    frames = callback._capture_frames()

    assert len(frames) >= 2, f"expected multiple frames, got {len(frames)}"
    first = frames[0].astype(np.int32)
    diffs = [int(np.abs(f.astype(np.int32) - first).sum()) for f in frames[1:]]
    assert any(d > 0 for d in diffs), (
        "Captured frames are identical — arm isn't moving or the cached sim handle is stale"
    )

    # Record frames to mp4 for manual inspection:
    with imageio.get_writer("test_frames.mp4", fps=2) as writer:
        for frame in frames:
            writer.append_data(frame)


def test_reward_shaping_wrapper():
    """
    Sanity-check RewardShapingWrapper: known-key updates, typo rejection, zero-all-weights
    producing exactly 0, isolated action_penalty matching its closed form, base_reward
    preserved in info, and resting_cube_z captured on reset.
    """
    env = robosuite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
    )
    wrapper = RewardShapingWrapper(env)
    wrapper.reset()

    with pytest.raises(KeyError):
        wrapper.update_reward_weights({"not_a_real_key": 1.0})

    wrapper.update_reward_weights({"reach_weight": 0.5})
    assert wrapper.weights["reach_weight"] == 0.5
    assert wrapper.weights["grasp_weight"] == RewardShapingWrapper.DEFAULT_WEIGHTS["grasp_weight"]

    assert wrapper._resting_cube_z is not None

    action_dim = env.action_spec[0].shape[0]
    zero_action = np.zeros(action_dim)

    wrapper.update_reward_weights(dict(RewardShapingWrapper.DEFAULT_WEIGHTS))
    _, reward, _, info = wrapper.step(zero_action)
    assert np.isfinite(reward)
    assert "base_reward" in info

    wrapper.reset()
    wrapper.update_reward_weights({k: 0.0 for k in RewardShapingWrapper.DEFAULT_WEIGHTS})
    _, reward, _, _ = wrapper.step(zero_action)
    assert reward == 0.0

    wrapper.update_reward_weights({"action_penalty": 1.0})
    big_action = np.ones(action_dim) * 0.5
    _, reward, _, _ = wrapper.step(big_action)
    assert reward == pytest.approx(-np.mean(big_action ** 2), rel=1e-6)
