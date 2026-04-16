import numpy as np
import pytest
import robosuite

from agent import CurriculumAgent
from train import CurriculumWrapper


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

    curriculum_wrapper = CurriculumWrapper(env)

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

