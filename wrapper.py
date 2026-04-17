import numpy as np

from robosuite.wrappers import Wrapper


class RewardShapingWrapper(Wrapper):
    """
    Replaces the base Lift reward with an additive, parameterized shaping function whose weights
    can be updated online via update_reward_weights(). Defaults to weights that approximate the
    base Lift reward (reach + grasp + success) with four additional knobs at zero. The underlying
    env must be constructed with reward_shaping=False so this wrapper isn't double-counting.
    """

    DEFAULT_WEIGHTS = {
        "reach_weight":           1.00,   # 1 - tanh(10 * dist_gripper_to_cube)  — base
        "grasp_weight":           0.25,   # 1.0 if grasping cube else 0         — base
        "success_weight":         2.25,   # 1.0 if _check_success() else 0      — base
        "lift_height_weight":     0.00,   # clip((cube_z - table_z) / 0.04, 0, 1) — fills the base reward's missing middle
        "vertical_align_weight":  0.00,   # exp(-20 * xy_dist) — reward top-down alignment
        "grasp_when_near_weight": 0.00,   # 1.0 if (dist<0.05 and gripper closed) else 0
        "action_penalty":         0.00,   # subtracted: mean(action**2)
    }

    def __init__(self, env):
        super().__init__(env)
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self._resting_cube_z = None  # Set in reset(); used as origin for lift_height term.

    def update_reward_weights(self, weights: dict):
        """
        Replace a subset of weights. Unknown keys raise KeyError so typos surface fast.
        """
        for key, value in weights.items():
            if key not in self.DEFAULT_WEIGHTS:
                raise KeyError(f"Unknown reward weight: {key}")
            self.weights[key] = float(value)

    def reset(self):
        obs = self.env.reset()
        # Capture the cube's resting z so lift_height measures progress above its starting pose,
        # not above the table (which would give a spurious 0.5 bonus for the cube just sitting there).
        self._resting_cube_z = float(self.env.sim.data.body_xpos[self.env.cube_body_id][2])
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)
        info["base_reward"] = float(base_reward)
        return obs, self._shaped(action), done, info

    def _shaped(self, action) -> float:
        env = self.env
        w = self.weights
        # robosuite stores grippers as {arm_name: Gripper}; Panda has one arm, so pick the first.
        gripper_dict = env.robots[0].gripper
        gripper_obj = next(iter(gripper_dict.values()))

        delta = env._gripper_to_target(
            gripper=gripper_dict,
            target=env.cube.root_body,
            target_type="body",
            return_distance=False,
        )
        dist = float(np.linalg.norm(delta))
        xy_dist = float(np.linalg.norm(delta[:2]))

        cube_z = float(env.sim.data.body_xpos[env.cube_body_id][2])

        # current_action is None until the first step sets it; treat None as "not closed"
        current_action = gripper_obj.current_action
        is_closed = current_action is not None and float(current_action[0]) > 0.5

        reach = 1.0 - float(np.tanh(10.0 * dist))
        grasp = 1.0 if env._check_grasp(gripper=gripper_dict, object_geoms=env.cube) else 0.0
        success = 1.0 if env._check_success() else 0.0
        # Lift progress above the cube's resting z, normalized to the 4cm success threshold.
        lift_origin = self._resting_cube_z if self._resting_cube_z is not None else cube_z
        lift_height = float(np.clip((cube_z - lift_origin) / 0.04, 0.0, 1.0))
        vertical_align = float(np.exp(-20.0 * xy_dist))
        grasp_when_near = 1.0 if (dist < 0.05 and is_closed) else 0.0
        action_penalty = float(np.mean(np.asarray(action) ** 2))

        return (
            w["reach_weight"]           * reach
            + w["grasp_weight"]         * grasp
            + w["success_weight"]       * success
            + w["lift_height_weight"]   * lift_height
            + w["vertical_align_weight"] * vertical_align
            + w["grasp_when_near_weight"] * grasp_when_near
            - w["action_penalty"]       * action_penalty
        )
