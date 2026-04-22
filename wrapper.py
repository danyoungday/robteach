import numpy as np

from robosuite.wrappers import Wrapper


class RewardShapingWrapper(Wrapper):
    """
    Replaces the base Lift reward with an additive, parameterized shaping function whose weights
    can be updated online via update_reward_weights(). Defaults to weights that approximate the
    base Lift reward (reach + grasp + success) plus one additional knob at zero (lift_height,
    which fills the partial-lift gap). The underlying env must be constructed with
    reward_shaping=False so this wrapper isn't double-counting.
    """

    DEFAULT_WEIGHTS = {
        "reach_weight":           1.00,   # 1 - tanh(10 * dist_gripper_to_cube)  — base
        "grasp_weight":           0.25,   # 1.0 if grasping cube else 0         — base
        "success_weight":         2.25,   # 1.0 if _check_success() else 0      — base
        "lift_height_weight":     0.00,   # clip((cube_z - resting_z) / 0.04, 0, 1) — fills the base reward's missing middle
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
        # Defer _resting_cube_z capture to the first step(). At reset the cube is placed
        # ~1 cm above its physical rest (UniformRandomSampler z_offset=0.01) and falls under
        # gravity on the first sim step, so capturing here creates a ~1 cm dead zone in
        # lift_height.
        self._resting_cube_z = None
        self._episode_base_return = 0.0
        return obs

    def step(self, action):
        obs, base_reward, done, info = self.env.step(action)
        if self._resting_cube_z is None:
            # First post-reset step — cube has settled onto the table.
            self._resting_cube_z = float(self.env.sim.data.body_xpos[self.env.cube_body_id][2])
        info["base_reward"] = float(base_reward)
        self._episode_base_return += float(base_reward)
        if done:
            info["ep_base_return"] = self._episode_base_return
        return obs, self._shaped(), done, info

    def _shaped(self) -> float:
        env = self.env
        w = self.weights
        # robosuite stores grippers as {arm_name: Gripper}; Panda has one arm, so pick the first.
        gripper_dict = env.robots[0].gripper

        dist = float(env._gripper_to_target(
            gripper=gripper_dict,
            target=env.cube.root_body,
            target_type="body",
            return_distance=True,
        ))

        cube_z = float(env.sim.data.body_xpos[env.cube_body_id][2])

        reach = 1.0 - float(np.tanh(10.0 * dist))
        grasp = 1.0 if env._check_grasp(gripper=gripper_dict, object_geoms=env.cube) else 0.0
        success = 1.0 if env._check_success() else 0.0
        # Lift progress above the cube's resting z, normalized to 4 cm. Reaches ~0.46 at
        # the success threshold (2 cm above rest) and saturates at 4 cm above rest.
        lift_origin = self._resting_cube_z if self._resting_cube_z is not None else cube_z
        lift_height = float(np.clip((cube_z - lift_origin) / 0.04, 0.0, 1.0))

        return (
            w["reach_weight"]           * reach
            + w["grasp_weight"]         * grasp
            + w["success_weight"]       * success
            + w["lift_height_weight"]   * lift_height
        )
