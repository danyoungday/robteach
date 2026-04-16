import imageio
import numpy as np
import robosuite

from test import CurriculumWrapper


def main():
    env = robosuite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        reward_shaping=False,
        control_freq=20,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
    )
    env = CurriculumWrapper(env)
    env.update_curriculum({
        "cube_x_range": (0, 0),
        "cube_y_range": (0, 0),
        "cube_rotation_range": (0, 0)
    })

    low, high = env.action_spec

    obs = env.reset()
    frames = [np.flipud(obs["agentview_image"]).astype(np.uint8)]
    for i in range(100):
        action = np.random.uniform(low, high)
        obs, _, done, _ = env.step(action)
        frames.append(np.flipud(obs["agentview_image"]).astype(np.uint8))
        if i % 10 == 0:
            obs = env.reset()

    imageio.mimsave("video.mp4", frames, fps=20)


if __name__ == "__main__":
    main()