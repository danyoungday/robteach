import imageio
import numpy as np
import robosuite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    run_name = "lift-video"
    timestep = 9598464
    policy_path = f"results/{run_name}/checkpoints/ppo_{timestep}_steps.zip"
    vecnorm_path = f"results/{run_name}/checkpoints/ppo_vecnormalize_{timestep}_steps.pkl"

    print("Setting up env")
    robo_env = robosuite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
        hard_reset=False,
    )
    gym_env = GymWrapper(robo_env)

    vec_normalize = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: gym_env]))
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    print("Loading model")
    policy = PPO.load(policy_path)

    n_episodes = 5
    max_steps = 500
    frames = []
    rewards = []
    sim = robo_env.sim
    print("Running episodes")
    for ep in range(n_episodes):
        obs, _ = gym_env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            norm_obs = vec_normalize.normalize_obs(np.asarray(obs, dtype=np.float32))
            action, _ = policy.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, _ = gym_env.step(action)
            ep_reward += float(reward)
            frame = sim.render(camera_name="agentview", width=256, height=256)
            frames.append(np.flipud(frame).astype(np.uint8))
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        print(f"Episode {ep+1}/{n_episodes} reward={ep_reward:.2f} steps={step+1}")

    print("Episode rewards:", rewards)
    imageio.mimsave(f"videos/{run_name}-{timestep}.mp4", frames, fps=20)


if __name__ == "__main__":
    main()
