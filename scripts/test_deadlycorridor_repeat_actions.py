import gymnasium as gym
import numpy as np
from vizdoom import gymnasium_wrapper


def test_action(action_id, episodes=5):
    rewards = []
    steps_list = []

    env = gym.make("VizdoomDeadlyCorridor-v1")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            obs, reward, terminated, truncated, info = env.step(action_id)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    env.close()

    print(
        f"action={action_id}, "
        f"mean_reward={np.mean(rewards):8.2f}, "
        f"std={np.std(rewards):8.2f}, "
        f"min={np.min(rewards):8.2f}, "
        f"max={np.max(rewards):8.2f}, "
        f"mean_steps={np.mean(steps_list):8.2f}"
    )


def main():
    env = gym.make("VizdoomDeadlyCorridor-v1")
    n = env.action_space.n
    env.close()

    print(f"action space size: {n}")
    print("-" * 90)

    for action_id in range(n):
        test_action(action_id, episodes=5)


if __name__ == "__main__":
    main()
