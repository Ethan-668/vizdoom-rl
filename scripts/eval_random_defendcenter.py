import argparse
import gymnasium as gym
import numpy as np
from vizdoom import gymnasium_wrapper


def evaluate_random(episodes):
    env = gym.make("VizdoomDefendCenter-v1")
    rewards = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            step += 1

        rewards.append(ep_reward)
        print(f"episode {ep + 1}: reward={ep_reward:.2f}, steps={step}")

    env.close()

    print("-" * 50)
    print("policy: random")
    print(f"episodes: {episodes}")
    print(f"mean reward: {np.mean(rewards):.2f}")
    print(f"std reward: {np.std(rewards):.2f}")
    print(f"min reward: {np.min(rewards):.2f}")
    print(f"max reward: {np.max(rewards):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    evaluate_random(args.episodes)


if __name__ == "__main__":
    main()
