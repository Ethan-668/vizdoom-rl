import argparse
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from vizdoom import gymnasium_wrapper
from stable_baselines3 import PPO


def preprocess_screen(screen, size=84):
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


def evaluate(model_path, episodes=20, deterministic=True, device="cuda"):
    env = gym.make("VizdoomDeadlyCorridor-v1")
    model = PPO.load(model_path, device=device)

    rewards = []
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset()

        frame = preprocess_screen(obs["screen"])
        frame_stack = deque([frame] * 4, maxlen=4)

        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            stacked_obs = np.stack(list(frame_stack), axis=0)
            action, _ = model.predict(stacked_obs, deterministic=deterministic)
            action_int = int(action)

            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            ep_reward += float(reward)
            step += 1

            frame = preprocess_screen(obs["screen"])
            frame_stack.append(frame)

        rewards.append(ep_reward)
        lengths.append(step)

        print(f"episode {ep + 1}: reward={ep_reward:.2f}, steps={step}")

    env.close()

    print("-" * 50)
    print(f"model: {model_path}")
    print(f"episodes: {episodes}")
    print(f"mean reward: {np.mean(rewards):.2f}")
    print(f"std reward: {np.std(rewards):.2f}")
    print(f"min reward: {np.min(rewards):.2f}")
    print(f"max reward: {np.max(rewards):.2f}")
    print(f"mean steps: {np.mean(lengths):.2f}")
    print(f"std steps: {np.std(lengths):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        device=args.device,
    )


if __name__ == "__main__":
    main()
