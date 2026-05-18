import argparse
from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np
from vizdoom import gymnasium_wrapper

from stable_baselines3 import PPO, A2C, DQN


def preprocess_screen(screen, size=84):
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


def load_model(algo, model_path, device):
    algo = algo.lower()
    if algo == "ppo":
        return PPO.load(model_path, device=device)
    if algo == "a2c":
        return A2C.load(model_path, device=device)
    if algo == "dqn":
        return DQN.load(model_path, device=device)
    raise ValueError(f"Unsupported algo: {algo}. Use ppo, a2c, or dqn.")


def evaluate(algo, model_path, episodes=20, video_path=None, deterministic=True, device="cuda"):
    env = gym.make("VizdoomDefendCenter-v1")
    model = load_model(algo, model_path, device=device)

    rewards = []
    video_frames = []

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

            if video_path is not None and ep == 0:
                video_frames.append(obs["screen"])

        rewards.append(ep_reward)
        print(f"episode {ep + 1}: reward={ep_reward:.2f}, steps={step}")

    env.close()

    print("-" * 50)
    print(f"algo: {algo}")
    print(f"model: {model_path}")
    print(f"episodes: {episodes}")
    print(f"mean reward: {np.mean(rewards):.2f}")
    print(f"std reward: {np.std(rewards):.2f}")
    print(f"min reward: {np.min(rewards):.2f}")
    print(f"max reward: {np.max(rewards):.2f}")

    if video_path is not None and len(video_frames) > 0:
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, video_frames, fps=35)
        print(f"saved video to: {video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "a2c", "dqn"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    evaluate(
        algo=args.algo,
        model_path=args.model,
        episodes=args.episodes,
        video_path=args.video,
        deterministic=not args.stochastic,
        device=args.device,
    )


if __name__ == "__main__":
    main()
