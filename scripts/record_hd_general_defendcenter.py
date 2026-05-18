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


def draw_overlay(frame, algo, ep, step, reward_sum, action, gamevariables):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, (1280, 960), interpolation=cv2.INTER_NEAREST)

    gv = np.asarray(gamevariables).round(2).tolist()

    text_lines = [
        f"Policy: {algo.upper()}",
        f"Episode: {ep}",
        f"Step: {step}",
        f"Reward: {reward_sum:.1f}",
        f"Action ID: {action}",
        f"GameVars: {gv}",
    ]

    y = 40
    for line in text_lines:
        cv2.putText(
            frame_bgr,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 42

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def record(algo, model_path, episodes, out_dir, deterministic=True, device="cuda"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("VizdoomDefendCenter-v1")
    model = load_model(algo, model_path, device=device)

    summary = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()

        frame = preprocess_screen(obs["screen"])
        frame_stack = deque([frame] * 4, maxlen=4)

        frames = []
        done = False
        reward_sum = 0.0
        step = 0

        while not done:
            stacked_obs = np.stack(list(frame_stack), axis=0)
            action, _ = model.predict(stacked_obs, deterministic=deterministic)
            action_int = int(action)

            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            reward_sum += float(reward)
            step += 1

            frame = preprocess_screen(obs["screen"])
            frame_stack.append(frame)

            hd_frame = draw_overlay(
                frame=obs["screen"],
                algo=algo,
                ep=ep,
                step=step,
                reward_sum=reward_sum,
                action=action_int,
                gamevariables=obs["gamevariables"],
            )
            frames.append(hd_frame)

        video_path = out_dir / f"{algo}_episode_{ep:02d}_reward_{reward_sum:.0f}_steps_{step}.mp4"
        imageio.mimsave(video_path, frames, fps=35, quality=8)

        print(f"episode {ep}: reward={reward_sum:.2f}, steps={step}, saved={video_path}")
        summary.append((ep, reward_sum, step, video_path))

    env.close()

    summary_sorted = sorted(summary, key=lambda x: x[1], reverse=True)

    print("-" * 60)
    print(f"Best {algo.upper()} episodes:")
    for ep, reward, step, path in summary_sorted[:5]:
        print(f"episode {ep}: reward={reward:.2f}, steps={step}, video={path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "a2c", "dqn"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stochastic", action="store_true")

    args = parser.parse_args()

    record(
        algo=args.algo,
        model_path=args.model,
        episodes=args.episodes,
        out_dir=args.out_dir,
        deterministic=not args.stochastic,
        device=args.device,
    )


if __name__ == "__main__":
    main()
