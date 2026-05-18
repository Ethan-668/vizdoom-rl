import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np
from vizdoom import gymnasium_wrapper


def draw_overlay(frame, ep, step, reward_sum, action, gamevariables):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, (1280, 960), interpolation=cv2.INTER_NEAREST)

    gv = np.asarray(gamevariables).round(2).tolist()

    text_lines = [
        "Policy: Random",
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


def record_random(episodes, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("VizdoomDefendCenter-v1")
    summary = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()

        frames = []
        done = False
        reward_sum = 0.0
        step = 0

        while not done:
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward_sum += float(reward)
            step += 1

            hd_frame = draw_overlay(
                obs["screen"],
                ep=ep,
                step=step,
                reward_sum=reward_sum,
                action=action,
                gamevariables=obs["gamevariables"],
            )
            frames.append(hd_frame)

        video_path = out_dir / f"random_episode_{ep:02d}_reward_{reward_sum:.0f}_steps_{step}.mp4"
        imageio.mimsave(video_path, frames, fps=35, quality=8)

        print(f"episode {ep}: reward={reward_sum:.2f}, steps={step}, saved={video_path}")
        summary.append((ep, reward_sum, step, video_path))

    env.close()

    summary_sorted = sorted(summary, key=lambda x: x[1], reverse=True)
    print("-" * 60)
    print("Random policy episodes:")
    for ep, reward, step, path in summary_sorted:
        print(f"episode {ep}: reward={reward:.2f}, steps={step}, video={path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="videos/hd_random")
    args = parser.parse_args()

    record_random(
        episodes=args.episodes,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
