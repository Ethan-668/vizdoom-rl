import argparse
from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np
from gymnasium import spaces
from vizdoom import gymnasium_wrapper
from stable_baselines3 import PPO


class ActionRemapWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_mapping = [0, 1, 2, 3, 5, 6]
        self.action_space = spaces.Discrete(len(self.action_mapping))

    def action(self, act):
        return self.action_mapping[int(act)]


def preprocess_screen(screen, size=84):
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


def draw_overlay(frame, ep, step, reward_sum, new_action, old_action, gamevariables):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, (1280, 960), interpolation=cv2.INTER_NEAREST)

    gv = np.asarray(gamevariables).round(2).tolist()

    action_names = {
        0: "MOVE_LEFT",
        1: "MOVE_RIGHT",
        2: "ATTACK",
        3: "MOVE_FORWARD",
        5: "TURN_LEFT",
        6: "TURN_RIGHT",
    }

    text_lines = [
        "Env: DeadlyCorridor-NoShortcut",
        "Policy: PPO",
        f"Episode: {ep}",
        f"Step: {step}",
        f"Reward: {reward_sum:.2f}",
        f"New Action: {new_action}",
        f"Old Action: {old_action} ({action_names.get(old_action, 'UNKNOWN')})",
        f"GameVars: {gv}",
    ]

    y = 40
    for line in text_lines:
        cv2.putText(frame_bgr, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
        y += 38

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def record(model_path, episodes, out_dir, deterministic=True, device="cuda"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("VizdoomDeadlyCorridor-v1")
    env = ActionRemapWrapper(env)

    model = PPO.load(model_path, device=device)

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
            new_action, _ = model.predict(stacked_obs, deterministic=deterministic)
            new_action = int(new_action)
            old_action = env.action_mapping[new_action]

            obs, reward, terminated, truncated, info = env.step(new_action)
            done = terminated or truncated

            reward_sum += float(reward)
            step += 1

            frame = preprocess_screen(obs["screen"])
            frame_stack.append(frame)

            frames.append(
                draw_overlay(
                    frame=obs["screen"],
                    ep=ep,
                    step=step,
                    reward_sum=reward_sum,
                    new_action=new_action,
                    old_action=old_action,
                    gamevariables=obs["gamevariables"],
                )
            )

        video_path = out_dir / f"noshortcut_episode_{ep:02d}_reward_{reward_sum:.0f}_steps_{step}.mp4"
        imageio.mimsave(video_path, frames, fps=35, quality=8)

        print(f"episode {ep}: reward={reward_sum:.2f}, steps={step}, saved={video_path}")
        summary.append((ep, reward_sum, step, video_path))

    env.close()

    summary_sorted = sorted(summary, key=lambda x: x[1], reverse=True)
    print("-" * 60)
    print("Best NoShortcut episodes:")
    for ep, reward, step, path in summary_sorted[:5]:
        print(f"episode {ep}: reward={reward:.2f}, steps={step}, video={path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/deadlycorridor_noshortcut/ppo_deadlycorridor_noshortcut_500k.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="videos/deadlycorridor/noshortcut_500k")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    record(
        model_path=args.model,
        episodes=args.episodes,
        out_dir=args.out_dir,
        deterministic=not args.stochastic,
        device=args.device,
    )


if __name__ == "__main__":
    main()
