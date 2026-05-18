import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import imageio
import numpy as np
from stable_baselines3 import PPO

from deadlycorridor_progress_env import DeadlyCorridorProgressEnv


def draw_overlay(frame, ep, skill, step, raw_sum, shaped_sum, info):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, (1280, 960), interpolation=cv2.INTER_NEAREST)
    lines = [
        "Env: DeadlyCorridor Progress",
        f"Skill: {skill}",
        f"Episode: {ep}",
        f"Step: {step}",
        f"Raw reward: {raw_sum:.2f}",
        f"Shaped reward: {shaped_sum:.2f}",
        f"Health: {info.get('health', 0.0):.1f}",
        f"Damage: {info.get('damage', 0.0):.1f}",
        f"Hits: {info.get('hits', 0.0):.1f}",
        f"Progress: {info.get('best_progress', 0.0):.1f}",
        f"Action: {info.get('action_name', 'RESET')}",
    ]
    y = 36
    for line in lines:
        cv2.putText(frame_bgr, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 1, cv2.LINE_AA)
        y += 36
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def title_frame(text_lines):
    frame = np.zeros((960, 1280, 3), dtype=np.uint8)
    y = 380
    for line in text_lines:
        cv2.putText(frame, line, (90, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        y += 58
    return frame


def make_env(args):
    return DeadlyCorridorProgressEnv(
        cfg_path=f"configs/deadlycorridor_progress/deadly_corridor_skill{args.skill}.cfg",
        action_mode=args.action_mode,
        window_visible=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--skill", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", type=str, default="videos/deadlycorridor/progress_from_renotte_400k/skill5_long.mp4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--action-mode", choices=["single", "macro"], default="single")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--max-total-frames", type=int, default=6000)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    model = PPO.load(args.model, device=args.device)
    frames = []
    summary = []
    pause_frames = max(int(args.pause_seconds * args.fps), 0)

    intro = title_frame([
        "DeadlyCorridor Progress Agent",
        f"Skill {args.skill} | Episodes {args.episodes}",
        "Overlay shows reward, health, damage, hits, progress, action",
    ])
    frames.extend([intro] * max(args.fps * 2, 1))

    for ep in range(1, args.episodes + 1):
        if len(frames) >= args.max_total_frames:
            break

        obs, info = env.reset()
        frame_stack = deque([obs[:, :, 0]] * args.frame_stack, maxlen=args.frame_stack)
        raw_sum = 0.0
        shaped_sum = 0.0
        best_progress = 0.0
        damage = 0.0
        hits = 0.0
        step = 0
        done = False
        action_counts = Counter()

        frames.extend([title_frame([f"Episode {ep}", "Starting new run"]) for _ in range(pause_frames)])

        while not done and len(frames) < args.max_total_frames:
            stacked_obs = np.stack(list(frame_stack), axis=0)
            action, _ = model.predict(stacked_obs, deterministic=not args.stochastic)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            raw_sum += float(info.get("raw_reward", 0.0))
            shaped_sum += float(reward)
            best_progress = max(best_progress, float(info.get("best_progress", 0.0)))
            damage += float(info.get("damage_delta", 0.0))
            hits += float(info.get("hit_delta", 0.0))
            step += 1
            action_counts[info.get("action_name", str(action))] += 1
            frame_stack.append(obs[:, :, 0])
            frames.append(draw_overlay(env.get_rgb_screen(), ep, args.skill, step, raw_sum, shaped_sum, info))

        summary.append((ep, raw_sum, shaped_sum, step, best_progress, damage, hits, dict(action_counts)))
        print(
            f"episode {ep}: raw={raw_sum:.2f}, shaped={shaped_sum:.2f}, "
            f"progress={best_progress:.1f}, damage={damage:.1f}, hits={hits:.1f}, "
            f"steps={step}, actions={dict(action_counts)}"
        )

    env.close()
    imageio.mimsave(out_path, frames, fps=args.fps, quality=8)

    print("-" * 80)
    print(f"saved long video: {out_path}")
    print(f"frames: {len(frames)}, fps: {args.fps}, seconds: {len(frames) / args.fps:.1f}")
    print("summary:")
    for item in summary:
        print(item)


if __name__ == "__main__":
    main()
