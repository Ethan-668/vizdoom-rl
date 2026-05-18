import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import imageio
import numpy as np
from stable_baselines3 import PPO

from deadlycorridor_progress_env import DeadlyCorridorProgressEnv


def draw_overlay(frame, skill, step, raw_sum, shaped_sum, info):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, (1280, 960), interpolation=cv2.INTER_NEAREST)
    lines = [
        "Env: DeadlyCorridor Progress Single Long Episode",
        f"Skill: {skill}",
        f"Step: {step}",
        f"Raw reward: {raw_sum:.2f}",
        f"Shaped reward: {shaped_sum:.2f}",
        f"Health: {info.get('health', 0.0):.1f}",
        f"Damage: {info.get('damage', 0.0):.1f}",
        f"Hits: {info.get('hits', 0.0):.1f}",
        f"Progress: {info.get('best_progress', 0.0):.1f}",
        f"X/Y: {info.get('x', 0.0):.1f}, {info.get('y', 0.0):.1f}",
        f"Action: {info.get('action_name', 'RESET')}",
    ]
    y = 34
    for line in lines:
        cv2.putText(frame_bgr, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 1, cv2.LINE_AA)
        y += 34
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--skill", type=int, default=5)
    parser.add_argument("--cfg-dir", type=str, default="configs/deadlycorridor_progress_long")
    parser.add_argument("--out", type=str, default="videos/deadlycorridor/progress_from_renotte_400k/skill5_single_long.mp4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--action-mode", choices=["single", "macro"], default="single")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = DeadlyCorridorProgressEnv(
        cfg_path=f"{args.cfg_dir}/deadly_corridor_skill{args.skill}.cfg",
        action_mode=args.action_mode,
        window_visible=False,
    )
    model = PPO.load(args.model, device=args.device)

    obs, info = env.reset()
    frame_stack = deque([obs[:, :, 0]] * args.frame_stack, maxlen=args.frame_stack)
    frames = [draw_overlay(env.get_rgb_screen(), args.skill, 0, 0.0, 0.0, info)]

    raw_sum = 0.0
    shaped_sum = 0.0
    best_progress = 0.0
    damage = 0.0
    hits = 0.0
    step = 0
    done = False
    action_counts = Counter()

    while not done and len(frames) < args.max_frames:
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
        frames.append(draw_overlay(env.get_rgb_screen(), args.skill, step, raw_sum, shaped_sum, info))

    env.close()
    imageio.mimsave(out_path, frames, fps=args.fps, quality=8)
    print(
        f"saved={out_path}, seconds={len(frames) / args.fps:.1f}, "
        f"steps={step}, terminated={done}, raw={raw_sum:.2f}, shaped={shaped_sum:.2f}, "
        f"progress={best_progress:.1f}, damage={damage:.1f}, hits={hits:.1f}, "
        f"actions={dict(action_counts)}"
    )


if __name__ == "__main__":
    main()
