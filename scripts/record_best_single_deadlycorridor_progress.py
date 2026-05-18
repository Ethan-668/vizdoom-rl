import argparse
import shutil
from collections import Counter, deque
from pathlib import Path

import cv2
import imageio
import numpy as np
from stable_baselines3 import PPO

from deadlycorridor_progress_env import DeadlyCorridorProgressEnv


def draw_overlay(frame, attempt, skill, step, raw_sum, shaped_sum, info):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, (1280, 960), interpolation=cv2.INTER_NEAREST)
    lines = [
        "Env: DeadlyCorridor Progress Best Single Episode",
        f"Skill: {skill}",
        f"Attempt: {attempt}",
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


def run_attempt(args, model, attempt, temp_path):
    env = DeadlyCorridorProgressEnv(
        cfg_path=f"{args.cfg_dir}/deadly_corridor_skill{args.skill}.cfg",
        action_mode=args.action_mode,
        window_visible=False,
    )

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

    writer = imageio.get_writer(temp_path, fps=args.fps, quality=args.quality)
    try:
        writer.append_data(draw_overlay(env.get_rgb_screen(), attempt, args.skill, 0, 0.0, 0.0, info))
        while not done and step < args.max_steps:
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
            writer.append_data(draw_overlay(env.get_rgb_screen(), attempt, args.skill, step, raw_sum, shaped_sum, info))
    finally:
        writer.close()
        env.close()

    return {
        "attempt": attempt,
        "path": temp_path,
        "steps": step,
        "seconds": (step + 1) / args.fps,
        "terminated": done,
        "raw": raw_sum,
        "shaped": shaped_sum,
        "progress": best_progress,
        "damage": damage,
        "hits": hits,
        "health": float(info.get("health", 0.0)),
        "is_dead": bool(info.get("is_dead", False)),
        "actions": dict(action_counts),
    }


def score_result(result):
    # Main purpose is a long continuous episode; hits/damage break ties.
    return (
        result["steps"],
        result["hits"],
        result["damage"],
        result["progress"],
        result["shaped"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--skill", type=int, default=5)
    parser.add_argument("--cfg-dir", type=str, default="configs/deadlycorridor_progress_long")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--action-mode", choices=["single", "macro"], default="single")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--keep-attempts", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path.parent / f"{out_path.stem}_attempts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model, device=args.device)
    results = []
    best = None

    for attempt in range(1, args.attempts + 1):
        temp_path = temp_dir / f"attempt_{attempt:02d}.mp4"
        result = run_attempt(args, model, attempt, temp_path)
        results.append(result)
        if best is None or score_result(result) > score_result(best):
            best = result
        print(
            f"attempt {attempt:02d}: steps={result['steps']}, seconds={result['seconds']:.1f}, "
            f"raw={result['raw']:.2f}, shaped={result['shaped']:.2f}, progress={result['progress']:.1f}, "
            f"damage={result['damage']:.1f}, hits={result['hits']:.1f}, health={result['health']:.1f}, "
            f"actions={result['actions']}"
        )

    final_name = (
        f"{out_path.stem}_attempt_{best['attempt']:02d}_raw_{best['raw']:.0f}_"
        f"shaped_{best['shaped']:.0f}_progress_{best['progress']:.0f}_"
        f"damage_{best['damage']:.0f}_hits_{best['hits']:.0f}_steps_{best['steps']}.mp4"
    )
    final_path = out_path.with_name(final_name)
    shutil.copy2(best["path"], final_path)
    if final_path != out_path:
        shutil.copy2(best["path"], out_path)

    if not args.keep_attempts:
        shutil.rmtree(temp_dir)

    print(
        f"best_saved={final_path}, alias={out_path}, attempt={best['attempt']}, "
        f"seconds={best['seconds']:.1f}, steps={best['steps']}, raw={best['raw']:.2f}, "
        f"shaped={best['shaped']:.2f}, progress={best['progress']:.1f}, "
        f"damage={best['damage']:.1f}, hits={best['hits']:.1f}, actions={best['actions']}"
    )


if __name__ == "__main__":
    main()
