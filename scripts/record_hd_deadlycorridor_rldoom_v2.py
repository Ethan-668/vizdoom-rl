import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import imageio
import numpy as np
from stable_baselines3 import PPO

from deadlycorridor_rldoom_env_v2 import DeadlyCorridorRLDoomEnvV2


def draw_overlay(frame, ep, skill, step, raw_sum, shaped_sum, info):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, (1280, 960), interpolation=cv2.INTER_NEAREST)

    text_lines = [
        "Env: DeadlyCorridor RLDoom v2",
        f"Skill: {skill}",
        f"Episode: {ep}",
        f"Step: {step}",
        f"Raw reward: {raw_sum:.2f}",
        f"Shaped reward: {shaped_sum:.2f}",
        f"Health: {info.get('health', 0.0):.1f}",
        f"Damage: {info.get('damage', 0.0):.1f}",
        f"Hits: {info.get('hits', 0.0):.1f}",
        f"Action name: {info.get('action_name', 'RESET')}",
    ]

    y = 38
    for line in text_lines:
        cv2.putText(
            frame_bgr,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 38

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def make_env(args):
    return DeadlyCorridorRLDoomEnvV2(
        cfg_path=f"configs/deadlycorridor_rldoom/deadly_corridor_skill{args.skill}.cfg",
        frame_skip=4,
        image_size=84,
        raw_reward_coef=args.raw_reward_coef,
        damage_reward_coef=args.damage_reward_coef,
        hit_reward_coef=args.hit_reward_coef,
        health_loss_coef=args.health_loss_coef,
        attack_penalty=args.attack_penalty,
        death_penalty=args.death_penalty,
        window_visible=False,
    )


def record(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    model = PPO.load(args.model, device=args.device)
    summary = []

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        frame_stack = deque([obs[:, :, 0]] * 4, maxlen=4)
        frames = [
            draw_overlay(
                frame=env.get_rgb_screen(),
                ep=ep,
                skill=args.skill,
                step=0,
                raw_sum=0.0,
                shaped_sum=0.0,
                info={"health": info.get("health", 0.0), "damage": 0.0, "hits": 0.0},
            )
        ]

        done = False
        shaped_sum = 0.0
        raw_sum = 0.0
        health_loss = 0.0
        damage = 0.0
        hits = 0.0
        step = 0
        ep_action_counts = Counter()

        while not done:
            stacked_obs = np.stack(list(frame_stack), axis=0)
            action, _ = model.predict(stacked_obs, deterministic=not args.stochastic)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            shaped_sum += float(reward)
            raw_sum += float(info.get("raw_reward", 0.0))
            health_loss += float(info.get("health_loss", 0.0))
            damage += float(info.get("damage_delta", 0.0))
            hits += float(info.get("hit_delta", 0.0))
            step += 1

            ep_action_counts[info.get("action_name", str(action))] += 1
            frame_stack.append(obs[:, :, 0])

            frames.append(
                draw_overlay(
                    frame=env.get_rgb_screen(),
                    ep=ep,
                    skill=args.skill,
                    step=step,
                    raw_sum=raw_sum,
                    shaped_sum=shaped_sum,
                    info=info,
                )
            )

        video_path = out_dir / (
            f"episode_{ep:02d}_raw_{raw_sum:.0f}_"
            f"shaped_{shaped_sum:.0f}_steps_{step}.mp4"
        )
        imageio.mimsave(video_path, frames, fps=args.fps, quality=8)

        print(
            f"episode {ep}: raw={raw_sum:.2f}, shaped={shaped_sum:.2f}, "
            f"steps={step}, health_loss={health_loss:.1f}, damage={damage:.1f}, "
            f"hits={hits:.1f}, actions={dict(ep_action_counts)}, saved={video_path}"
        )
        summary.append((ep, raw_sum, shaped_sum, step, health_loss, damage, hits, video_path))

    env.close()

    print("-" * 80)
    print("Recorded DeadlyCorridor RLDoom v2 episodes:")
    for ep, raw_sum, shaped_sum, step, health_loss, damage, hits, path in summary:
        print(
            f"episode {ep}: raw={raw_sum:.2f}, shaped={shaped_sum:.2f}, "
            f"steps={step}, health_loss={health_loss:.1f}, damage={damage:.1f}, "
            f"hits={hits:.1f}, video={path}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--skill", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="videos/deadlycorridor/rldoom_v2/skill5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=35)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--raw-reward-coef", type=float, default=0.05)
    parser.add_argument("--damage-reward-coef", type=float, default=10.0)
    parser.add_argument("--hit-reward-coef", type=float, default=2.0)
    parser.add_argument("--health-loss-coef", type=float, default=5.0)
    parser.add_argument("--attack-penalty", type=float, default=0.02)
    parser.add_argument("--death-penalty", type=float, default=200.0)
    args = parser.parse_args()

    record(args)


if __name__ == "__main__":
    main()
