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
        f"Kills: {info.get('kills', 0.0):.1f}",
        f"Progress: {info.get('best_progress', 0.0):.1f}",
        f"Gate: {info.get('progress_gate_index', 0)} Ready: {int(bool(info.get('combat_ready', True)))}",
        f"Total Dmg/Hits/Kills: {info.get('total_damage', 0.0):.1f}/{info.get('total_hits', 0.0):.1f}/{info.get('total_kills', 0.0):.1f}",
        f"FB steps: {info.get('forward_backward_steps', 0)}",
        f"Action: {info.get('action_name', 'RESET')}",
    ]
    y = 36
    for line in lines:
        cv2.putText(frame_bgr, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 1, cv2.LINE_AA)
        y += 36
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def make_env(args):
    return DeadlyCorridorProgressEnv(
        cfg_path=f"{args.cfg_dir}/deadly_corridor_skill{args.skill}.cfg",
        frame_skip=args.frame_skip,
        raw_reward_coef=args.raw_reward_coef,
        damage_reward_coef=args.damage_reward_coef,
        hit_count_coef=args.hit_count_coef,
        kill_count_coef=args.kill_count_coef,
        ammo_coef=args.ammo_coef,
        progress_coef=args.progress_coef,
        progress_gate_size=args.progress_gate_size,
        progress_gate_reward=args.progress_gate_reward,
        combat_gate_damage=args.combat_gate_damage,
        combat_gate_hits=args.combat_gate_hits,
        combat_gate_kills=args.combat_gate_kills,
        safe_progress_coef=args.safe_progress_coef,
        unsafe_progress_penalty_coef=args.unsafe_progress_penalty_coef,
        health_loss_coef=args.health_loss_coef,
        forward_action_bonus=args.forward_action_bonus,
        no_progress_penalty=args.no_progress_penalty,
        stall_penalty=args.stall_penalty,
        stall_threshold=args.stall_threshold,
        oscillation_penalty=args.oscillation_penalty,
        oscillation_window=args.oscillation_window,
        backward_penalty_coef=args.backward_penalty_coef,
        backward_tolerance=args.backward_tolerance,
        repeat_turn_penalty=args.repeat_turn_penalty,
        repeat_turn_threshold=args.repeat_turn_threshold,
        repeat_strafe_penalty=args.repeat_strafe_penalty,
        repeat_strafe_threshold=args.repeat_strafe_threshold,
        death_penalty=args.death_penalty,
        action_mode=args.action_mode,
        window_visible=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--skill", type=int, default=5)
    parser.add_argument("--cfg-dir", type=str, default="configs/deadlycorridor_progress")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="videos/deadlycorridor/progress/skill5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--action-mode", choices=["single", "macro", "combat_macro"], default="single")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--fps", type=int, default=35)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--raw-reward-coef", type=float, default=0.02)
    parser.add_argument("--damage-reward-coef", type=float, default=10.0)
    parser.add_argument("--hit-count-coef", type=float, default=200.0)
    parser.add_argument("--kill-count-coef", type=float, default=0.0)
    parser.add_argument("--ammo-coef", type=float, default=5.0)
    parser.add_argument("--progress-coef", type=float, default=0.25)
    parser.add_argument("--progress-gate-size", type=float, default=120.0)
    parser.add_argument("--progress-gate-reward", type=float, default=0.0)
    parser.add_argument("--combat-gate-damage", type=float, default=0.0)
    parser.add_argument("--combat-gate-hits", type=float, default=0.0)
    parser.add_argument("--combat-gate-kills", type=float, default=0.0)
    parser.add_argument("--safe-progress-coef", type=float, default=0.0)
    parser.add_argument("--unsafe-progress-penalty-coef", type=float, default=0.0)
    parser.add_argument("--health-loss-coef", type=float, default=0.0)
    parser.add_argument("--forward-action-bonus", type=float, default=0.03)
    parser.add_argument("--no-progress-penalty", type=float, default=0.01)
    parser.add_argument("--stall-penalty", type=float, default=0.0)
    parser.add_argument("--stall-threshold", type=int, default=25)
    parser.add_argument("--oscillation-penalty", type=float, default=0.0)
    parser.add_argument("--oscillation-window", type=int, default=14)
    parser.add_argument("--backward-penalty-coef", type=float, default=0.0)
    parser.add_argument("--backward-tolerance", type=float, default=60.0)
    parser.add_argument("--repeat-turn-penalty", type=float, default=0.03)
    parser.add_argument("--repeat-turn-threshold", type=int, default=20)
    parser.add_argument("--repeat-strafe-penalty", type=float, default=0.0)
    parser.add_argument("--repeat-strafe-threshold", type=int, default=20)
    parser.add_argument("--death-penalty", type=float, default=100.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args)
    model = PPO.load(args.model, device=args.device)

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        frame_stack = deque([obs[:, :, 0]] * args.frame_stack, maxlen=args.frame_stack)
        frames = [draw_overlay(env.get_rgb_screen(), ep, args.skill, 0, 0.0, 0.0, info)]
        done = False
        raw_sum = 0.0
        shaped_sum = 0.0
        step = 0
        best_progress = 0.0
        damage = 0.0
        hits = 0.0
        kills = 0.0
        action_counts = Counter()
        while not done:
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
            kills += float(info.get("kill_delta", 0.0))
            step += 1
            action_counts[info.get("action_name", str(action))] += 1
            frame_stack.append(obs[:, :, 0])
            frames.append(draw_overlay(env.get_rgb_screen(), ep, args.skill, step, raw_sum, shaped_sum, info))

        video_path = out_dir / f"episode_{ep:02d}_raw_{raw_sum:.0f}_shaped_{shaped_sum:.0f}_progress_{best_progress:.0f}_steps_{step}.mp4"
        imageio.mimsave(video_path, frames, fps=args.fps, quality=8)
        print(
            f"episode {ep}: raw={raw_sum:.2f}, shaped={shaped_sum:.2f}, "
            f"progress={best_progress:.1f}, damage={damage:.1f}, hits={hits:.1f}, kills={kills:.1f}, "
            f"actions={dict(action_counts)}, saved={video_path}"
        )
    env.close()


if __name__ == "__main__":
    main()
