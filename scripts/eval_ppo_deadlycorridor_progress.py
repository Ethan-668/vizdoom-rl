import argparse
from collections import Counter, deque

import numpy as np
from stable_baselines3 import PPO

from deadlycorridor_progress_env import DeadlyCorridorProgressEnv


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


def evaluate(args):
    env = make_env(args)
    model = PPO.load(args.model, device=args.device)

    shaped_rewards = []
    raw_rewards = []
    lengths = []
    health_losses = []
    damage_gains = []
    hit_gains = []
    kill_gains = []
    best_progresses = []
    action_counts = Counter()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        frame_stack = deque([obs[:, :, 0]] * args.frame_stack, maxlen=args.frame_stack)

        done = False
        ep_shaped = 0.0
        ep_raw = 0.0
        ep_health_loss = 0.0
        ep_damage = 0.0
        ep_hits = 0.0
        ep_kills = 0.0
        ep_best_progress = 0.0
        step = 0
        ep_action_counts = Counter()

        while not done:
            stacked_obs = np.stack(list(frame_stack), axis=0)
            action, _ = model.predict(stacked_obs, deterministic=not args.stochastic)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            action_name = info.get("action_name", str(action))
            ep_shaped += float(reward)
            ep_raw += float(info.get("raw_reward", 0.0))
            ep_health_loss += float(info.get("health_loss", 0.0))
            ep_damage += float(info.get("damage_delta", 0.0))
            ep_hits += float(info.get("hit_delta", 0.0))
            ep_kills += float(info.get("kill_delta", 0.0))
            ep_best_progress = max(ep_best_progress, float(info.get("best_progress", 0.0)))
            step += 1
            action_counts[action_name] += 1
            ep_action_counts[action_name] += 1
            frame_stack.append(obs[:, :, 0])

        shaped_rewards.append(ep_shaped)
        raw_rewards.append(ep_raw)
        lengths.append(step)
        health_losses.append(ep_health_loss)
        damage_gains.append(ep_damage)
        hit_gains.append(ep_hits)
        kill_gains.append(ep_kills)
        best_progresses.append(ep_best_progress)
        print(
            f"episode {ep + 1}: shaped={ep_shaped:.2f}, raw={ep_raw:.2f}, "
            f"steps={step}, health_loss={ep_health_loss:.1f}, damage={ep_damage:.1f}, "
            f"hits={ep_hits:.1f}, kills={ep_kills:.1f}, best_progress={ep_best_progress:.1f}, "
            f"actions={dict(ep_action_counts)}"
        )

    env.close()
    print("-" * 80)
    print(f"model: {args.model}")
    print(f"skill: {args.skill}")
    print(f"episodes: {args.episodes}")
    print(f"deterministic: {not args.stochastic}")
    print(f"mean shaped reward: {np.mean(shaped_rewards):.2f}")
    print(f"std shaped reward: {np.std(shaped_rewards):.2f}")
    print(f"mean raw reward: {np.mean(raw_rewards):.2f}")
    print(f"std raw reward: {np.std(raw_rewards):.2f}")
    print(f"min raw reward: {np.min(raw_rewards):.2f}")
    print(f"max raw reward: {np.max(raw_rewards):.2f}")
    print(f"mean steps: {np.mean(lengths):.2f}")
    print(f"mean health loss: {np.mean(health_losses):.2f}")
    print(f"mean damage: {np.mean(damage_gains):.2f}")
    print(f"mean hits: {np.mean(hit_gains):.2f}")
    print(f"mean kills: {np.mean(kill_gains):.2f}")
    print(f"mean best progress: {np.mean(best_progresses):.2f}")
    print(f"action counts: {action_counts}")


def add_args(parser):
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--skill", type=int, default=5)
    parser.add_argument("--cfg-dir", type=str, default="configs/deadlycorridor_progress")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--action-mode", choices=["single", "macro", "combat_macro"], default="single")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
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


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
