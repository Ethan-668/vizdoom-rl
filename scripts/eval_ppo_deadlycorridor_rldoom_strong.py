import argparse
from collections import Counter, deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from deadlycorridor_rldoom_strong_env import DeadlyCorridorRLDoomStrongEnv


def make_env(args):
    return DeadlyCorridorRLDoomStrongEnv(
        cfg_path=f"configs/deadlycorridor_rldoom/deadly_corridor_skill{args.skill}.cfg",
        frame_skip=args.frame_skip,
        image_size=84,
        raw_reward_coef=args.raw_reward_coef,
        damage_reward_coef=args.damage_reward_coef,
        hit_reward_coef=args.hit_reward_coef,
        health_loss_coef=args.health_loss_coef,
        attack_penalty=args.attack_penalty,
        attack_miss_penalty=args.attack_miss_penalty,
        living_penalty=args.living_penalty,
        repeat_action_penalty=args.repeat_action_penalty,
        repeat_action_threshold=args.repeat_action_threshold,
        death_penalty=args.death_penalty,
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
    action_counts = Counter()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        frame_stack = deque([obs[:, :, 0]] * 4, maxlen=4)

        done = False
        ep_shaped = 0.0
        ep_raw = 0.0
        ep_health_loss = 0.0
        ep_damage = 0.0
        ep_hits = 0.0
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

        print(
            f"episode {ep + 1}: shaped={ep_shaped:.2f}, raw={ep_raw:.2f}, "
            f"steps={step}, health_loss={ep_health_loss:.1f}, damage={ep_damage:.1f}, "
            f"hits={ep_hits:.1f}, actions={dict(ep_action_counts)}"
        )

    env.close()
    Path("results/deadlycorridor_rldoom_strong").mkdir(parents=True, exist_ok=True)

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
    print(f"action counts: {action_counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--skill", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--raw-reward-coef", type=float, default=0.02)
    parser.add_argument("--damage-reward-coef", type=float, default=10.0)
    parser.add_argument("--hit-reward-coef", type=float, default=2.0)
    parser.add_argument("--health-loss-coef", type=float, default=2.0)
    parser.add_argument("--attack-penalty", type=float, default=0.01)
    parser.add_argument("--attack-miss-penalty", type=float, default=0.03)
    parser.add_argument("--living-penalty", type=float, default=0.005)
    parser.add_argument("--repeat-action-penalty", type=float, default=0.02)
    parser.add_argument("--repeat-action-threshold", type=int, default=10)
    parser.add_argument("--death-penalty", type=float, default=80.0)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
