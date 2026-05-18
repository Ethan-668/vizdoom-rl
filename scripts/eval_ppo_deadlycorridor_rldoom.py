import argparse
from collections import Counter, deque

import numpy as np
from stable_baselines3 import PPO

from deadlycorridor_rldoom_env import DeadlyCorridorRLDoomEnv


def evaluate(model_path, skill=5, episodes=20, deterministic=True, device="cuda"):
    env = DeadlyCorridorRLDoomEnv(
        cfg_path=f"configs/deadlycorridor_rldoom/deadly_corridor_skill{skill}.cfg",
        frame_skip=4,
        image_size=84,
        damage_reward_coef=1.0,
        hit_reward_coef=0.5,
        health_loss_coef=2.0,
        attack_penalty=0.02,
        ammo_waste_penalty=0.10,
        death_penalty=50.0,
        window_visible=False,
    )

    model = PPO.load(model_path, device=device)

    shaped_rewards = []
    raw_rewards = []
    lengths = []
    health_losses = []
    damage_gains = []
    hit_gains = []
    action_counts = Counter()

    for ep in range(episodes):
        obs, info = env.reset()
        frame_stack = deque([obs[:, :, 0]] * 4, maxlen=4)

        done = False
        ep_shaped = 0.0
        ep_raw = 0.0
        ep_health_loss = 0.0
        ep_damage = 0.0
        ep_hit = 0.0
        step = 0
        ep_action_counts = Counter()

        while not done:
            stacked_obs = np.stack(list(frame_stack), axis=0)
            action, _ = model.predict(stacked_obs, deterministic=deterministic)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            action_name = info.get("action_name", str(action))

            ep_shaped += float(reward)
            ep_raw += float(info.get("raw_reward", 0.0))
            ep_health_loss += float(info.get("health_loss", 0.0))
            ep_damage += float(info.get("damage_delta", 0.0))
            ep_hit += float(info.get("hit_delta", 0.0))
            step += 1

            action_counts[action_name] += 1
            ep_action_counts[action_name] += 1

            frame_stack.append(obs[:, :, 0])

        shaped_rewards.append(ep_shaped)
        raw_rewards.append(ep_raw)
        lengths.append(step)
        health_losses.append(ep_health_loss)
        damage_gains.append(ep_damage)
        hit_gains.append(ep_hit)

        print(
            f"episode {ep + 1}: "
            f"shaped={ep_shaped:.2f}, raw={ep_raw:.2f}, "
            f"steps={step}, health_loss={ep_health_loss:.1f}, "
            f"damage={ep_damage:.1f}, hits={ep_hit:.1f}, "
            f"actions={dict(ep_action_counts)}"
        )

    env.close()

    print("-" * 80)
    print(f"model: {model_path}")
    print(f"skill: {skill}")
    print(f"episodes: {episodes}")
    print(f"deterministic: {deterministic}")
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
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        skill=args.skill,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        device=args.device,
    )


if __name__ == "__main__":
    main()
