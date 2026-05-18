import argparse
import os
import shutil
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecTransposeImage

from deadlycorridor_rldoom_env_v2 import DeadlyCorridorRLDoomEnvV2


MODEL_DIR = "models/deadlycorridor_rldoom_v2"
TB_DIR = "logs/tensorboard/deadlycorridor_rldoom_v2"


def make_env(skill, args):
    def _init():
        return DeadlyCorridorRLDoomEnvV2(
            cfg_path=f"configs/deadlycorridor_rldoom/deadly_corridor_skill{skill}.cfg",
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

    return _init


def make_vec_env(skill, n_envs, args):
    env = DummyVecEnv([make_env(skill, args) for _ in range(n_envs)])
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4, channels_order="first")
    return env


def backup_model(path):
    if not os.path.exists(path):
        return None
    stem, ext = os.path.splitext(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{stem}_backup_{stamp}{ext}"
    shutil.copy2(path, backup_path)
    return backup_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps-per-skill", type=int, default=40_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--raw-reward-coef", type=float, default=0.05)
    parser.add_argument("--damage-reward-coef", type=float, default=10.0)
    parser.add_argument("--hit-reward-coef", type=float, default=2.0)
    parser.add_argument("--health-loss-coef", type=float, default=5.0)
    parser.add_argument("--attack-penalty", type=float, default=0.02)
    parser.add_argument("--death-penalty", type=float, default=200.0)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(f"{MODEL_DIR}/checkpoints", exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    print("DeadlyCorridor RLDoom v2 reward coefficients:")
    print(f"  raw_reward_coef={args.raw_reward_coef}")
    print(f"  damage_reward_coef={args.damage_reward_coef}")
    print(f"  hit_reward_coef={args.hit_reward_coef}")
    print(f"  health_loss_coef={args.health_loss_coef}")
    print(f"  attack_penalty={args.attack_penalty}")
    print(f"  death_penalty={args.death_penalty}")

    model = None
    total_so_far = 0

    for skill in range(1, 6):
        print("=" * 80)
        print(f"Training DeadlyCorridor RLDoom v2 curriculum stage: skill {skill}")
        print("=" * 80)

        env = make_vec_env(skill=skill, n_envs=args.n_envs, args=args)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.timesteps_per_skill // args.n_envs, 1),
            save_path=f"{MODEL_DIR}/checkpoints",
            name_prefix=f"ppo_rldoom_v2_skill{skill}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        if model is None:
            model = PPO(
                policy="CnnPolicy",
                env=env,
                learning_rate=2.5e-4,
                n_steps=512,
                batch_size=128,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.1,
                ent_coef=0.02,
                verbose=1,
                tensorboard_log=TB_DIR,
                device=args.device,
            )
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=args.timesteps_per_skill,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            tb_log_name="PPO_RLDoom_V2_Curriculum",
            progress_bar=True,
        )

        total_so_far += args.timesteps_per_skill
        stage_path = f"{MODEL_DIR}/ppo_rldoom_v2_after_skill{skill}_{total_so_far}.zip"
        model.save(stage_path)
        backup_path = backup_model(stage_path)
        print(f"Saved stage model: {stage_path}")
        if backup_path:
            print(f"Backed up stage model: {backup_path}")
        env.close()

    final_path = f"{MODEL_DIR}/ppo_rldoom_v2_curriculum_final.zip"
    model.save(final_path)
    backup_path = backup_model(final_path)

    print("Curriculum training finished.")
    print(f"Saved final model to {final_path}")
    if backup_path:
        print(f"Backed up final model to {backup_path}")


if __name__ == "__main__":
    main()
