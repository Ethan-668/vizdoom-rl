import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecTransposeImage

from deadlycorridor_rldoom_env import DeadlyCorridorRLDoomEnv


def make_env(skill):
    def _init():
        return DeadlyCorridorRLDoomEnv(
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
    return _init


def make_vec_env(skill, n_envs):
    env = DummyVecEnv([make_env(skill) for _ in range(n_envs)])
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4, channels_order="first")
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps-per-skill", type=int, default=40_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs("models/deadlycorridor_rldoom", exist_ok=True)
    os.makedirs("models/deadlycorridor_rldoom/checkpoints", exist_ok=True)
    os.makedirs("logs/tensorboard/deadlycorridor_rldoom", exist_ok=True)

    model = None
    total_so_far = 0

    for skill in range(1, 6):
        print("=" * 80)
        print(f"Training curriculum stage: skill {skill}")
        print("=" * 80)

        env = make_vec_env(skill=skill, n_envs=args.n_envs)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.timesteps_per_skill // args.n_envs, 1),
            save_path="models/deadlycorridor_rldoom/checkpoints",
            name_prefix=f"ppo_rldoom_skill{skill}",
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
                tensorboard_log="logs/tensorboard/deadlycorridor_rldoom",
                device=args.device,
            )
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=args.timesteps_per_skill,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            tb_log_name=f"PPO_RLDoom_Curriculum",
            progress_bar=True,
        )

        total_so_far += args.timesteps_per_skill
        model.save(f"models/deadlycorridor_rldoom/ppo_rldoom_after_skill{skill}_{total_so_far}.zip")
        env.close()

    model.save("models/deadlycorridor_rldoom/ppo_rldoom_curriculum_final.zip")

    print("Curriculum training finished.")
    print("Saved final model to models/deadlycorridor_rldoom/ppo_rldoom_curriculum_final.zip")


if __name__ == "__main__":
    main()
