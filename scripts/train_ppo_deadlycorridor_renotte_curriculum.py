import argparse
import os
import shutil
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecTransposeImage

from deadlycorridor_renotte_env import DeadlyCorridorRenotteEnv


DEFAULT_MODEL_DIR = "models/deadlycorridor_renotte"
DEFAULT_TB_DIR = "logs/tensorboard/deadlycorridor_renotte"


def make_env(skill, args):
    def _init():
        return DeadlyCorridorRenotteEnv(
            cfg_path=f"configs/deadlycorridor_rldoom/deadly_corridor_skill{skill}.cfg",
            frame_skip=args.frame_skip,
            raw_reward_coef=args.raw_reward_coef,
            damage_taken_coef=args.damage_taken_coef,
            hit_count_coef=args.hit_count_coef,
            ammo_coef=args.ammo_coef,
            death_penalty=args.death_penalty,
            window_visible=False,
        )

    return _init


def make_vec_env(skill, n_envs, args):
    env = DummyVecEnv([make_env(skill, args) for _ in range(n_envs)])
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=args.frame_stack, channels_order="first")
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
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tensorboard-dir", type=str, default=DEFAULT_TB_DIR)
    parser.add_argument("--timesteps-per-skill", type=int, default=40_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--n-steps", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--raw-reward-coef", type=float, default=1.0)
    parser.add_argument("--damage-taken-coef", type=float, default=10.0)
    parser.add_argument("--hit-count-coef", type=float, default=200.0)
    parser.add_argument("--ammo-coef", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.model_dir}/checkpoints", exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    print("DeadlyCorridor Renotte-style PPO curriculum")
    for key, value in vars(args).items():
        print(f"  {key}={value}")

    model = None
    total_so_far = 0

    for skill in range(1, 6):
        print("=" * 80)
        print(f"Training Renotte-style curriculum stage: skill {skill}")
        print("=" * 80)

        env = make_vec_env(skill=skill, n_envs=args.n_envs, args=args)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.timesteps_per_skill // args.n_envs, 1),
            save_path=f"{args.model_dir}/checkpoints",
            name_prefix=f"ppo_renotte_skill{skill}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        if model is None:
            model = PPO(
                "CnnPolicy",
                env,
                verbose=1,
                tensorboard_log=args.tensorboard_dir,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                clip_range=args.clip_range,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                ent_coef=args.ent_coef,
                device=args.device,
            )
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=args.timesteps_per_skill,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            tb_log_name="PPO_Renotte_Curriculum",
            progress_bar=True,
        )

        total_so_far += args.timesteps_per_skill
        stage_path = f"{args.model_dir}/ppo_renotte_after_skill{skill}_{total_so_far}.zip"
        model.save(stage_path)
        backup_path = backup_model(stage_path)
        print(f"Saved stage model: {stage_path}")
        if backup_path:
            print(f"Backed up stage model: {backup_path}")
        env.close()

    final_path = f"{args.model_dir}/ppo_renotte_curriculum_final.zip"
    model.save(final_path)
    backup_path = backup_model(final_path)
    print(f"Saved final model to {final_path}")
    if backup_path:
        print(f"Backed up final model to {backup_path}")


if __name__ == "__main__":
    main()
