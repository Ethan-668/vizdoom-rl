import argparse
import os
import shutil
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecTransposeImage

from deadlycorridor_progress_env import DeadlyCorridorProgressEnv


def make_env(skill, args):
    def _init():
        return DeadlyCorridorProgressEnv(
            cfg_path=f"{args.cfg_dir}/deadly_corridor_skill{skill}.cfg",
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

    return _init


def make_vec_env(skill, args):
    env = DummyVecEnv([make_env(skill, args) for _ in range(args.n_envs)])
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
    parser.add_argument("--model-dir", type=str, default="models/deadlycorridor_progress")
    parser.add_argument("--tensorboard-dir", type=str, default="logs/tensorboard/deadlycorridor_progress")
    parser.add_argument("--cfg-dir", type=str, default="configs/deadlycorridor_progress")
    parser.add_argument("--init-model", type=str, default="")
    parser.add_argument("--timesteps-per-skill", type=int, default=80_000)
    parser.add_argument("--start-skill", type=int, default=1)
    parser.add_argument("--end-skill", type=int, default=5)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--action-mode", choices=["single", "macro", "combat_macro"], default="single")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--ent-coef", type=float, default=0.02)
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

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.model_dir}/checkpoints", exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    print("DeadlyCorridor progress curriculum")
    for key, value in vars(args).items():
        print(f"  {key}={value}")

    model = None
    total_so_far = 0
    for skill in range(args.start_skill, args.end_skill + 1):
        env = make_vec_env(skill, args)
        callback = CheckpointCallback(
            save_freq=max(args.timesteps_per_skill // args.n_envs, 1),
            save_path=f"{args.model_dir}/checkpoints",
            name_prefix=f"ppo_progress_skill{skill}",
        )
        if model is None:
            if args.init_model:
                model = PPO.load(args.init_model, env=env, device=args.device)
                model.tensorboard_log = args.tensorboard_dir
            else:
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
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.ent_coef = args.ent_coef
        model.clip_range = get_schedule_fn(args.clip_range)

        print("=" * 80)
        print(f"Training progress stage: skill {skill}")
        print("=" * 80)
        model.learn(
            total_timesteps=args.timesteps_per_skill,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name="PPO_Progress_Curriculum",
            progress_bar=True,
        )
        total_so_far += args.timesteps_per_skill
        stage_path = f"{args.model_dir}/ppo_progress_after_skill{skill}_{total_so_far}.zip"
        model.save(stage_path)
        backup = backup_model(stage_path)
        print(f"Saved stage model: {stage_path}")
        if backup:
            print(f"Backed up stage model: {backup}")
        env.close()

    final_path = f"{args.model_dir}/ppo_progress_curriculum_final.zip"
    model.save(final_path)
    backup = backup_model(final_path)
    print(f"Saved final model to {final_path}")
    if backup:
        print(f"Backed up final model to {backup}")


if __name__ == "__main__":
    main()
