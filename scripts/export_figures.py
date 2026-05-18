from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LOGDIR = Path("logs/tensorboard")
OUTDIR = Path("final_artifacts/results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


RUN_LABELS = {
    "PPO_1": "PPO 10k",
    "PPO_2": "PPO 100k",
    "PPO_3": "PPO 500k",
    "PPO_4": "PPO 1M",
    "A2C_1": "A2C 100k",
    "A2C_2": "A2C 500k",
    "DQN_1": "DQN 100k",
    "DQN_2": "DQN 500k",
}


def find_event_dirs(logdir):
    return sorted({p.parent for p in logdir.rglob("events.out.tfevents.*")})


def smooth(values, weight=0.6):
    if not values:
        return values
    smoothed = []
    last = values[0]
    for value in values:
        last = last * weight + value * (1 - weight)
        smoothed.append(last)
    return smoothed


def load_scalar(run_dir, tag):
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return None, None
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def export_tensorboard_curve(tag, filename, title, ylabel):
    run_dirs = find_event_dirs(LOGDIR)

    plt.figure(figsize=(10, 6))
    has_data = False

    for run_dir in run_dirs:
        run_name = run_dir.name
        steps, values = load_scalar(run_dir, tag)

        if steps is None:
            continue

        values = smooth(values, weight=0.6)
        label = RUN_LABELS.get(run_name, run_name)

        plt.plot(steps, values, label=label)
        has_data = True

    if not has_data:
        print(f"[skip] no data for {tag}")
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = OUTDIR / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


def export_eval_bar_chart():
    methods = ["Random", "A2C-500k", "DQN-500k", "PPO-500k", "PPO-1M"]
    means = [0.20, 0.90, 11.35, 21.20, 21.15]
    stds = [0.98, 0.70, 4.56, 2.82, 6.13]

    plt.figure(figsize=(9, 6))
    plt.bar(methods, means, yerr=stds, capsize=5)
    plt.title("Final Evaluation Results on ViZDoom DefendCenter")
    plt.xlabel("Method")
    plt.ylabel("Mean Reward over Evaluation Episodes")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = OUTDIR / "final_eval_bar_chart.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


def main():
    export_tensorboard_curve(
        tag="rollout/ep_rew_mean",
        filename="training_reward_curve.png",
        title="Training Reward Curve",
        ylabel="Episode Reward Mean",
    )

    export_tensorboard_curve(
        tag="rollout/ep_len_mean",
        filename="training_episode_length_curve.png",
        title="Episode Length Curve",
        ylabel="Episode Length Mean",
    )

    export_tensorboard_curve(
        tag="time/fps",
        filename="training_fps_curve.png",
        title="Training FPS Curve",
        ylabel="FPS",
    )

    export_eval_bar_chart()


if __name__ == "__main__":
    main()
