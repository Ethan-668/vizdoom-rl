import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_LABELS = {
    "PPO_1": "PPO 10k",
    "PPO_2": "PPO 100k",
    "PPO_3": "PPO 500k",
    "PPO_4": "PPO 1M",
    "A2C_1": "A2C 100k",
    "A2C_2": "A2C 500k",
}

TAGS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "time/fps",
]


def smooth(values, weight=0.6):
    if not values:
        return values

    smoothed = []
    last = values[0]
    for value in values:
        last = last * weight + value * (1 - weight)
        smoothed.append(last)
    return smoothed


def find_run_dirs(logdir):
    logdir = Path(logdir)
    run_dirs = sorted({p.parent for p in logdir.rglob("events.out.tfevents.*")})
    return run_dirs


def load_scalars(run_dir, tag):
    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()

    if tag not in accumulator.Tags().get("scalars", []):
        return None, None

    events = accumulator.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def export_tag_plot(logdir, outdir, tag, smoothing):
    run_dirs = find_run_dirs(logdir)

    plt.figure(figsize=(10, 6))

    has_data = False
    for run_dir in run_dirs:
        run_name = run_dir.name
        steps, values = load_scalars(run_dir, tag)

        if steps is None:
            continue

        values = smooth(values, weight=smoothing)
        label = RUN_LABELS.get(run_name, run_name)

        plt.plot(steps, values, label=label)
        has_data = True

    if not has_data:
        print(f"[skip] no data for tag: {tag}")
        plt.close()
        return

    title = tag.replace("/", " / ")
    filename = tag.replace("/", "_") + ".png"

    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel(tag)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = Path(outdir) / filename
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs/tensorboard")
    parser.add_argument("--outdir", type=str, default="results/figures")
    parser.add_argument("--smoothing", type=float, default=0.6)
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for tag in TAGS:
        export_tag_plot(
            logdir=args.logdir,
            outdir=args.outdir,
            tag=tag,
            smoothing=args.smoothing,
        )


if __name__ == "__main__":
    main()
