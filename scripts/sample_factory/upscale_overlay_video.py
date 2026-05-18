import argparse
from pathlib import Path

import cv2


def draw_overlay(frame, lines):
    pad = 18
    line_h = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness = 2

    box_w = 590
    box_h = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (12 + box_w, 12 + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.56, frame, 0.44, 0)

    y = 12 + pad + 20
    for idx, line in enumerate(lines):
        color = (255, 255, 255) if idx else (120, 220, 255)
        cv2.putText(frame, line, (28, y), font, font_scale, color, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (28, y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        y += line_h
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fps", type=float, default=0.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 35.0
    fps = args.fps if args.fps > 0 else src_fps
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output_path}")

    lines = [
        "DeadlyCorridor - Sample Factory baseline",
        "Algo: APPO | Model: GRU-512 recurrent policy",
        "Training: 50M env steps, 16 workers x 8 envs",
        "Obs: RGB 128x72 | frameskip: train 4, eval 1",
        "Action: turn + forward/back + strafe + attack",
        "Reward scaling: 0.01 | game variable: HEALTH",
        "Eval video: pretrained Apocalypse-19/doom_deadly_corridor",
    ]

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        upscaled = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(upscaled, (0, 0), 1.1)
        sharpened = cv2.addWeighted(upscaled, 1.35, blurred, -0.35, 0)
        output = draw_overlay(sharpened, lines)
        writer.write(output)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"saved={output_path}, frames={frame_count}, fps={fps:.2f}, size={args.width}x{args.height}")


if __name__ == "__main__":
    main()
