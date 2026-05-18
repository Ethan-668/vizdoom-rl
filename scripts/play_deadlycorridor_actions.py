import cv2
import gymnasium as gym
import numpy as np
from vizdoom import gymnasium_wrapper


WINDOW_NAME = "ViZDoom DeadlyCorridor Manual Action Test"


def draw_overlay(frame, action_id, reward, total_reward, step, gamevariables, paused=False):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, (960, 720), interpolation=cv2.INTER_NEAREST)

    gv = np.asarray(gamevariables).round(2).tolist()

    lines = [
        "ViZDoom DeadlyCorridor Manual Test",
        "Press 0-7 to execute one action",
        "Hold 0-7 to repeat action",
        "Press r to reset, q to quit",
        f"Step: {step}",
        f"Last Action ID: {action_id}",
        f"Reward: {reward:.2f}",
        f"Total Reward: {total_reward:.2f}",
        f"GameVars: {gv}",
    ]

    if paused:
        lines.append("Episode ended. Press r to reset or q to quit.")

    y = 35
    for line in lines:
        cv2.putText(frame_bgr, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        y += 30

    return frame_bgr


def main():
    env = gym.make("VizdoomDeadlyCorridor-v1")

    obs, info = env.reset()
    total_reward = 0.0
    step = 0
    last_action = None
    last_reward = 0.0
    done = False

    print("Manual action test started.")
    print("Controls:")
    print("  0-7: execute corresponding discrete action")
    print("  r: reset")
    print("  q: quit")
    print()
    print("Action space:", env.action_space)

    while True:
        frame = draw_overlay(
            obs["screen"],
            action_id=last_action,
            reward=last_reward,
            total_reward=total_reward,
            step=step,
            gamevariables=obs["gamevariables"],
            paused=done,
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            obs, info = env.reset()
            total_reward = 0.0
            step = 0
            last_action = None
            last_reward = 0.0
            done = False
            print("\n[reset]")
            continue

        if done:
            continue

        if ord("0") <= key <= ord("7"):
            action = key - ord("0")
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            step += 1
            last_action = action
            last_reward = float(reward)
            done = terminated or truncated

            print(
                f"step={step:4d}, "
                f"action={action}, "
                f"reward={float(reward):8.2f}, "
                f"total={total_reward:8.2f}, "
                f"gamevars={obs['gamevariables']}, "
                f"done={done}"
            )

            if done:
                print("[episode ended] press r to reset or q to quit")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
