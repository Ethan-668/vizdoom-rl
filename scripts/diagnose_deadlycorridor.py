from collections import Counter, deque

import cv2
import gymnasium as gym
import numpy as np
from vizdoom import gymnasium_wrapper
from stable_baselines3 import PPO


def preprocess_screen(screen, size=84):
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


env = gym.make("VizdoomDeadlyCorridor-v1")
model = PPO.load("models/deadlycorridor/ppo_deadlycorridor_500k.zip", device="cuda")

obs, info = env.reset()
frame = preprocess_screen(obs["screen"])
frame_stack = deque([frame] * 4, maxlen=4)

done = False
step = 0
reward_sum = 0.0
action_counter = Counter()

print("initial gamevariables:", obs["gamevariables"])
print("action space:", env.action_space)
print("-" * 80)

while not done:
    stacked_obs = np.stack(list(frame_stack), axis=0)
    action, _ = model.predict(stacked_obs, deterministic=True)
    action = int(action)

    obs, reward, terminated, truncated, info = env.step(action)

    reward_sum += float(reward)
    step += 1
    action_counter[action] += 1

    frame = preprocess_screen(obs["screen"])
    frame_stack.append(frame)

    # 打印每一步，方便看 reward 是怎么来的
    print(
        f"step={step:4d}, "
        f"action={action}, "
        f"reward={float(reward):8.2f}, "
        f"reward_sum={reward_sum:8.2f}, "
        f"terminated={terminated}, "
        f"truncated={truncated}, "
        f"gamevariables={obs['gamevariables']}, "
        f"info={info}"
    )

    done = terminated or truncated

env.close()

print("-" * 80)
print("final reward:", reward_sum)
print("total steps:", step)
print("action counts:", action_counter)
