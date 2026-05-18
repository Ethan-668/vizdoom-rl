import argparse
import os

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from vizdoom import gymnasium_wrapper

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecTransposeImage


class ScreenGrayResize(gym.ObservationWrapper):
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        screen = obs["screen"]
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return resized[:, :, None].astype(np.uint8)


class ActionRemapWrapper(gym.ActionWrapper):
    """
    Remove suspected shortcut action 4.

    Original action space:
        Discrete(8)

    New action space:
        Discrete(7)

    Mapping:
        new 0 -> old 0
        new 1 -> old 1
        new 2 -> old 2
        new 3 -> old 3
        new 4 -> old 5
        new 5 -> old 6
        new 6 -> old 7

    Old action 4 is excluded.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_mapping = [0, 1, 2, 3, 5, 6, 7]
        self.action_space = spaces.Discrete(len(self.action_mapping))

    def action(self, act):
        return self.action_mapping[int(act)]


def make_env():
    def _init():
        env = gym.make("VizdoomDeadlyCorridor-v1")
        env = ActionRemapWrapper(env)
        env = ScreenGrayResize(env, size=84)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs("models/deadlycorridor_noback", exist_ok=True)
    os.makedirs("models/deadlycorridor_noback/checkpoints", exist_ok=True)
    os.makedirs("logs/tensorboard/deadlycorridor_noback", exist_ok=True)

    env = DummyVecEnv([make_env() for _ in range(args.n_envs)])
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4, channels_order="first")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path="models/deadlycorridor_noback/checkpoints",
        name_prefix="ppo_deadlycorridor_noback",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=2.5e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.02,
        verbose=1,
        tensorboard_log="logs/tensorboard/deadlycorridor_noback",
        device=args.device,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save("models/deadlycorridor_noback/ppo_deadlycorridor_noback_final")
    env.close()

    print("Training finished.")
    print("Saved model to models/deadlycorridor_noback/ppo_deadlycorridor_noback_final.zip")


if __name__ == "__main__":
    main()
