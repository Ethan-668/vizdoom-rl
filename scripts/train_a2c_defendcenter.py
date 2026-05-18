import argparse
import os

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from vizdoom import gymnasium_wrapper

from stable_baselines3 import A2C
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


def make_env():
    def _init():
        env = gym.make("VizdoomDefendCenter-v1")
        env = ScreenGrayResize(env, size=84)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)

    env = DummyVecEnv([make_env() for _ in range(args.n_envs)])
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4, channels_order="first")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path="models/checkpoints",
        name_prefix="a2c_defendcenter",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = A2C(
        policy="CnnPolicy",
        env=env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        verbose=1,
        tensorboard_log="logs/tensorboard",
        device=args.device,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save("models/a2c_defendcenter_final")
    env.close()

    print("A2C training finished.")
    print("Saved model to models/a2c_defendcenter_final.zip")


if __name__ == "__main__":
    main()
