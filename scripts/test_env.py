import gymnasium as gym
from vizdoom import gymnasium_wrapper

env = gym.make("VizdoomDefendCenter-v1")

obs, info = env.reset()
print("obs keys:", obs.keys())
print("screen shape:", obs["screen"].shape, obs["screen"].dtype)
print("gamevariables:", obs["gamevariables"].shape, obs["gamevariables"].dtype)
print("action space:", env.action_space)

total_reward = 0.0

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        obs, info = env.reset()

print("random policy test finished")
print("total reward:", total_reward)

env.close()
