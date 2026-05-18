from deadlycorridor_rldoom_env import DeadlyCorridorRLDoomEnv

env = DeadlyCorridorRLDoomEnv(
    cfg_path="configs/deadlycorridor_rldoom/deadly_corridor_skill1.cfg",
    window_visible=False,
)

obs, info = env.reset()

print("obs shape:", obs.shape, obs.dtype)
print("action space:", env.action_space)
print("info:", info)

total_shaped = 0.0
total_raw = 0.0

for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_shaped += reward
    total_raw += info["raw_reward"]

    print(
        f"step={step + 1}, "
        f"action={action}, "
        f"action_name={info['action_name']}, "
        f"raw={info['raw_reward']:.2f}, "
        f"shaped={reward:.2f}, "
        f"health={info['health']:.1f}, "
        f"damage_delta={info['damage_delta']:.1f}, "
        f"hit_delta={info['hit_delta']:.1f}"
    )

    if terminated or truncated:
        break

env.close()

print("total raw:", total_raw)
print("total shaped:", total_shaped)
