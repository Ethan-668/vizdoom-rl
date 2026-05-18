from deadlycorridor_rldoom_strong_env import DeadlyCorridorRLDoomStrongEnv


def main():
    env = DeadlyCorridorRLDoomStrongEnv(
        cfg_path="configs/deadlycorridor_rldoom/deadly_corridor_skill1.cfg",
        window_visible=False,
    )
    obs, info = env.reset()
    print("obs shape:", obs.shape, obs.dtype)
    print("button names:", info["button_names"])
    print("action space:", env.action_space)
    print("action names:")
    for idx, name in enumerate(info["action_names"]):
        print(f"  {idx}: {name}")
    print("variable names:", info["variable_names"])
    print("initial info:", info)

    total_raw = 0.0
    total_shaped = 0.0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_raw += info["raw_reward"]
        total_shaped += reward
        print(
            f"step={step + 1}, action={action}, action_name={info['action_name']}, "
            f"raw={info['raw_reward']:.2f}, shaped={reward:.2f}, "
            f"health={info['health']:.1f}, health_loss={info['health_loss']:.1f}, "
            f"damage_delta={info['damage_delta']:.1f}, hit_delta={info['hit_delta']:.1f}, "
            f"terms={info['reward_terms']}"
        )
        if terminated or truncated:
            break

    env.close()
    print("total raw:", total_raw)
    print("total shaped:", total_shaped)


if __name__ == "__main__":
    main()
