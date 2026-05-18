from pathlib import Path
import time
import vizdoom as vzd


def find_scenario_cfg(keyword="deadly_corridor"):
    package_dir = Path(vzd.__file__).resolve().parent
    candidates = list(package_dir.rglob(f"*{keyword}*.cfg"))

    if not candidates:
        print("Could not find scenario config automatically.")
        print("ViZDoom package dir:", package_dir)
        raise FileNotFoundError(f"No cfg file matching *{keyword}*.cfg")

    return candidates[0]


def main():
    cfg_path = find_scenario_cfg("deadly_corridor")

    print("Using config:", cfg_path)
    print("Starting ViZDoom in human/spectator mode...")
    print("A game window should appear.")
    print("Use keyboard/mouse in the ViZDoom window.")
    print("Close the window or press Ctrl+C in terminal to stop.")

    game = vzd.DoomGame()
    game.load_config(str(cfg_path))

    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)

    # 可选：降低速度，让你更容易观察
    game.set_ticrate(35)

    game.init()
    game.new_episode()

    while not game.is_episode_finished():
        game.advance_action()
        state = game.get_state()

        if state is not None:
            reward = game.get_last_reward()
            total_reward = game.get_total_reward()
            vars_ = state.game_variables
            last_action = game.get_last_action()

            print(
                f"reward={reward:8.2f}, "
                f"total={total_reward:8.2f}, "
                f"game_vars={vars_}, "
                f"last_action={last_action}"
            )

        time.sleep(0.02)

    print("Episode finished.")
    print("Total reward:", game.get_total_reward())
    game.close()


if __name__ == "__main__":
    main()
