from pathlib import Path
import os
import time
import vizdoom as vzd


def find_scenario_cfg(keyword="deadly_corridor"):
    package_dir = Path(vzd.__file__).resolve().parent
    candidates = list(package_dir.rglob(f"*{keyword}*.cfg"))
    if not candidates:
        raise FileNotFoundError(f"No cfg file matching *{keyword}*.cfg in {package_dir}")
    return candidates[0]


def main():
    cfg_path = find_scenario_cfg("deadly_corridor")

    print("Using config:", cfg_path)
    print("A ViZDoom window should appear.")
    print("Click the game window first, then try controls.")
    print("Common controls: W/A/S/D, arrow keys, Ctrl, Space, mouse left.")
    print("Press Ctrl+C in this terminal to quit.")
    print("-" * 60)

    game = vzd.DoomGame()
    game.load_config(str(cfg_path))
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_ticrate(35)

    game.init()

    episode = 0

    try:
        while True:
            episode += 1
            game.new_episode()
            step = 0

            print(f"\nEpisode {episode} started.")

            while not game.is_episode_finished():
                game.advance_action()
                step += 1

                if step % 35 == 0:
                    state = game.get_state()
                    if state is not None:
                        print(
                            f"step={step:4d}, "
                            f"last_reward={game.get_last_reward():8.2f}, "
                            f"total_reward={game.get_total_reward():8.2f}, "
                            f"game_vars={state.game_variables}, "
                            f"last_action={game.get_last_action()}"
                        )

                time.sleep(0.01)

            print(f"Episode {episode} finished.")
            print("Total reward:", game.get_total_reward())
            print("Starting next episode in 2 seconds...")
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nExiting human play mode.")

    # 避免某些 WSL / ViZDoom 组合在析构时触发底层崩溃
    os._exit(0)


if __name__ == "__main__":
    main()
