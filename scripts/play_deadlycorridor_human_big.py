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
    print("Controls:")
    print("  W / Up       : forward")
    print("  S / Down     : backward")
    print("  A            : move left")
    print("  D            : move right")
    print("  Left Arrow   : turn left")
    print("  Right Arrow  : turn right")
    print("  Space / Ctrl : attack")
    print("  Ctrl+C in terminal to quit")
    print("-" * 60)

    game = vzd.DoomGame()
    game.load_config(str(cfg_path))

    # 放大窗口
    game.set_screen_resolution(vzd.ScreenResolution.RES_1024X768)
    game.set_window_visible(True)

    # 人类控制模式
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_ticrate(35)

    # 尝试绑定更熟悉的按键
    game.add_game_args("+bind w +forward")
    game.add_game_args("+bind s +back")
    game.add_game_args("+bind a +moveleft")
    game.add_game_args("+bind d +moveright")
    game.add_game_args("+bind left +left")
    game.add_game_args("+bind right +right")
    game.add_game_args("+bind space +attack")
    game.add_game_args("+bind ctrl +attack")
    game.add_game_args("+bind mouse1 +attack")

    game.init()

    episode = 0

    try:
        while True:
            episode += 1
            game.new_episode()
            step = 0

            print(f"\nEpisode {episode} started.")
            print("Click the Doom window first, then use W/A/S/D, arrows, Space/Ctrl.")

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
            print("Next episode starts in 2 seconds...")
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nExiting human play mode.")

    os._exit(0)


if __name__ == "__main__":
    main()
