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
    print()
    print("FPS-style controls:")
    print("  W          : move forward")
    print("  S          : move backward")
    print("  A          : strafe left")
    print("  D          : strafe right")
    print("  Mouse      : turn / aim")
    print("  Mouse Left : attack")
    print("  Space      : attack backup")
    print("  Ctrl+C in terminal to quit")
    print()
    print("Important:")
    print("  1. Click inside the Doom window first.")
    print("  2. If mouse does not turn, click the window again.")
    print("  3. WSLg may not lock the mouse perfectly, but keyboard movement should work.")
    print("-" * 70)

    game = vzd.DoomGame()
    game.load_config(str(cfg_path))

    # Bigger window.
    game.set_screen_resolution(vzd.ScreenResolution.RES_1024X768)
    game.set_window_visible(True)

    # Human-controlled mode.
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_ticrate(35)

    # Force FPS-like key bindings.
    game.add_game_args("+bind w +forward")
    game.add_game_args("+bind s +back")
    game.add_game_args("+bind a +moveleft")
    game.add_game_args("+bind d +moveright")

    # Backup arrow keys.
    game.add_game_args("+bind up +forward")
    game.add_game_args("+bind down +back")
    game.add_game_args("+bind left +left")
    game.add_game_args("+bind right +right")

    # Fire.
    game.add_game_args("+bind mouse1 +attack")
    game.add_game_args("+bind space +attack")
    game.add_game_args("+bind ctrl +attack")

    # Mouse / aim related settings.
    # These are ZDoom-style console arguments passed to the engine.
    game.add_game_args("+freelook 1")
    game.add_game_args("+lookstrafe 0")
    game.add_game_args("+m_yaw 1.0")
    game.add_game_args("+m_pitch 1.0")
    game.add_game_args("+mouse_sensitivity 5.0")
    game.add_game_args("+grabmouse 1")

    game.init()

    episode = 0

    try:
        while True:
            episode += 1
            game.new_episode()
            step = 0

            print(f"\nEpisode {episode} started.")
            print("Click inside the Doom window. Use WASD + mouse + left click.")

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

    # Avoid occasional ViZDoom shutdown segfault under WSL.
    os._exit(0)


if __name__ == "__main__":
    main()
