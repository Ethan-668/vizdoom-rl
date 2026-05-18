import gymnasium as gym
from vizdoom import gymnasium_wrapper

env = gym.make("VizdoomDeadlyCorridor-v1")

print("action space:", env.action_space)

# 尝试读取 ViZDoom 底层按钮
game = None
if hasattr(env.unwrapped, "game"):
    game = env.unwrapped.game
elif hasattr(env.unwrapped, "_game"):
    game = env.unwrapped._game

if game is not None:
    print("available buttons:")
    for i, button in enumerate(game.get_available_buttons()):
        print(i, button)
else:
    print("Could not access underlying ViZDoom game object.")

obs, info = env.reset()

print("-" * 50)
print("Testing one step for each action:")
for action in range(env.action_space.n):
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    print(
        f"action={action}, "
        f"reward={reward:.2f}, "
        f"terminated={terminated}, "
        f"truncated={truncated}, "
        f"gamevariables={obs['gamevariables']}, "
        f"info={info}"
    )

env.close()
