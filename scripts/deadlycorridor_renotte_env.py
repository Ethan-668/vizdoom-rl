from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces


class DeadlyCorridorRenotteEnv(gym.Env):
    """
    DeadlyCorridor environment aligned with the commonly reproduced
    Nicholas Renotte / MichaelFish PPO curriculum recipe.

    Key choices:
    - original 7 single-button actions from deadly_corridor.cfg
    - frame skip 4
    - grayscale 80x160 observation with HUD cropped away
    - reward shaping dominated by HITCOUNT delta, not raw movement reward
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg_path,
        frame_skip=4,
        height=80,
        width=160,
        raw_reward_coef=1.0,
        damage_taken_coef=10.0,
        hit_count_coef=200.0,
        ammo_coef=0.0,
        death_penalty=0.0,
        window_visible=False,
    ):
        super().__init__()

        self.cfg_path = str(Path(cfg_path))
        self.frame_skip = int(frame_skip)
        self.height = int(height)
        self.width = int(width)
        self.raw_reward_coef = float(raw_reward_coef)
        self.damage_taken_coef = float(damage_taken_coef)
        self.hit_count_coef = float(hit_count_coef)
        self.ammo_coef = float(ammo_coef)
        self.death_penalty = float(death_penalty)

        self.game = vzd.DoomGame()
        self.game.load_config(self.cfg_path)
        self.game.set_window_visible(window_visible)
        self.game.set_sound_enabled(False)
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.init()

        self.buttons = list(self.game.get_available_buttons())
        self.variables = list(self.game.get_available_game_variables())
        self.variable_names = [str(var).replace("GameVariable.", "") for var in self.variables]

        self.actions = []
        self.action_names = []
        for idx, button in enumerate(self.buttons):
            action = [0] * len(self.buttons)
            action[idx] = 1
            self.actions.append(action)
            self.action_names.append(str(button).replace("Button.", ""))

        expected_actions = [
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "ATTACK",
            "MOVE_FORWARD",
            "MOVE_BACKWARD",
            "TURN_LEFT",
            "TURN_RIGHT",
        ]
        if self.action_names != expected_actions:
            raise RuntimeError(
                f"Unexpected action order: {self.action_names}. Expected {expected_actions}."
            )

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8,
        )

        self.prev_health = 100.0
        self.prev_damage_taken = 0.0
        self.prev_hit_count = 0.0
        self.prev_ammo = -1.0
        self.last_rgb_screen = np.zeros((240, 320, 3), dtype=np.uint8)

    @staticmethod
    def _screen_to_hwc(screen):
        if screen.ndim == 3 and screen.shape[0] in (1, 3):
            return np.transpose(screen, (1, 2, 0))
        return screen

    def _read_vars(self):
        state = self.game.get_state()
        values = {}
        if state is not None and state.game_variables is not None:
            for idx, name in enumerate(self.variable_names):
                if idx < len(state.game_variables):
                    values[name] = float(state.game_variables[idx])

        fallbacks = {
            "HEALTH": (vzd.GameVariable.HEALTH, self.prev_health),
            "DAMAGECOUNT": (vzd.GameVariable.DAMAGECOUNT, self.prev_damage_taken),
            "HITCOUNT": (vzd.GameVariable.HITCOUNT, self.prev_hit_count),
            "SELECTED_WEAPON_AMMO": (vzd.GameVariable.SELECTED_WEAPON_AMMO, self.prev_ammo),
        }
        for name, (var, fallback) in fallbacks.items():
            if name in values:
                continue
            try:
                values[name] = float(self.game.get_game_variable(var))
            except Exception:
                values[name] = fallback

        return (
            values.get("HEALTH", self.prev_health),
            values.get("DAMAGECOUNT", self.prev_damage_taken),
            values.get("HITCOUNT", self.prev_hit_count),
            values.get("SELECTED_WEAPON_AMMO", self.prev_ammo),
        )

    def _preprocess(self, screen):
        screen = self._screen_to_hwc(screen)
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, 100), interpolation=cv2.INTER_CUBIC)
        cropped = resized[: self.height, :]
        return cropped[:, :, None].astype(np.uint8)

    def _get_obs(self):
        state = self.game.get_state()
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        screen = self._screen_to_hwc(state.screen_buffer)
        self.last_rgb_screen = screen.copy()
        return self._preprocess(screen)

    def get_rgb_screen(self):
        state = self.game.get_state()
        if state is not None:
            screen = self._screen_to_hwc(state.screen_buffer)
            self.last_rgb_screen = screen.copy()
        return self.last_rgb_screen.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()

        health, damage_taken, hit_count, ammo = self._read_vars()
        self.prev_health = health
        self.prev_damage_taken = damage_taken
        self.prev_hit_count = hit_count
        self.prev_ammo = ammo

        return self._get_obs(), {
            "health": health,
            "damage": damage_taken,
            "hits": hit_count,
            "hit": hit_count,
            "ammo": ammo,
            "action_names": self.action_names,
            "variable_names": self.variable_names,
        }

    def step(self, action):
        action = int(action)
        action_vec = self.actions[action]
        action_name = self.action_names[action]

        old_health = self.prev_health
        old_damage_taken = self.prev_damage_taken
        old_hit_count = self.prev_hit_count
        old_ammo = self.prev_ammo

        raw_reward = float(self.game.make_action(action_vec, self.frame_skip))
        terminated = self.game.is_episode_finished()
        truncated = False

        health, damage_taken, hit_count, ammo = self._read_vars()
        is_dead = False
        if terminated:
            try:
                is_dead = bool(self.game.is_player_dead())
            except Exception:
                is_dead = health <= 0.0
            if is_dead:
                health = min(health, 0.0)

        health_loss = max(old_health - health, 0.0)
        damage_taken_delta = old_damage_taken - damage_taken
        hit_count_delta = hit_count - old_hit_count

        ammo_delta = 0.0
        if old_ammo >= 0.0 and ammo >= 0.0:
            ammo_delta = ammo - old_ammo

        shaped_reward = (
            self.raw_reward_coef * raw_reward
            + self.damage_taken_coef * damage_taken_delta
            + self.hit_count_coef * hit_count_delta
            + self.ammo_coef * ammo_delta
            - self.death_penalty * float(is_dead)
        )

        self.prev_health = health
        self.prev_damage_taken = damage_taken
        self.prev_hit_count = hit_count
        self.prev_ammo = ammo

        obs = self._get_obs()
        info = {
            "raw_reward": raw_reward,
            "shaped_reward": shaped_reward,
            "health": health,
            "damage": damage_taken,
            "hits": hit_count,
            "hit": hit_count,
            "ammo": ammo,
            "health_loss": health_loss,
            "damage_delta": max(damage_taken - old_damage_taken, 0.0),
            "hit_delta": max(hit_count_delta, 0.0),
            "damage_taken_delta": damage_taken_delta,
            "hit_count_delta": hit_count_delta,
            "ammo_delta": ammo_delta,
            "is_dead": is_dead,
            "action_name": action_name,
            "reward_terms": {
                "raw": self.raw_reward_coef * raw_reward,
                "damage_taken": self.damage_taken_coef * damage_taken_delta,
                "hit_count": self.hit_count_coef * hit_count_delta,
                "ammo": self.ammo_coef * ammo_delta,
                "death": -self.death_penalty * float(is_dead),
            },
        }

        return obs, float(shaped_reward), terminated, truncated, info

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass
