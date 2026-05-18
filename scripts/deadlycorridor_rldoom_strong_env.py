from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces


class DeadlyCorridorRLDoomStrongEnv(gym.Env):
    """
    DeadlyCorridor environment for a stronger, player-like agent.

    This version follows the key lesson from ViZDoom and public DeadlyCorridor
    agents: Doom actions are button combinations, not just isolated buttons.
    The environment keeps the official 7 cfg buttons but exposes a compact
    discrete macro-action set with movement, turning, strafing and shooting
    combinations.
    """

    metadata = {"render_modes": []}

    DEFAULT_MACRO_ACTIONS = [
        ("NOOP", []),
        ("MOVE_FORWARD", ["MOVE_FORWARD"]),
        ("MOVE_BACKWARD", ["MOVE_BACKWARD"]),
        ("MOVE_LEFT", ["MOVE_LEFT"]),
        ("MOVE_RIGHT", ["MOVE_RIGHT"]),
        ("TURN_LEFT", ["TURN_LEFT"]),
        ("TURN_RIGHT", ["TURN_RIGHT"]),
        ("ATTACK", ["ATTACK"]),
        ("MOVE_FORWARD+MOVE_LEFT", ["MOVE_FORWARD", "MOVE_LEFT"]),
        ("MOVE_FORWARD+MOVE_RIGHT", ["MOVE_FORWARD", "MOVE_RIGHT"]),
        ("MOVE_FORWARD+TURN_LEFT", ["MOVE_FORWARD", "TURN_LEFT"]),
        ("MOVE_FORWARD+TURN_RIGHT", ["MOVE_FORWARD", "TURN_RIGHT"]),
        ("MOVE_BACKWARD+TURN_LEFT", ["MOVE_BACKWARD", "TURN_LEFT"]),
        ("MOVE_BACKWARD+TURN_RIGHT", ["MOVE_BACKWARD", "TURN_RIGHT"]),
        ("ATTACK+TURN_LEFT", ["ATTACK", "TURN_LEFT"]),
        ("ATTACK+TURN_RIGHT", ["ATTACK", "TURN_RIGHT"]),
        ("ATTACK+MOVE_LEFT", ["ATTACK", "MOVE_LEFT"]),
        ("ATTACK+MOVE_RIGHT", ["ATTACK", "MOVE_RIGHT"]),
        ("ATTACK+MOVE_FORWARD", ["ATTACK", "MOVE_FORWARD"]),
        ("ATTACK+MOVE_FORWARD+MOVE_LEFT", ["ATTACK", "MOVE_FORWARD", "MOVE_LEFT"]),
        ("ATTACK+MOVE_FORWARD+MOVE_RIGHT", ["ATTACK", "MOVE_FORWARD", "MOVE_RIGHT"]),
        ("ATTACK+MOVE_FORWARD+TURN_LEFT", ["ATTACK", "MOVE_FORWARD", "TURN_LEFT"]),
        ("ATTACK+MOVE_FORWARD+TURN_RIGHT", ["ATTACK", "MOVE_FORWARD", "TURN_RIGHT"]),
    ]

    def __init__(
        self,
        cfg_path,
        frame_skip=4,
        image_size=84,
        crop_top=20,
        crop_bottom=35,
        raw_reward_coef=0.02,
        damage_reward_coef=12.0,
        hit_reward_coef=3.0,
        health_loss_coef=6.0,
        attack_penalty=0.01,
        attack_miss_penalty=0.03,
        living_penalty=0.005,
        repeat_action_penalty=0.02,
        repeat_action_threshold=10,
        death_penalty=250.0,
        window_visible=False,
    ):
        super().__init__()

        self.cfg_path = str(Path(cfg_path))
        self.frame_skip = int(frame_skip)
        self.image_size = int(image_size)
        self.crop_top = int(crop_top)
        self.crop_bottom = int(crop_bottom)

        self.raw_reward_coef = float(raw_reward_coef)
        self.damage_reward_coef = float(damage_reward_coef)
        self.hit_reward_coef = float(hit_reward_coef)
        self.health_loss_coef = float(health_loss_coef)
        self.attack_penalty = float(attack_penalty)
        self.attack_miss_penalty = float(attack_miss_penalty)
        self.living_penalty = float(living_penalty)
        self.repeat_action_penalty = float(repeat_action_penalty)
        self.repeat_action_threshold = int(repeat_action_threshold)
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
        self.button_names = [str(button).replace("Button.", "") for button in self.buttons]
        self.variables = list(self.game.get_available_game_variables())
        self.variable_names = [str(var).replace("GameVariable.", "") for var in self.variables]

        expected_buttons = [
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "ATTACK",
            "MOVE_FORWARD",
            "MOVE_BACKWARD",
            "TURN_LEFT",
            "TURN_RIGHT",
        ]
        if self.button_names != expected_buttons:
            raise RuntimeError(
                "Unexpected DeadlyCorridor button order: "
                f"{self.button_names}. Expected {expected_buttons}."
            )

        self.button_index = {name: idx for idx, name in enumerate(self.button_names)}
        self.actions = []
        self.action_names = []
        for action_name, button_names in self.DEFAULT_MACRO_ACTIONS:
            action_vec = [0] * len(self.buttons)
            for button_name in button_names:
                action_vec[self.button_index[button_name]] = 1
            self.actions.append(action_vec)
            self.action_names.append(action_name)

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size, self.image_size, 1),
            dtype=np.uint8,
        )

        self.prev_health = 100.0
        self.prev_damage = 0.0
        self.prev_hits = 0.0
        self.prev_action = None
        self.repeat_action_count = 0
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
            "DAMAGECOUNT": (vzd.GameVariable.DAMAGECOUNT, self.prev_damage),
            "HITCOUNT": (vzd.GameVariable.HITCOUNT, self.prev_hits),
            "SELECTED_WEAPON_AMMO": (vzd.GameVariable.SELECTED_WEAPON_AMMO, -1.0),
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
            values.get("DAMAGECOUNT", self.prev_damage),
            values.get("HITCOUNT", self.prev_hits),
            values.get("SELECTED_WEAPON_AMMO", -1.0),
        )

    def _preprocess(self, screen):
        screen = self._screen_to_hwc(screen)
        if self.crop_top or self.crop_bottom:
            bottom = screen.shape[0] - self.crop_bottom if self.crop_bottom > 0 else screen.shape[0]
            screen = screen[self.crop_top:bottom, :, :]
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            gray,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        return resized[:, :, None].astype(np.uint8)

    def _get_obs(self):
        state = self.game.get_state()
        if state is None:
            return np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
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

        health, damage, hits, ammo = self._read_vars()
        self.prev_health = health
        self.prev_damage = damage
        self.prev_hits = hits
        self.prev_action = None
        self.repeat_action_count = 0

        return self._get_obs(), {
            "health": health,
            "damage": damage,
            "hits": hits,
            "hit": hits,
            "ammo": ammo,
            "button_names": self.button_names,
            "action_names": self.action_names,
            "variable_names": self.variable_names,
        }

    def step(self, action):
        action = int(action)
        action_vec = self.actions[action]
        action_name = self.action_names[action]

        old_health = self.prev_health
        old_damage = self.prev_damage
        old_hits = self.prev_hits

        raw_reward = float(self.game.make_action(action_vec, self.frame_skip))
        terminated = self.game.is_episode_finished()
        truncated = False

        health, damage, hits, ammo = self._read_vars()
        is_dead = False
        if terminated:
            try:
                is_dead = bool(self.game.is_player_dead())
            except Exception:
                is_dead = health <= 0.0
            if is_dead:
                health = min(health, 0.0)

        health_loss = max(old_health - health, 0.0)
        damage_delta = max(damage - old_damage, 0.0)
        hit_delta = max(hits - old_hits, 0.0)
        attack_used = "ATTACK" in action_name
        attack_missed = attack_used and damage_delta <= 0.0 and hit_delta <= 0.0

        if self.prev_action == action:
            self.repeat_action_count += 1
        else:
            self.repeat_action_count = 1
        self.prev_action = action
        repeat_over = max(self.repeat_action_count - self.repeat_action_threshold, 0)
        repeat_penalty = self.repeat_action_penalty * repeat_over

        shaped_reward = (
            self.raw_reward_coef * raw_reward
            + self.damage_reward_coef * damage_delta
            + self.hit_reward_coef * hit_delta
            - self.health_loss_coef * health_loss
            - self.attack_penalty * float(attack_used)
            - self.attack_miss_penalty * float(attack_missed)
            - self.living_penalty
            - repeat_penalty
            - self.death_penalty * float(is_dead)
        )

        self.prev_health = health
        self.prev_damage = damage
        self.prev_hits = hits

        obs = self._get_obs()
        info = {
            "raw_reward": raw_reward,
            "shaped_reward": shaped_reward,
            "health": health,
            "damage": damage,
            "hits": hits,
            "hit": hits,
            "ammo": ammo,
            "health_loss": health_loss,
            "damage_delta": damage_delta,
            "hit_delta": hit_delta,
            "attack_used": attack_used,
            "attack_missed": attack_missed,
            "is_dead": is_dead,
            "action_name": action_name,
            "action_vector": action_vec,
            "repeat_action_count": self.repeat_action_count,
            "reward_terms": {
                "raw": self.raw_reward_coef * raw_reward,
                "damage": self.damage_reward_coef * damage_delta,
                "hit": self.hit_reward_coef * hit_delta,
                "health_loss": -self.health_loss_coef * health_loss,
                "attack": -self.attack_penalty * float(attack_used),
                "attack_miss": -self.attack_miss_penalty * float(attack_missed),
                "living": -self.living_penalty,
                "repeat": -repeat_penalty,
                "death": -self.death_penalty * float(is_dead),
            },
        }
        return obs, float(shaped_reward), terminated, truncated, info

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass
