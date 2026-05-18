from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces


class DeadlyCorridorRLDoomEnvV2(gym.Env):
    """
    RLDoom-style DeadlyCorridor environment with scaled reward shaping.

    The action space intentionally stays as one discrete action per original
    available button from the cfg:
    MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT,
    TURN_RIGHT.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg_path,
        frame_skip=4,
        image_size=84,
        raw_reward_coef=0.05,
        damage_reward_coef=10.0,
        hit_reward_coef=2.0,
        health_loss_coef=5.0,
        attack_penalty=0.02,
        death_penalty=200.0,
        window_visible=False,
    ):
        super().__init__()

        self.cfg_path = str(Path(cfg_path))
        self.frame_skip = int(frame_skip)
        self.image_size = int(image_size)

        self.raw_reward_coef = float(raw_reward_coef)
        self.damage_reward_coef = float(damage_reward_coef)
        self.hit_reward_coef = float(hit_reward_coef)
        self.health_loss_coef = float(health_loss_coef)
        self.attack_penalty = float(attack_penalty)
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
                "Unexpected DeadlyCorridor action order: "
                f"{self.action_names}. Expected {expected_actions}."
            )

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
        self.last_rgb_screen = np.zeros((240, 320, 3), dtype=np.uint8)

    def _var_from_state(self, name, default):
        if not hasattr(self, "_current_vars"):
            return default
        if name not in self._current_vars:
            return default
        return self._current_vars[name]

    def _read_vars(self):
        state = self.game.get_state()
        self._current_vars = {}

        if state is not None and state.game_variables is not None:
            for idx, name in enumerate(self.variable_names):
                if idx < len(state.game_variables):
                    self._current_vars[name] = float(state.game_variables[idx])

        health = self._var_from_state("HEALTH", self.prev_health)
        damage = self._var_from_state("DAMAGECOUNT", self.prev_damage)
        hits = self._var_from_state("HITCOUNT", self.prev_hits)
        ammo = self._var_from_state("SELECTED_WEAPON_AMMO", -1.0)

        # ViZDoom often still exposes variables after terminal states even when
        # get_state() is None. Use that as a fallback so death health loss is not
        # hidden by a missing screen state.
        fallback_vars = {
            "HEALTH": (vzd.GameVariable.HEALTH, health),
            "DAMAGECOUNT": (vzd.GameVariable.DAMAGECOUNT, damage),
            "HITCOUNT": (vzd.GameVariable.HITCOUNT, hits),
            "SELECTED_WEAPON_AMMO": (vzd.GameVariable.SELECTED_WEAPON_AMMO, ammo),
        }
        for name, (var, current) in fallback_vars.items():
            if name in self._current_vars:
                continue
            try:
                self._current_vars[name] = float(self.game.get_game_variable(var))
            except Exception:
                self._current_vars[name] = current

        return (
            self._current_vars.get("HEALTH", health),
            self._current_vars.get("DAMAGECOUNT", damage),
            self._current_vars.get("HITCOUNT", hits),
            self._current_vars.get("SELECTED_WEAPON_AMMO", ammo),
        )

    @staticmethod
    def _screen_to_hwc(screen):
        if screen.ndim == 3 and screen.shape[0] in (1, 3):
            return np.transpose(screen, (1, 2, 0))
        return screen

    def _preprocess(self, screen):
        screen = self._screen_to_hwc(screen)
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

        obs = self._get_obs()
        info = {
            "health": health,
            "damage": damage,
            "hits": hits,
            "hit": hits,
            "ammo": ammo,
            "action_names": self.action_names,
            "variable_names": self.variable_names,
        }
        return obs, info

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
        if terminated:
            try:
                if self.game.is_player_dead():
                    health = min(health, 0.0)
            except Exception:
                pass

        health_loss = max(old_health - health, 0.0)
        damage_delta = max(damage - old_damage, 0.0)
        hit_delta = max(hits - old_hits, 0.0)
        attack_used = action_name == "ATTACK"

        is_dead = False
        if terminated:
            try:
                is_dead = bool(self.game.is_player_dead())
            except Exception:
                is_dead = health <= 0.0

        shaped_reward = (
            self.raw_reward_coef * raw_reward
            + self.damage_reward_coef * damage_delta
            + self.hit_reward_coef * hit_delta
            - self.health_loss_coef * health_loss
            - self.attack_penalty * float(attack_used)
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
            "is_dead": is_dead,
            "action_name": action_name,
            "reward_terms": {
                "raw": self.raw_reward_coef * raw_reward,
                "damage": self.damage_reward_coef * damage_delta,
                "hit": self.hit_reward_coef * hit_delta,
                "health_loss": -self.health_loss_coef * health_loss,
                "attack": -self.attack_penalty * float(attack_used),
                "death": -self.death_penalty * float(is_dead),
            },
        }

        return obs, float(shaped_reward), terminated, truncated, info

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass
