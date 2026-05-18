from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces


class DeadlyCorridorRLDoomEnv(gym.Env):
    """
    ViZDoom DeadlyCorridor environment rebuilt following common RL-Doom style ideas:

    - Use original 7 available buttons:
      MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT
    - Use 84x84 grayscale visual input
    - Use reward shaping based on:
      raw reward, damage count, hit count, health loss, ammo use if available
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg_path,
        frame_skip=4,
        image_size=84,
        damage_reward_coef=1.0,
        hit_reward_coef=0.5,
        health_loss_coef=2.0,
        attack_penalty=0.02,
        ammo_waste_penalty=0.10,
        death_penalty=50.0,
        window_visible=False,
    ):
        super().__init__()

        self.cfg_path = str(cfg_path)
        self.frame_skip = frame_skip
        self.image_size = image_size

        self.damage_reward_coef = damage_reward_coef
        self.hit_reward_coef = hit_reward_coef
        self.health_loss_coef = health_loss_coef
        self.attack_penalty = attack_penalty
        self.ammo_waste_penalty = ammo_waste_penalty
        self.death_penalty = death_penalty

        self.game = vzd.DoomGame()
        self.game.load_config(self.cfg_path)
        self.game.set_window_visible(window_visible)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.init()

        self.buttons = list(self.game.get_available_buttons())

        # One discrete action per original available button.
        self.actions = []
        self.action_names = []
        for button in self.buttons:
            action = [0] * len(self.buttons)
            idx = self.buttons.index(button)
            action[idx] = 1
            self.actions.append(action)
            self.action_names.append(str(button).replace("Button.", ""))

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(image_size, image_size, 1),
            dtype=np.uint8,
        )

        self.prev_health = None
        self.prev_damage = None
        self.prev_hit = None
        self.prev_ammo = None

    def _read_vars(self):
        state = self.game.get_state()
        if state is None or state.game_variables is None:
            return 100.0, 0.0, 0.0, -1.0

        vars_ = state.game_variables
        health = float(vars_[0]) if len(vars_) > 0 else 100.0
        damage = float(vars_[1]) if len(vars_) > 1 else 0.0
        hit = float(vars_[2]) if len(vars_) > 2 else 0.0
        ammo = float(vars_[3]) if len(vars_) > 3 else -1.0

        return health, damage, hit, ammo

    def _preprocess(self, screen):
        if screen.ndim == 3 and screen.shape[0] in (1, 3):
            screen = np.transpose(screen, (1, 2, 0))

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
        return self._preprocess(state.screen_buffer)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game.new_episode()

        health, damage, hit, ammo = self._read_vars()
        self.prev_health = health
        self.prev_damage = damage
        self.prev_hit = hit
        self.prev_ammo = ammo

        obs = self._get_obs()
        info = {
            "health": health,
            "damage": damage,
            "hit": hit,
            "ammo": ammo,
            "action_names": self.action_names,
        }
        return obs, info

    def step(self, action):
        action = int(action)
        action_vec = self.actions[action]
        action_name = self.action_names[action]

        old_health = self.prev_health
        old_damage = self.prev_damage
        old_hit = self.prev_hit
        old_ammo = self.prev_ammo

        raw_reward = float(self.game.make_action(action_vec, self.frame_skip))

        terminated = self.game.is_episode_finished()
        truncated = False

        health, damage, hit, ammo = self._read_vars()

        health_loss = max(old_health - health, 0.0)
        damage_delta = max(damage - old_damage, 0.0)
        hit_delta = max(hit - old_hit, 0.0)

        ammo_loss = 0.0
        if old_ammo is not None and old_ammo >= 0 and ammo >= 0:
            ammo_loss = max(old_ammo - ammo, 0.0)

        attack_used = "ATTACK" in action_name

        shaped_reward = raw_reward
        shaped_reward += self.damage_reward_coef * damage_delta
        shaped_reward += self.hit_reward_coef * hit_delta
        shaped_reward -= self.health_loss_coef * health_loss

        if attack_used:
            shaped_reward -= self.attack_penalty

        # If ammo is valid, penalize shots that do not cause hit/damage.
        if attack_used and ammo_loss > 0 and damage_delta <= 0 and hit_delta <= 0:
            shaped_reward -= self.ammo_waste_penalty * ammo_loss

        if terminated:
            try:
                if self.game.is_player_dead():
                    shaped_reward -= self.death_penalty
            except Exception:
                pass

        self.prev_health = health
        self.prev_damage = damage
        self.prev_hit = hit
        self.prev_ammo = ammo

        obs = self._get_obs()

        info = {
            "raw_reward": raw_reward,
            "shaped_reward": shaped_reward,
            "health": health,
            "damage": damage,
            "hit": hit,
            "ammo": ammo,
            "health_loss": health_loss,
            "damage_delta": damage_delta,
            "hit_delta": hit_delta,
            "ammo_loss": ammo_loss,
            "action_name": action_name,
        }

        return obs, float(shaped_reward), terminated, truncated, info

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass
