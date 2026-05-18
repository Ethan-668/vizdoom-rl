from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces


class DeadlyCorridorProgressEnv(gym.Env):
    """
    DeadlyCorridor phase-2 environment: keep the learned combat behavior, but
    add explicit corridor-progress shaping using POSITION_X/Y.

    Progress is rewarded only when the agent reaches a new farthest projection
    along the start-to-current axis, which avoids paying for oscillation.
    """

    metadata = {"render_modes": []}

    MACRO_ACTIONS = [
        ("MOVE_FORWARD", ["MOVE_FORWARD"]),
        ("MOVE_LEFT", ["MOVE_LEFT"]),
        ("MOVE_RIGHT", ["MOVE_RIGHT"]),
        ("TURN_LEFT", ["TURN_LEFT"]),
        ("TURN_RIGHT", ["TURN_RIGHT"]),
        ("ATTACK", ["ATTACK"]),
        ("MOVE_FORWARD+MOVE_LEFT", ["MOVE_FORWARD", "MOVE_LEFT"]),
        ("MOVE_FORWARD+MOVE_RIGHT", ["MOVE_FORWARD", "MOVE_RIGHT"]),
        ("MOVE_FORWARD+TURN_LEFT", ["MOVE_FORWARD", "TURN_LEFT"]),
        ("MOVE_FORWARD+TURN_RIGHT", ["MOVE_FORWARD", "TURN_RIGHT"]),
        ("ATTACK+MOVE_FORWARD", ["ATTACK", "MOVE_FORWARD"]),
        ("ATTACK+MOVE_LEFT", ["ATTACK", "MOVE_LEFT"]),
        ("ATTACK+MOVE_RIGHT", ["ATTACK", "MOVE_RIGHT"]),
        ("ATTACK+TURN_LEFT", ["ATTACK", "TURN_LEFT"]),
        ("ATTACK+TURN_RIGHT", ["ATTACK", "TURN_RIGHT"]),
        ("ATTACK+MOVE_FORWARD+MOVE_LEFT", ["ATTACK", "MOVE_FORWARD", "MOVE_LEFT"]),
        ("ATTACK+MOVE_FORWARD+MOVE_RIGHT", ["ATTACK", "MOVE_FORWARD", "MOVE_RIGHT"]),
        ("ATTACK+MOVE_FORWARD+TURN_LEFT", ["ATTACK", "MOVE_FORWARD", "TURN_LEFT"]),
        ("ATTACK+MOVE_FORWARD+TURN_RIGHT", ["ATTACK", "MOVE_FORWARD", "TURN_RIGHT"]),
    ]
    COMBAT_MACRO_ACTIONS = [
        ("ATTACK", ["ATTACK"]),
        ("TURN_LEFT", ["TURN_LEFT"]),
        ("TURN_RIGHT", ["TURN_RIGHT"]),
        ("MOVE_FORWARD", ["MOVE_FORWARD"]),
        ("MOVE_BACKWARD", ["MOVE_BACKWARD"]),
        ("MOVE_FORWARD+TURN_LEFT", ["MOVE_FORWARD", "TURN_LEFT"]),
        ("MOVE_FORWARD+TURN_RIGHT", ["MOVE_FORWARD", "TURN_RIGHT"]),
        ("ATTACK+TURN_LEFT", ["ATTACK", "TURN_LEFT"]),
        ("ATTACK+TURN_RIGHT", ["ATTACK", "TURN_RIGHT"]),
        ("ATTACK+MOVE_FORWARD", ["ATTACK", "MOVE_FORWARD"]),
        ("ATTACK+MOVE_FORWARD+TURN_LEFT", ["ATTACK", "MOVE_FORWARD", "TURN_LEFT"]),
        ("ATTACK+MOVE_FORWARD+TURN_RIGHT", ["ATTACK", "MOVE_FORWARD", "TURN_RIGHT"]),
    ]

    def __init__(
        self,
        cfg_path,
        frame_skip=4,
        height=80,
        width=160,
        raw_reward_coef=0.02,
        damage_reward_coef=10.0,
        hit_count_coef=200.0,
        kill_count_coef=0.0,
        ammo_coef=5.0,
        progress_coef=0.25,
        progress_gate_size=120.0,
        progress_gate_reward=0.0,
        combat_gate_damage=0.0,
        combat_gate_hits=0.0,
        combat_gate_kills=0.0,
        safe_progress_coef=0.0,
        unsafe_progress_penalty_coef=0.0,
        health_loss_coef=0.0,
        forward_action_bonus=0.03,
        no_progress_penalty=0.01,
        stall_penalty=0.0,
        stall_threshold=25,
        oscillation_penalty=0.0,
        oscillation_window=14,
        combat_reset_on_hit=True,
        backward_penalty_coef=0.0,
        backward_tolerance=60.0,
        repeat_turn_penalty=0.03,
        repeat_turn_threshold=20,
        repeat_strafe_penalty=0.0,
        repeat_strafe_threshold=20,
        death_penalty=100.0,
        action_mode="single",
        window_visible=False,
    ):
        super().__init__()
        self.cfg_path = str(Path(cfg_path))
        self.frame_skip = int(frame_skip)
        self.height = int(height)
        self.width = int(width)

        self.raw_reward_coef = float(raw_reward_coef)
        self.damage_reward_coef = float(damage_reward_coef)
        self.hit_count_coef = float(hit_count_coef)
        self.kill_count_coef = float(kill_count_coef)
        self.ammo_coef = float(ammo_coef)
        self.progress_coef = float(progress_coef)
        self.progress_gate_size = float(progress_gate_size)
        self.progress_gate_reward = float(progress_gate_reward)
        self.combat_gate_damage = float(combat_gate_damage)
        self.combat_gate_hits = float(combat_gate_hits)
        self.combat_gate_kills = float(combat_gate_kills)
        self.safe_progress_coef = float(safe_progress_coef)
        self.unsafe_progress_penalty_coef = float(unsafe_progress_penalty_coef)
        self.health_loss_coef = float(health_loss_coef)
        self.forward_action_bonus = float(forward_action_bonus)
        self.no_progress_penalty = float(no_progress_penalty)
        self.stall_penalty = float(stall_penalty)
        self.stall_threshold = int(stall_threshold)
        self.oscillation_penalty = float(oscillation_penalty)
        self.oscillation_window = int(oscillation_window)
        self.combat_reset_on_hit = bool(combat_reset_on_hit)
        self.backward_penalty_coef = float(backward_penalty_coef)
        self.backward_tolerance = float(backward_tolerance)
        self.repeat_turn_penalty = float(repeat_turn_penalty)
        self.repeat_turn_threshold = int(repeat_turn_threshold)
        self.repeat_strafe_penalty = float(repeat_strafe_penalty)
        self.repeat_strafe_threshold = int(repeat_strafe_threshold)
        self.death_penalty = float(death_penalty)
        self.action_mode = action_mode

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
        self.button_index = {name: idx for idx, name in enumerate(self.button_names)}

        if self.action_mode == "single":
            action_specs = [(name, [name]) for name in self.button_names]
        elif self.action_mode == "macro":
            action_specs = self.MACRO_ACTIONS
        elif self.action_mode == "combat_macro":
            action_specs = self.COMBAT_MACRO_ACTIONS
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        self.actions = []
        self.action_names = []
        for action_name, button_names in action_specs:
            action_vec = [0] * len(self.buttons)
            for button_name in button_names:
                action_vec[self.button_index[button_name]] = 1
            self.actions.append(action_vec)
            self.action_names.append(action_name)

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
        self.prev_kill_count = 0.0
        self.prev_ammo = -1.0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.progress_axis = np.array([1.0, 0.0], dtype=np.float32)
        self.best_progress = 0.0
        self.progress_gate_index = 0
        self.total_damage = 0.0
        self.total_hits = 0.0
        self.total_kills = 0.0
        self.no_progress_steps = 0
        self.forward_backward_steps = 0
        self.same_turn_count = 0
        self.same_strafe_count = 0
        self.prev_action_name = ""
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
            "KILLCOUNT": (vzd.GameVariable.KILLCOUNT, self.prev_kill_count),
            "SELECTED_WEAPON_AMMO": (vzd.GameVariable.SELECTED_WEAPON_AMMO, self.prev_ammo),
            "POSITION_X": (vzd.GameVariable.POSITION_X, self.prev_x),
            "POSITION_Y": (vzd.GameVariable.POSITION_Y, self.prev_y),
            "ANGLE": (vzd.GameVariable.ANGLE, 0.0),
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
            values.get("KILLCOUNT", self.prev_kill_count),
            values.get("SELECTED_WEAPON_AMMO", self.prev_ammo),
            values.get("POSITION_X", self.prev_x),
            values.get("POSITION_Y", self.prev_y),
            values.get("ANGLE", 0.0),
        )

    def _preprocess(self, screen):
        screen = self._screen_to_hwc(screen)
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, 100), interpolation=cv2.INTER_CUBIC)
        return resized[: self.height, :, None].astype(np.uint8)

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

    def _project_progress(self, x, y):
        pos = np.array([x - self.start_x, y - self.start_y], dtype=np.float32)
        return float(np.dot(pos, self.progress_axis))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()

        health, damage, hits, kills, ammo, x, y, angle = self._read_vars()
        self.prev_health = health
        self.prev_damage_taken = damage
        self.prev_hit_count = hits
        self.prev_kill_count = kills
        self.prev_ammo = ammo
        self.prev_x = x
        self.prev_y = y
        self.start_x = x
        self.start_y = y
        self.best_progress = 0.0
        self.progress_gate_index = 0
        self.total_damage = 0.0
        self.total_hits = 0.0
        self.total_kills = 0.0
        self.no_progress_steps = 0
        self.forward_backward_steps = 0
        self.same_turn_count = 0
        self.same_strafe_count = 0
        self.prev_action_name = ""

        # Estimate corridor axis from a short forward probe, then restart.
        try:
            forward = [0] * len(self.buttons)
            forward[self.button_index["MOVE_FORWARD"]] = 1
            self.game.make_action(forward, self.frame_skip)
            _, _, _, _, _, probe_x, probe_y, _ = self._read_vars()
            delta = np.array([probe_x - x, probe_y - y], dtype=np.float32)
            norm = float(np.linalg.norm(delta))
            if norm > 1e-3:
                self.progress_axis = delta / norm
            self.game.new_episode()
            health, damage, hits, kills, ammo, x, y, angle = self._read_vars()
            self.prev_health = health
            self.prev_damage_taken = damage
            self.prev_hit_count = hits
            self.prev_kill_count = kills
            self.prev_ammo = ammo
            self.prev_x = x
            self.prev_y = y
            self.start_x = x
            self.start_y = y
        except Exception:
            pass

        return self._get_obs(), {
            "health": health,
            "damage": damage,
            "hits": hits,
            "kills": kills,
            "ammo": ammo,
            "x": x,
            "y": y,
            "angle": angle,
            "progress": 0.0,
            "action_names": self.action_names,
            "variable_names": self.variable_names,
        }

    def step(self, action):
        action = int(action)
        action_vec = self.actions[action]
        action_name = self.action_names[action]

        old_health = self.prev_health
        old_damage = self.prev_damage_taken
        old_hits = self.prev_hit_count
        old_kills = self.prev_kill_count
        old_ammo = self.prev_ammo

        raw_reward = float(self.game.make_action(action_vec, self.frame_skip))
        terminated = self.game.is_episode_finished()
        truncated = False

        health, damage, hits, kills, ammo, x, y, angle = self._read_vars()
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
        hit_delta = hits - old_hits
        kill_delta = kills - old_kills
        ammo_delta = ammo - old_ammo if old_ammo >= 0.0 and ammo >= 0.0 else 0.0
        self.total_damage += damage_delta
        self.total_hits += max(hit_delta, 0.0)
        self.total_kills += max(kill_delta, 0.0)

        progress = self._project_progress(x, y)
        progress_delta = max(progress - self.best_progress, 0.0)
        self.best_progress = max(self.best_progress, progress)
        if progress_delta > 1e-3:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        crossed_gates = 0
        safe_crossed_gates = 0
        unsafe_crossed_gates = 0
        required_damage_for_gate = 0.0
        required_hits_for_gate = 0.0
        required_kills_for_gate = 0.0
        combat_ready = True
        if self.progress_gate_size > 0.0:
            new_gate_index = int(max(self.best_progress, 0.0) // self.progress_gate_size)
            crossed_gates = max(new_gate_index - self.progress_gate_index, 0)
            required_damage_for_gate = self.combat_gate_damage * new_gate_index
            required_hits_for_gate = self.combat_gate_hits * new_gate_index
            required_kills_for_gate = self.combat_gate_kills * new_gate_index
            combat_ready = (
                self.total_damage >= required_damage_for_gate
                and self.total_hits >= required_hits_for_gate
                and self.total_kills >= required_kills_for_gate
            )
            if crossed_gates > 0:
                if combat_ready:
                    safe_crossed_gates = crossed_gates
                else:
                    unsafe_crossed_gates = crossed_gates
            self.progress_gate_index = max(self.progress_gate_index, new_gate_index)
        elif self.combat_gate_damage > 0.0 or self.combat_gate_hits > 0.0:
            combat_ready = (
                self.total_damage >= self.combat_gate_damage
                and self.total_hits >= self.combat_gate_hits
                and self.total_kills >= self.combat_gate_kills
            )
        safe_progress_delta = progress_delta if combat_ready else 0.0
        unsafe_progress_delta = progress_delta if not combat_ready else 0.0
        backward_distance = max(self.best_progress - progress - self.backward_tolerance, 0.0)
        stall_penalty = self.stall_penalty * max(self.no_progress_steps - self.stall_threshold, 0)
        backward_penalty = self.backward_penalty_coef * backward_distance

        has_forward = "MOVE_FORWARD" in action_name
        has_backward = "MOVE_BACKWARD" in action_name
        has_combat_gain = damage_delta > 0.0 or hit_delta > 0.0 or kill_delta > 0.0
        if self.combat_reset_on_hit and has_combat_gain:
            self.forward_backward_steps = 0
        elif has_forward or has_backward:
            self.forward_backward_steps += 1
        else:
            self.forward_backward_steps = max(self.forward_backward_steps - 1, 0)
        oscillation_penalty = self.oscillation_penalty * max(
            self.forward_backward_steps - self.oscillation_window,
            0,
        )
        is_pure_turn = action_name in ("TURN_LEFT", "TURN_RIGHT")
        is_pure_strafe = action_name in ("MOVE_LEFT", "MOVE_RIGHT")
        if is_pure_turn and action_name == self.prev_action_name:
            self.same_turn_count += 1
        elif is_pure_turn:
            self.same_turn_count = 1
        else:
            self.same_turn_count = 0
        if is_pure_strafe and action_name == self.prev_action_name:
            self.same_strafe_count += 1
        elif is_pure_strafe:
            self.same_strafe_count = 1
        else:
            self.same_strafe_count = 0
        self.prev_action_name = action_name
        turn_repeat_penalty = self.repeat_turn_penalty * max(
            self.same_turn_count - self.repeat_turn_threshold,
            0,
        )
        strafe_repeat_penalty = self.repeat_strafe_penalty * max(
            self.same_strafe_count - self.repeat_strafe_threshold,
            0,
        )

        shaped_reward = (
            self.raw_reward_coef * raw_reward
            + self.damage_reward_coef * damage_delta
            + self.hit_count_coef * hit_delta
            + self.kill_count_coef * kill_delta
            + self.ammo_coef * ammo_delta
            + self.progress_coef * progress_delta
            + self.safe_progress_coef * safe_progress_delta
            + self.progress_gate_reward * safe_crossed_gates
            - self.unsafe_progress_penalty_coef * unsafe_progress_delta
            - self.progress_gate_reward * unsafe_crossed_gates
            - self.health_loss_coef * health_loss
            + self.forward_action_bonus * float(has_forward)
            - self.no_progress_penalty * float(progress_delta <= 0.0)
            - stall_penalty
            - oscillation_penalty
            - backward_penalty
            - turn_repeat_penalty
            - strafe_repeat_penalty
            - self.death_penalty * float(is_dead)
        )

        self.prev_health = health
        self.prev_damage_taken = damage
        self.prev_hit_count = hits
        self.prev_kill_count = kills
        self.prev_ammo = ammo
        self.prev_x = x
        self.prev_y = y

        obs = self._get_obs()
        info = {
            "raw_reward": raw_reward,
            "shaped_reward": shaped_reward,
            "health": health,
            "damage": damage,
            "hits": hits,
            "kills": kills,
            "ammo": ammo,
            "x": x,
            "y": y,
            "angle": angle,
            "progress": progress,
            "best_progress": self.best_progress,
            "progress_delta": progress_delta,
            "crossed_gates": crossed_gates,
            "safe_crossed_gates": safe_crossed_gates,
            "unsafe_crossed_gates": unsafe_crossed_gates,
            "progress_gate_index": self.progress_gate_index,
            "combat_ready": combat_ready,
            "required_damage_for_gate": required_damage_for_gate,
            "required_hits_for_gate": required_hits_for_gate,
            "required_kills_for_gate": required_kills_for_gate,
            "total_damage": self.total_damage,
            "total_hits": self.total_hits,
            "total_kills": self.total_kills,
            "no_progress_steps": self.no_progress_steps,
            "forward_backward_steps": self.forward_backward_steps,
            "backward_distance": backward_distance,
            "health_loss": health_loss,
            "damage_delta": damage_delta,
            "hit_delta": max(hit_delta, 0.0),
            "kill_delta": max(kill_delta, 0.0),
            "ammo_delta": ammo_delta,
            "is_dead": is_dead,
            "action_name": action_name,
            "reward_terms": {
                "raw": self.raw_reward_coef * raw_reward,
                "damage": self.damage_reward_coef * damage_delta,
                "hit_count": self.hit_count_coef * hit_delta,
                "kill_count": self.kill_count_coef * kill_delta,
                "ammo": self.ammo_coef * ammo_delta,
                "progress": self.progress_coef * progress_delta,
                "safe_progress": self.safe_progress_coef * safe_progress_delta,
                "progress_gate": self.progress_gate_reward * safe_crossed_gates,
                "unsafe_progress": -self.unsafe_progress_penalty_coef * unsafe_progress_delta,
                "unsafe_gate": -self.progress_gate_reward * unsafe_crossed_gates,
                "health_loss": -self.health_loss_coef * health_loss,
                "forward": self.forward_action_bonus * float(has_forward),
                "no_progress": -self.no_progress_penalty * float(progress_delta <= 0.0),
                "stall": -stall_penalty,
                "oscillation": -oscillation_penalty,
                "backward": -backward_penalty,
                "turn_repeat": -turn_repeat_penalty,
                "strafe_repeat": -strafe_repeat_penalty,
                "death": -self.death_penalty * float(is_dead),
            },
        }
        return obs, float(shaped_reward), terminated, truncated, info

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass
