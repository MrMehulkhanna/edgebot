"""Lidar safety shield for a learned navigation policy (the project's novel component).

The RL policy proposes an action. Before it reaches the motors, the shield
predicts where the robot will be over the next ``horizon`` control steps if it
holds that action (exact unicycle model) and checks that predicted footprint
against the current 24-beam lidar scan. If the action would bring the robot
closer than ``robot_r + margin`` to any lidar return, the shield replaces it with
the safe candidate action that is closest to what the policy wanted.

Only the lidar scan and the robot's own kinematics are used: no map, no
retraining, no change to the network. That makes it a drop-in guard for an
INT8-quantised policy, whose occasional wrong actions are exactly what the
shield is meant to catch.
"""

from __future__ import annotations

import numpy as np

from edgebot.envs.jetbot_nav import DISCRETE_ACTIONS

# candidate (v fraction, w fraction) commands the shield may fall back to
_FALLBACK = np.concatenate([DISCRETE_ACTIONS, [[0.3, 0.0], [0.3, 0.8], [0.3, -0.8]]]).astype(np.float32)


class LidarShield:
    def __init__(self, env, horizon: int = 5, margin: float = 0.04):
        self.env, self.horizon, self.margin = env, horizon, margin
        self.interventions = 0
        self.steps = 0

    def _points(self) -> np.ndarray:
        """Lidar returns as (x, y) points in the robot frame (x forward)."""
        r = self.env.lidar()
        keep = r < self.env.max_range - 1e-6
        a = self.env.ray_angles[keep]
        return np.stack([r[keep] * np.cos(a), r[keep] * np.sin(a)], axis=1)

    def _safe(self, cmd: np.ndarray, pts: np.ndarray) -> bool:
        if len(pts) == 0:
            return True
        v, w = cmd[0] * self.env.v_max, cmd[1] * self.env.w_max
        x = y = th = 0.0
        for _ in range(self.horizon):
            thm = th + 0.5 * w * self.env.dt
            x, y, th = x + v * np.cos(thm) * self.env.dt, y + v * np.sin(thm) * self.env.dt, th + w * self.env.dt
            if np.hypot(pts[:, 0] - x, pts[:, 1] - y).min() < self.env.robot_r + self.margin:
                return False
        return True

    def __call__(self, action):
        self.steps += 1
        cmd = self.env._to_cmd(action)
        pts = self._points()
        if self._safe(cmd, pts):
            return action
        self.interventions += 1
        # nearest safe fallback in (v, w) space; pure spins never move the robot so one always exists
        order = np.argsort(np.linalg.norm(_FALLBACK - cmd, axis=1))
        best = next((c for c in _FALLBACK[order] if self._safe(c, pts)), _FALLBACK[3])
        if self.env.action_mode == "discrete":
            d = np.linalg.norm(DISCRETE_ACTIONS - best, axis=1)
            return int(np.argmin(d))
        return np.array([2 * best[0] - 1, best[1]], dtype=np.float32)

    @property
    def intervention_rate(self) -> float:
        return self.interventions / max(self.steps, 1)
