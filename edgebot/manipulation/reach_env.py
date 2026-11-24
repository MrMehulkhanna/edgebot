"""Learning-based manipulation: a continuous-control reaching task for the planar arm.

The agent outputs joint velocity commands and must bring the gripper tip to a
random target with a downward-facing gripper, the precondition for a grasp. PPO
learns this from reward alone; the result is compared against the analytic IK +
trajectory pipeline used by the workcell (``scripts/run_manipulation.py``).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from edgebot.manipulation.arm import PlanarArm, wrap


class ArmReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, dt: float = 0.05, qd_max: float = 1.5, max_steps: int = 100, tol: float = 0.015,
                 orient_weight: float = 0.05):
        self.arm = PlanarArm()
        self.dt, self.qd_max, self.max_steps, self.tol = dt, qd_max, max_steps, tol
        self.orient_weight = orient_weight
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)
        self.q = np.zeros(3)
        self.target = np.zeros(2)

    def _sample_target(self):
        while True:
            p = np.array([self.np_random.uniform(0.15, 0.55), self.np_random.uniform(0.03, 0.30)])
            if self.arm.ik_analytic(p[0], p[1], -np.pi / 2) is not None:
                return p

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.q = np.radians([110.0, -100.0, -100.0]) + self.np_random.normal(0, 0.1, 3)
        self.target = self._sample_target() if not options or "target" not in options else np.asarray(options["target"])
        self.t = 0
        self._prev = self._cost()
        return self._obs(), {}

    def _cost(self):
        ee = self.arm.fk(self.q)
        pos_err = np.linalg.norm(ee[:2] - self.target)
        ori_err = abs(wrap(ee[2] + np.pi / 2))
        return pos_err + self.orient_weight * ori_err

    def _obs(self):
        ee = self.arm.fk(self.q)
        err = self.target - ee[:2]
        return np.concatenate([np.sin(self.q), np.cos(self.q), ee[:2], err * 5, [np.sin(ee[2]), np.cos(ee[2])],
                               [np.linalg.norm(err) * 5]]).astype(np.float32)

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.q = np.clip(self.q + a * self.qd_max * self.dt, self.arm.q_min, self.arm.q_max)
        self.t += 1
        cost = self._cost()
        ee = self.arm.fk(self.q)
        pos_err = float(np.linalg.norm(ee[:2] - self.target))
        reward = 10.0 * (self._prev - cost) - 0.01 - 0.002 * float(np.sum(a ** 2))
        self._prev = cost
        success = pos_err < self.tol and abs(wrap(ee[2] + np.pi / 2)) < 0.15
        if success:
            reward += 5.0
        truncated = self.t >= self.max_steps
        return self._obs(), reward, bool(success), truncated, {"success": success, "pos_err": pos_err,
                                                               "collision": False}


class ArmVecEnv:
    """Synchronous vector of ArmReachEnv with auto-reset (same interface as SyncVecEnv)."""

    def __init__(self, n: int, seed: int = 0):
        from edgebot.envs.jetbot_nav import SyncVecEnv

        self._v = SyncVecEnv([ArmReachEnv for _ in range(n)], seed=seed)
        self.n = n
        self.observation_space = self._v.observation_space
        self.action_space = self._v.action_space

    def reset(self):
        return self._v.reset()

    def step(self, a):
        return self._v.step(a)
