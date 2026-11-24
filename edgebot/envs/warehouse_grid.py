"""Tabular warehouse grid world (finite MDP).

The robot starts at the loading bay ``S`` and must reach the packing station ``G``.
Shelves ``#`` block motion. A strip of oil spill cells ``x`` lies along the
shortest route: entering one costs -100 and teleports the robot back to ``S``
(the warehouse version of Sutton & Barto's Cliff Walking, Example 6.6).

With ``slip > 0`` the floor is slippery: the chosen action is executed with
probability ``1 - slip`` and each perpendicular action with ``slip / 2``. The full
transition model ``P[s][a] = [(prob, s', r, done), ...]`` is exposed so the same
environment serves model-based Dynamic Programming and model-free MC / TD control.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DEFAULT_MAP = (
    "..........",
    ".##.##.##.",
    "..........",
    "..........",
    "SxxxxxxxxG",
)

# up, right, down, left
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("up", "right", "down", "left")
ARROWS = ("↑", "→", "↓", "←")


class WarehouseGridEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(self, layout=DEFAULT_MAP, slip: float = 0.0, step_reward: float = -1.0,
                 spill_reward: float = -100.0, max_steps: int = 200, render_mode=None):
        self.grid = np.array([list(r) for r in layout])
        self.nrow, self.ncol = self.grid.shape
        self.slip = float(slip)
        self.step_reward = step_reward
        self.spill_reward = spill_reward
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.n_states = self.nrow * self.ncol
        self.n_actions = 4
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

        self.start = self._find("S")
        self.goal = self._find("G")
        self.P = self._build_model()
        self.s = self.start
        self.t = 0

    # ------------------------------------------------------------------ model
    def _find(self, ch: str) -> int:
        r, c = np.argwhere(self.grid == ch)[0]
        return int(r * self.ncol + c)

    def to_rc(self, s: int):
        return divmod(int(s), self.ncol)

    def cell(self, s: int) -> str:
        r, c = self.to_rc(s)
        return self.grid[r, c]

    def _move(self, s: int, a: int) -> int:
        r, c = self.to_rc(s)
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.nrow and 0 <= nc < self.ncol) or self.grid[nr, nc] == "#":
            return s
        return nr * self.ncol + nc

    def _outcome(self, s: int, a: int):
        ns = self._move(s, a)
        ch = self.cell(ns)
        if ch == "x":
            return self.start, self.spill_reward, False
        if ch == "G":
            return ns, self.step_reward, True
        return ns, self.step_reward, False

    def _build_model(self):
        P = {}
        for s in range(self.n_states):
            P[s] = {}
            terminal = self.cell(s) in ("G", "#")
            for a in range(self.n_actions):
                if terminal:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                if self.slip > 0:
                    probs = {a: 1 - self.slip, (a + 1) % 4: self.slip / 2, (a + 3) % 4: self.slip / 2}
                else:
                    probs = {a: 1.0}
                agg = {}
                for aa, p in probs.items():
                    ns, r, d = self._outcome(s, aa)
                    key = (ns, r, d)
                    agg[key] = agg.get(key, 0.0) + p
                P[s][a] = [(p, ns, r, d) for (ns, r, d), p in agg.items()]
        return P

    def valid_states(self):
        return [s for s in range(self.n_states) if self.cell(s) not in ("#",)]

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = int(options["start"]) if options and "start" in options else self.start
        self.t = 0
        return self.s, {}

    def step(self, a: int):
        transitions = self.P[self.s][int(a)]
        probs = [t[0] for t in transitions]
        i = self.np_random.choice(len(transitions), p=probs)
        _, ns, r, done = transitions[i]
        self.s = ns
        self.t += 1
        truncated = self.t >= self.max_steps and not done
        return ns, r, done, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self.rgb_frame()
        return self.ansi()

    def ansi(self, policy=None) -> str:
        rows = []
        for r in range(self.nrow):
            row = []
            for c in range(self.ncol):
                s = r * self.ncol + c
                ch = self.grid[r, c]
                if s == self.s:
                    row.append("R")
                elif policy is not None and ch in (".", "S"):
                    row.append(ARROWS[int(policy[s])])
                else:
                    row.append(ch)
            rows.append(" ".join(row))
        return "\n".join(rows)

    def rgb_frame(self, cell: int = 24) -> np.ndarray:
        colors = {".": (235, 235, 235), "#": (90, 70, 50), "x": (40, 40, 40),
                  "S": (120, 180, 255), "G": (90, 200, 120)}
        img = np.zeros((self.nrow * cell, self.ncol * cell, 3), dtype=np.uint8)
        for r in range(self.nrow):
            for c in range(self.ncol):
                img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = colors[self.grid[r, c]]
        r, c = self.to_rc(self.s)
        img[r * cell + 6:(r + 1) * cell - 6, c * cell + 6:(c + 1) * cell - 6] = (220, 60, 60)
        return img
