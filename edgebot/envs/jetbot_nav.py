"""JetBot navigation environment (continuous state space).

A differential-drive robot (NVIDIA JetBot footprint, radius 0.12 m) must reach a
goal pose in a cluttered warehouse while avoiding shelves and boxes. The
environment is written in pure NumPy so thousands of steps per second run on a
laptop CPU, and it follows the Gymnasium API so any standard agent can train on it.

Observation modes
-----------------
``lidar``  : 24-beam 360 deg range scan + goal distance/bearing + previous action (29-D)
``pixels`` : 2x64x64 egocentric occupancy / goal image (vision-based RL)

Action modes
------------
``continuous`` : Box([-1, 1]^2) -> linear velocity v in [0, v_max], angular velocity w
``discrete``   : 5 motion primitives (forward, forward-left, forward-right, spin left/right)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DISCRETE_ACTIONS = np.array([
    [1.0, 0.0],    # forward
    [0.6, 0.6],    # forward + left
    [0.6, -0.6],   # forward + right
    [0.0, 1.0],    # spin left
    [0.0, -1.0],   # spin right
], dtype=np.float32)


@dataclass
class Scenario:
    width: float
    height: float
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))    # [xmin, ymin, xmax, ymax]
    circles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # [cx, cy, r]
    start: np.ndarray = field(default_factory=lambda: np.zeros(3))         # x, y, theta
    goal: np.ndarray = field(default_factory=lambda: np.zeros(2))


class JetBotNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, obs_mode: str = "lidar", action_mode: str = "continuous", n_rays: int = 24,
                 max_range: float = 3.0, max_steps: int = 300, dt: float = 0.1, v_max: float = 0.5,
                 w_max: float = 2.0, world: tuple = (6.0, 5.0), goal_tolerance: float = 0.2,
                 render_mode: str | None = None, fixed_scenario: Scenario | None = None,
                 difficulty: float = 1.0, lidar_noise: float = 0.0, lidar_mask: float = 0.0):
        assert obs_mode in ("lidar", "pixels")
        assert action_mode in ("continuous", "discrete")
        self.obs_mode, self.action_mode = obs_mode, action_mode
        self.n_rays, self.max_range = n_rays, max_range
        self.max_steps, self.dt = max_steps, dt
        self.v_max, self.w_max = v_max, w_max
        self.W, self.H = world
        self.robot_r = 0.12
        self.goal_tol = goal_tolerance
        self.render_mode = render_mode
        self.fixed_scenario = fixed_scenario
        self.difficulty = difficulty
        # sensor corruption (off by default), following the LiDAR randomisation of Omni-Perception (2025):
        # lidar_mask = fraction of beams replaced by a spurious short return in [0, 0.3] m,
        # lidar_noise = relative range noise (uniform, +/- lidar_noise of the true distance)
        self.lidar_noise, self.lidar_mask = lidar_noise, lidar_mask
        self._scan, self._scan_key = None, None
        self.ray_angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)

        if action_mode == "continuous":
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        if obs_mode == "lidar":
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(n_rays + 5,), dtype=np.float32)
        else:
            self.px, self.res = 64, 0.05  # 64 px * 5 cm = 3.2 m egocentric window
            self.observation_space = spaces.Box(0.0, 1.0, shape=(2, self.px, self.px), dtype=np.float32)

        self.scn: Scenario | None = None
        self.pose = np.zeros(3)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.t = 0
        self.trail: list = []

    # ============================================================ scenarios
    def _random_scenario(self) -> Scenario:
        rng = self.np_random
        W, H = self.W, self.H
        n_boxes = int(rng.integers(0, int(4 * self.difficulty) + 1))
        n_circ = int(rng.integers(1, int(6 * self.difficulty) + 2))
        boxes = []
        for _ in range(n_boxes):
            horizontal = rng.random() < 0.5
            w, h = (rng.uniform(1.0, 2.0), rng.uniform(0.3, 0.5))
            if not horizontal:
                w, h = h, w
            x0, y0 = rng.uniform(0.3, W - w - 0.3), rng.uniform(0.3, H - h - 0.3)
            boxes.append([x0, y0, x0 + w, y0 + h])
        circles = [[rng.uniform(0.4, W - 0.4), rng.uniform(0.4, H - 0.4), rng.uniform(0.12, 0.3)]
                   for _ in range(n_circ)]
        scn = Scenario(W, H, np.array(boxes).reshape(-1, 4), np.array(circles).reshape(-1, 3))
        for _ in range(500):
            s = np.array([rng.uniform(0.3, W - 0.3), rng.uniform(0.3, H - 0.3)])
            g = np.array([rng.uniform(0.3, W - 0.3), rng.uniform(0.3, H - 0.3)])
            if (self._clearance(s, scn) > self.robot_r + 0.15 and self._clearance(g, scn) > self.robot_r + 0.1
                    and 1.0 < np.linalg.norm(s - g) < 2.5 + 3.0 * self.difficulty):
                scn.start = np.array([s[0], s[1], rng.uniform(-np.pi, np.pi)])
                scn.goal = g
                return scn
        return self._random_scenario()

    # ============================================================ geometry
    @staticmethod
    def _clearance(p: np.ndarray, scn: Scenario) -> float:
        """Signed distance from point p to the nearest obstacle or wall."""
        d = min(p[0], p[1], scn.width - p[0], scn.height - p[1])
        if len(scn.boxes):
            b = scn.boxes
            dx = np.maximum(np.maximum(b[:, 0] - p[0], 0), p[0] - b[:, 2])
            dy = np.maximum(np.maximum(b[:, 1] - p[1], 0), p[1] - b[:, 3])
            inside = (dx == 0) & (dy == 0)
            dist = np.hypot(dx, dy)
            dist[inside] = 0.0
            d = min(d, dist.min())
        if len(scn.circles):
            c = scn.circles
            d = min(d, (np.hypot(c[:, 0] - p[0], c[:, 1] - p[1]) - c[:, 2]).min())
        return float(d)

    def lidar(self, pose: np.ndarray | None = None) -> np.ndarray:
        x, y, th = self.pose if pose is None else pose
        scn = self.scn
        ang = th + self.ray_angles
        dx, dy = np.cos(ang), np.sin(ang)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 0, (scn.width - x) / dx, np.where(dx < 0, -x / dx, np.inf))
            ty = np.where(dy > 0, (scn.height - y) / dy, np.where(dy < 0, -y / dy, np.inf))
            t = np.minimum(tx, ty)
            if len(scn.boxes):
                b = scn.boxes[None]  # 1 x B x 4
                ddx, ddy = dx[:, None], dy[:, None]
                t1x, t2x = (b[..., 0] - x) / ddx, (b[..., 2] - x) / ddx
                t1y, t2y = (b[..., 1] - y) / ddy, (b[..., 3] - y) / ddy
                tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
                tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
                hit = (tmax >= np.maximum(tmin, 0)) & np.isfinite(tmin)
                tb = np.where(hit, np.maximum(tmin, 0), np.inf).min(axis=1)
                t = np.minimum(t, tb)
            if len(scn.circles):
                c = scn.circles[None]
                ox, oy = x - c[..., 0], y - c[..., 1]
                bq = dx[:, None] * ox + dy[:, None] * oy
                cq = ox ** 2 + oy ** 2 - c[..., 2] ** 2
                disc = bq ** 2 - cq
                tc = -bq - np.sqrt(np.maximum(disc, 0))
                tc = np.where((disc >= 0) & (tc >= 0), tc, np.inf)
                tc = np.where(cq < 0, 0.0, tc)
                t = np.minimum(t, tc.min(axis=1))
        return np.clip(t, 0.0, self.max_range)

    # ============================================================ API
    def set_goal(self, goal_xy) -> None:
        self.scn.goal = np.asarray(goal_xy, dtype=float)
        self._prev_dist = float(np.linalg.norm(self.pose[:2] - self.scn.goal))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "scenario" in options:
            self.scn = options["scenario"]
        elif self.fixed_scenario is not None:
            self.scn = self.fixed_scenario
        else:
            self.scn = self._random_scenario()
        self.pose = np.array(options.get("start", self.scn.start), dtype=float)
        if "goal" in options:
            self.scn.goal = np.asarray(options["goal"], dtype=float)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.t = 0
        self._scan = None
        self.trail = [self.pose[:2].copy()]
        self._prev_dist = float(np.linalg.norm(self.pose[:2] - self.scn.goal))
        if self.obs_mode == "pixels":
            self._raster_world()
        return self._obs(), {}

    def _to_cmd(self, action) -> np.ndarray:
        if self.action_mode == "discrete":
            a = DISCRETE_ACTIONS[int(action)]
        else:
            a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
            a = np.array([(a[0] + 1) / 2, a[1]], dtype=np.float32)
        return a  # (v fraction in [0,1], w fraction in [-1,1])

    def step(self, action):
        a = self._to_cmd(action)
        v, w = a[0] * self.v_max, a[1] * self.w_max
        x, y, th = self.pose
        # exact unicycle integration over dt (midpoint heading)
        thm = th + 0.5 * w * self.dt
        self.pose = np.array([x + v * np.cos(thm) * self.dt, y + v * np.sin(thm) * self.dt,
                              (th + w * self.dt + np.pi) % (2 * np.pi) - np.pi])
        self.t += 1
        self._scan = None
        self.trail.append(self.pose[:2].copy())

        dist = float(np.linalg.norm(self.pose[:2] - self.scn.goal))
        clearance = self._clearance(self.pose[:2], self.scn)
        collided = clearance < self.robot_r
        reached = dist < self.goal_tol

        reward = 3.0 * (self._prev_dist - dist) - 0.01
        reward -= 0.05 * max(0.0, 0.35 - (clearance - self.robot_r)) / 0.35
        reward -= 0.01 * abs(a[1] - self.prev_action[1])  # smoothness
        terminated = False
        if collided:
            reward -= 10.0
            terminated = True
        elif reached:
            reward += 10.0
            terminated = True
        self._prev_dist = dist
        self.prev_action = a
        truncated = self.t >= self.max_steps and not terminated
        info = {"collision": collided, "success": reached, "distance": dist, "clearance": clearance}
        return self._obs(), float(reward), terminated, truncated, info

    # ============================================================ observations
    def _goal_features(self):
        g = self.scn.goal - self.pose[:2]
        dist = np.linalg.norm(g)
        bearing = np.arctan2(g[1], g[0]) - self.pose[2]
        return dist, bearing

    def sensed_lidar(self) -> np.ndarray:
        """The scan the robot actually receives this step (true scan unless corruption is enabled)."""
        key = (self.t, *self.pose)
        if self._scan is None or self._scan_key != key:
            self._scan_key = key
            scan = self.lidar()
            if self.lidar_noise > 0:
                scan = scan * (1 + self.np_random.uniform(-self.lidar_noise, self.lidar_noise, scan.shape))
            if self.lidar_mask > 0:
                m = self.np_random.random(scan.shape) < self.lidar_mask
                scan = np.where(m, self.np_random.uniform(0.0, 0.3, scan.shape), scan)
            self._scan = np.clip(scan, 0.0, self.max_range)
        return self._scan

    def _obs(self):
        if self.obs_mode == "lidar":
            scan = self.sensed_lidar() / self.max_range
            dist, bearing = self._goal_features()
            o = np.concatenate([scan * 2 - 1, [min(dist / 5.0, 1.0) * 2 - 1, np.sin(bearing), np.cos(bearing)],
                                self.prev_action * np.array([2, 1]) - np.array([1, 0])])
            return o.astype(np.float32)
        return self._pixel_obs()

    def _raster_world(self):
        """Pre-rasterises the static map once per episode (5 cm cells)."""
        r = self.res
        nx, ny = int(np.ceil(self.W / r)), int(np.ceil(self.H / r))
        xs = (np.arange(nx) + 0.5) * r
        ys = (np.arange(ny) + 0.5) * r
        X, Y = np.meshgrid(xs, ys)  # ny x nx
        occ = np.zeros_like(X, dtype=bool)
        for b in self.scn.boxes:
            occ |= (X >= b[0]) & (X <= b[2]) & (Y >= b[1]) & (Y <= b[3])
        for c in self.scn.circles:
            occ |= (X - c[0]) ** 2 + (Y - c[1]) ** 2 <= c[2] ** 2
        self._occ = np.pad(occ, 40, constant_values=True).astype(np.float32)  # walls outside the map
        half = (np.arange(self.px) - self.px / 2 + 0.5) * r
        U, V = np.meshgrid(half, half[::-1])  # u forward-right frame, v left
        self._grid_uv = (U.ravel(), V.ravel())

    def _pixel_obs(self):
        x, y, th = self.pose
        u, v = self._grid_uv
        # image rows = forward axis (up), cols = left/right
        fwd, left = v, -u
        wx = x + fwd * np.cos(th) - left * np.sin(th)
        wy = y + fwd * np.sin(th) + left * np.cos(th)
        ix = np.clip((wx / self.res).astype(int) + 40, 0, self._occ.shape[1] - 1)
        iy = np.clip((wy / self.res).astype(int) + 40, 0, self._occ.shape[0] - 1)
        occ = self._occ[iy, ix].reshape(self.px, self.px)
        dist, bearing = self._goal_features()
        d = min(dist, 1.5)
        gf, gl = d * np.cos(bearing), d * np.sin(bearing)
        gimg = np.exp(-(((fwd - gf) ** 2 + (left - gl) ** 2) / (2 * 0.12 ** 2))).reshape(self.px, self.px)
        return np.stack([occ, gimg]).astype(np.float32)

    # ============================================================ rendering
    def render(self, scale: int = 100, show_lidar: bool = True):
        from PIL import Image, ImageDraw

        W, H = int(self.W * scale), int(self.H * scale)
        img = Image.new("RGB", (W, H), (245, 245, 240))
        d = ImageDraw.Draw(img)

        def P(x, y):
            return x * scale, H - y * scale

        for b in self.scn.boxes:
            x0, y1 = P(b[0], b[1])
            x1, y0 = P(b[2], b[3])
            d.rectangle([x0, y0, x1, y1], fill=(140, 105, 70), outline=(90, 65, 40))
        for c in self.scn.circles:
            cx, cy = P(c[0], c[1])
            rr = c[2] * scale
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(200, 150, 60), outline=(120, 90, 30))
        gx, gy = P(*self.scn.goal)
        gr = self.goal_tol * scale
        d.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], outline=(40, 160, 70), width=3)
        if len(self.trail) > 1:
            d.line([P(*p) for p in self.trail], fill=(70, 120, 220), width=2)
        x, y, th = self.pose
        if show_lidar:
            scan = self.lidar()
            for ang, rng_ in zip(self.ray_angles, scan):
                ex, ey = x + rng_ * np.cos(th + ang), y + rng_ * np.sin(th + ang)
                d.line([P(x, y), P(ex, ey)], fill=(255, 120, 120), width=1)
        rx, ry = P(x, y)
        rr = self.robot_r * scale
        d.ellipse([rx - rr, ry - rr, rx + rr, ry + rr], fill=(30, 30, 30))
        d.line([P(x, y), P(x + 0.18 * np.cos(th), y + 0.18 * np.sin(th))], fill=(80, 220, 80), width=3)
        return np.asarray(img)


def make_vec_envs(n: int, seed: int = 0, **kwargs):
    """A light synchronous vector of environments with auto-reset."""
    return SyncVecEnv([lambda: JetBotNavEnv(**kwargs) for _ in range(n)], seed=seed)


class SyncVecEnv:
    def __init__(self, fns, seed: int = 0):
        self.envs = [f() for f in fns]
        self.n = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.seed = seed
        self._ep = 0

    def reset(self):
        obs = []
        for i, e in enumerate(self.envs):
            o, _ = e.reset(seed=self.seed + i)
            obs.append(o)
        return np.stack(obs)

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, term, trunc, info = e.step(a)
            info["terminated"] = term
            info["truncated"] = trunc
            if term or trunc:
                info["final_obs"] = o
                o, _ = e.reset()
            obs.append(o)
            rews.append(r)
            dones.append(term or trunc)
            infos.append(info)
        return np.stack(obs), np.array(rews, dtype=np.float32), np.array(dones), infos
