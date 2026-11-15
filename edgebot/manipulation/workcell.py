"""Pick-and-place and object-assembly workcell for the planar arm.

The workcell is a side view (x horizontal, z vertical) of a table with coloured
blocks. Tasks are compiled into a sequence of motion primitives
(home -> pre-grasp -> descend -> grip -> lift -> transfer -> descend -> release ->
retreat). Each primitive is a min-jerk trajectory (joint space for free motion,
Cartesian straight lines for approach/retreat) tracked by the PD controller.
A block is grasped only if the gripper tip is within tolerance of its grasp point,
and released blocks fall onto the highest supporting surface below them, so
success is decided by simulation and not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edgebot.manipulation.arm import JointController, PlanarArm, cartesian_line, quintic, wrap

DOWN = -np.pi / 2
COLORS = {
    "red": (215, 60, 55), "green": (60, 170, 80), "blue": (55, 105, 210),
    "yellow": (230, 190, 40), "orange": (240, 130, 40), "purple": (140, 80, 180),
}


@dataclass
class Block:
    name: str
    color: str
    x: float
    z: float           # bottom of the block
    w: float = 0.045
    h: float = 0.045
    held: bool = False

    @property
    def top(self) -> float:
        return self.z + self.h

    def grasp_point(self) -> np.ndarray:
        return np.array([self.x, self.z + self.h * 0.6])


@dataclass
class Workcell:
    arm: PlanarArm = field(default_factory=PlanarArm)
    ctrl: JointController = field(default_factory=JointController)
    dt: float = 0.01
    table: tuple = (0.12, 0.62)          # table x-range, surface at z = 0
    grasp_tol: float = 0.012
    blocks: dict = field(default_factory=dict)
    q_home: np.ndarray = field(default_factory=lambda: np.radians([110.0, -100.0, -100.0]))

    def __post_init__(self):
        self.q = self.q_home.copy()
        self.gripper = 0.08                # opening width (m)
        self.holding: str | None = None
        self.log = {"t": [], "q": [], "q_ref": [], "tau": [], "ee": [], "gripper": []}
        self.events: list = []
        self.t = 0.0
        self.on_step = None

    # ------------------------------------------------------------ scene
    def add_block(self, name: str, color: str, x: float, on: str | None = None) -> Block:
        z = self.blocks[on].top if on else 0.0
        b = Block(name, color, x, z)
        self.blocks[name] = b
        return b

    def support_height(self, x: float, w: float, exclude: str) -> float:
        z = 0.0
        for b in self.blocks.values():
            if b.name == exclude or b.held:
                continue
            if abs(b.x - x) < (b.w + w) / 2 - 1e-4:
                z = max(z, b.top)
        return z

    def ee(self) -> np.ndarray:
        return self.arm.fk(self.q)

    # ------------------------------------------------------------ execution
    def _run(self, q_ref, qd_ref, qdd_ref, label: str):
        payload = 0.08 if self.holding else 0.0
        Q, QD, TAU = self.ctrl.track(q_ref, qd_ref, qdd_ref, self.q, self.dt, payload)
        for qi, qr, ti in zip(Q, q_ref, TAU):
            self.t += self.dt
            self.q = qi
            pose = self.arm.fk(qi)
            if self.holding:
                b = self.blocks[self.holding]
                b.x = pose[0]
                b.z = pose[1] - b.h * 0.6
            self.log["t"].append(self.t)
            self.log["q"].append(qi)
            self.log["q_ref"].append(qr)
            self.log["tau"].append(ti)
            self.log["ee"].append(pose)
            self.log["gripper"].append(self.gripper)
            if self.on_step is not None:
                self.on_step(label)
        if not self.events or self.events[-1][1] != label:
            self.events.append((self.t, label))

    def move_joint(self, q_goal, T: float = 1.5, label="move"):
        _, qr, qdr, qddr = quintic(self.q, self.q + wrap(np.asarray(q_goal) - self.q), T, self.dt)
        self._run(qr[1:], qdr[1:], qddr[1:], label)

    def move_cartesian(self, p_goal, T: float = 1.0, label="line"):
        p0 = self.ee()
        p0[2] = p_goal[2]
        qr, qdr, qddr = cartesian_line(self.arm, p0, p_goal, T, self.dt, self.q)
        self._run(qr[1:], qdr[1:], qddr[1:], label)

    def settle(self, T: float = 0.3, label="settle"):
        n = int(T / self.dt)
        q = np.repeat(self.q_cmd()[None], n, 0)
        self._run(q, np.zeros_like(q), np.zeros_like(q), label)

    def q_cmd(self):
        return self.log["q_ref"][-1] if self.log["q_ref"] else self.q

    def set_gripper(self, width: float, T: float = 0.3, label="gripper"):
        n = int(T / self.dt)
        start = self.gripper
        for k in range(1, n + 1):
            self.gripper = start + (width - start) * k / n
            self._run(np.array([self.q_cmd()]), np.zeros((1, 3)), np.zeros((1, 3)), label)

    # ------------------------------------------------------------ primitives
    def pick(self, name: str) -> bool:
        b = self.blocks[name]
        gp = b.grasp_point()
        clearance = 0.10
        pre = np.array([gp[0], b.top + clearance, DOWN])
        q_pre = self.arm.ik_analytic(*pre)
        if q_pre is None:
            self.events.append((self.t, f"unreachable:{name}"))
            return False
        self.set_gripper(0.08, label="open")
        self.move_joint(q_pre, 1.6, label=f"pre-grasp {name}")
        self.move_cartesian(np.array([gp[0], gp[1], DOWN]), 0.9, label="descend")
        self.settle(0.25)
        err = np.linalg.norm(self.ee()[:2] - b.grasp_point())
        self.set_gripper(b.w - 0.004, label="close")
        if err > self.grasp_tol:
            self.events.append((self.t, f"grasp-miss:{name}:{err * 1000:.1f}mm"))
            self.move_cartesian(pre, 0.8, label="retreat")
            return False
        b.held = True
        self.holding = name
        self.move_cartesian(pre, 0.9, label="lift")
        return True

    def place(self, x: float, on: str | None = None) -> bool:
        if self.holding is None:
            return False
        b = self.blocks[self.holding]
        if on is not None:
            x = self.blocks[on].x
        z_support = self.support_height(x, b.w, b.name)
        target_bottom = z_support + 0.002
        gp_z = target_bottom + b.h * 0.6
        pre = np.array([x, gp_z + 0.10, DOWN])
        q_pre = self.arm.ik_analytic(*pre)
        if q_pre is None:
            self.events.append((self.t, "place-unreachable"))
            return False
        self.move_joint(q_pre, 1.8, label="transfer")
        self.move_cartesian(np.array([x, gp_z, DOWN]), 0.9, label="descend")
        self.settle(0.25)
        self.set_gripper(0.08, label="release")
        b.held = False
        self.holding = None
        b.z = self.support_height(b.x, b.w, b.name)  # gravity: drop onto support
        self.move_cartesian(pre, 0.8, label="retreat")
        return True

    def home(self):
        self.move_joint(self.q_home, 1.5, label="home")

    # ------------------------------------------------------------ tasks
    def pick_and_place(self, name: str, x: float | None = None, on: str | None = None) -> bool:
        ok = self.pick(name) and self.place(x if x is not None else 0.0, on)
        self.home()
        return ok

    def stack(self, order: list[str], x: float) -> bool:
        """Object assembly: build a tower with ``order[0]`` at the bottom."""
        ok = self.pick(order[0]) and self.place(x)
        for below, name in zip(order, order[1:]):
            ok = ok and self.pick(name) and self.place(x, on=below)
        self.home()
        return ok

    # ------------------------------------------------------------ metrics
    def tracking_rmse(self) -> np.ndarray:
        q = np.array(self.log["q"])
        qr = np.array(self.log["q_ref"])
        return np.sqrt(np.mean(wrap(qr - q) ** 2, axis=0))

    def tower_ok(self, order: list[str], x: float, tol: float = 0.01) -> bool:
        z = 0.0
        for name in order:
            b = self.blocks[name]
            if abs(b.x - x) > tol or abs(b.z - z) > 0.005:
                return False
            z = b.top
        return True

    # ------------------------------------------------------------ rendering
    def render(self, q=None, gripper=None, size=(520, 400), title: str = "") -> np.ndarray:
        from PIL import Image, ImageDraw

        q = self.q if q is None else q
        gripper = self.gripper if gripper is None else gripper
        W, H = size
        sx = W / 0.92
        ox, oz = 0.22 * sx, H - 36

        def P(x, z):
            return ox + x * sx, oz - z * sx

        img = Image.new("RGB", (W, H), (248, 248, 252))
        d = ImageDraw.Draw(img)
        x0, y0 = P(self.table[0], 0)
        x1, _ = P(self.table[1], 0)
        d.rectangle([x0, y0, x1, y0 + 12], fill=(160, 130, 100))
        bx, bz = P(*self.arm.base)
        d.rectangle([bx - 18, bz, bx + 18, oz + 12], fill=(80, 80, 90))
        for b in self.blocks.values():
            a = P(b.x - b.w / 2, b.top)
            c = P(b.x + b.w / 2, b.z)
            d.rectangle([a[0], a[1], c[0], c[1]], fill=COLORS.get(b.color, (120, 120, 120)), outline=(40, 40, 40))
        pts = self.arm.joint_positions(q)
        for i in range(len(pts) - 1):
            d.line([P(*pts[i]), P(*pts[i + 1])], fill=(60, 60, 70), width=10 - 2 * i)
        for p in pts[:-1]:
            cx, cz = P(*p)
            d.ellipse([cx - 6, cz - 6, cx + 6, cz + 6], fill=(230, 120, 40))
        # gripper jaws (perpendicular to the last link)
        tip = pts[-1]
        phi = q.sum()
        n = np.array([-np.sin(phi), np.cos(phi)])
        t = np.array([np.cos(phi), np.sin(phi)])
        for s in (1, -1):
            base_pt = tip - 0.02 * t + s * n * gripper / 2
            d.line([P(*base_pt), P(*(base_pt + 0.035 * t))], fill=(30, 30, 30), width=4)
        d.line([P(*(tip - 0.02 * t + n * gripper / 2)), P(*(tip - 0.02 * t - n * gripper / 2))],
               fill=(30, 30, 30), width=4)
        if title:
            d.text((10, 8), title, fill=(20, 20, 20))
        return np.asarray(img)


class Recorder:
    """Captures a rendered frame every ``every`` control steps while a Workcell executes."""

    def __init__(self, cell: Workcell, every: int = 4, size=(520, 400)):
        self.cell, self.every, self.size = cell, every, size
        self.frames: list = []
        self._k = 0
        cell.on_step = self._on_step

    def _on_step(self, label: str):
        if self._k % self.every == 0:
            self.frames.append(self.cell.render(size=self.size, title=label))
        self._k += 1
