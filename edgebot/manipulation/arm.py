"""Planar 3-DOF (RRR) manipulator: DH kinematics, Jacobian, inverse kinematics,
trajectory generation and joint-space PD control.

The arm works in the vertical x-z plane of the workcell. Joint 3 is the wrist, so
the end-effector pose is (x, z, phi) with phi = q1 + q2 + q3; grasping from above
uses phi = -pi/2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard Denavit-Hartenberg homogeneous transform A_i."""
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class PlanarArm:
    links: tuple = (0.35, 0.30, 0.12)
    base: tuple = (0.0, 0.06)                     # base joint position in the workcell (x, z)
    q_min: np.ndarray = field(default_factory=lambda: np.radians([-10.0, -160.0, -200.0]))
    q_max: np.ndarray = field(default_factory=lambda: np.radians([190.0, 160.0, 200.0]))

    # ------------------------------------------------------------ DH / FK
    def dh_table(self, q):
        """Rows of (a_i, alpha_i, d_i, theta_i). The planar plane x-y of DH maps to workcell x-z."""
        return [(L, 0.0, 0.0, qi) for L, qi in zip(self.links, q)]

    def fk_frames(self, q) -> list:
        T = np.eye(4)
        T[0, 3], T[1, 3] = self.base
        frames = [T.copy()]
        for row in self.dh_table(q):
            T = T @ dh_transform(*row)
            frames.append(T.copy())
        return frames

    def joint_positions(self, q) -> np.ndarray:
        return np.array([F[:2, 3] for F in self.fk_frames(q)])

    def fk(self, q) -> np.ndarray:
        """End-effector pose (x, z, phi)."""
        T = self.fk_frames(q)[-1]
        return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])

    # ------------------------------------------------------------ Jacobian
    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian d(x, z, phi)/dq (3x3)."""
        L1, L2, L3 = self.links
        q1, q12, q123 = q[0], q[0] + q[1], q[0] + q[1] + q[2]
        s1, s12, s123 = np.sin(q1), np.sin(q12), np.sin(q123)
        c1, c12, c123 = np.cos(q1), np.cos(q12), np.cos(q123)
        return np.array([
            [-L1 * s1 - L2 * s12 - L3 * s123, -L2 * s12 - L3 * s123, -L3 * s123],
            [L1 * c1 + L2 * c12 + L3 * c123, L2 * c12 + L3 * c123, L3 * c123],
            [1.0, 1.0, 1.0],
        ])

    def manipulability(self, q) -> float:
        """Yoshikawa manipulability of the position sub-Jacobian."""
        J = self.jacobian(q)[:2]
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def within_limits(self, q) -> bool:
        return bool(np.all(q >= self.q_min - 1e-9) and np.all(q <= self.q_max + 1e-9))

    # ------------------------------------------------------------ IK
    def ik_analytic(self, x: float, z: float, phi: float, elbow: str = "up"):
        """Closed-form IK. Returns joint vector or None if unreachable / outside limits."""
        L1, L2, L3 = self.links
        wx = x - self.base[0] - L3 * np.cos(phi)
        wz = z - self.base[1] - L3 * np.sin(phi)
        c2 = (wx ** 2 + wz ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        if abs(c2) > 1.0 + 1e-9:
            return None
        c2 = np.clip(c2, -1.0, 1.0)
        candidates = []
        for sign in ((1, -1) if elbow == "up" else (-1, 1)):
            q2 = -sign * np.arccos(c2)  # elbow-up = negative q2 when reaching forward
            q1 = np.arctan2(wz, wx) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
            q3 = phi - q1 - q2
            q = np.array([q1, q2, q3])
            q[0] = q1 if self.q_min[0] <= q1 <= self.q_max[0] else wrap(q1)
            q[2] = wrap(q3)
            if self.within_limits(q):
                candidates.append(q)
        return candidates[0] if candidates else None

    def ik_dls(self, target, q0=None, damping: float = 0.05, tol: float = 1e-5, max_iter: int = 300,
               weights=(1.0, 1.0, 0.3)):
        """Damped least squares (Levenberg-Marquardt) IK.

        Returns (q, iterations, final error norm)."""
        q = np.array(q0 if q0 is not None else [np.pi / 2, -np.pi / 2, 0.0], dtype=float)
        Wt = np.diag(weights)
        target = np.asarray(target, dtype=float)
        err = np.inf
        for it in range(1, max_iter + 1):
            pose = self.fk(q)
            e = target - pose
            e[2] = wrap(e[2])
            err = float(np.linalg.norm(Wt @ e))
            if err < tol:
                break
            J = Wt @ self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), Wt @ e)
            q = np.clip(q + dq, self.q_min, self.q_max)
        return q, it, err


# ---------------------------------------------------------------- trajectories
def quintic(q0, q1, T: float, dt: float):
    """Minimum-jerk quintic profile with zero boundary velocity/acceleration.

    Returns (t, q, qd, qdd) sampled every dt."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    n = max(int(np.ceil(T / dt)), 1)
    t = np.linspace(0, T, n + 1)
    s = t / T
    pos = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    vel = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
    acc = (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2
    d = q1 - q0
    return t, q0 + np.outer(pos, d), np.outer(vel, d), np.outer(acc, d)


def cartesian_line(arm: PlanarArm, p0, p1, T: float, dt: float, q_seed):
    """Straight-line end-effector motion with min-jerk timing, solved point-wise by IK."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(int(np.ceil(T / dt)), 1)
    s = np.linspace(0, 1, n + 1)
    s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    qs, q = [], np.asarray(q_seed, float)
    for si in s:
        p = p0 + si * (p1 - p0)
        q_new = arm.ik_analytic(*p)
        if q_new is None or np.linalg.norm(wrap(q_new - q)) > 0.5:
            q_new, _, err = arm.ik_dls(p, q)
            if err > 1e-3:
                raise ValueError(f"Cartesian waypoint {p} unreachable")
        q = q + wrap(q_new - q)
        qs.append(q.copy())
    qs = np.array(qs)
    qd = np.gradient(qs, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    return qs, qd, qdd


# ---------------------------------------------------------------- control
@dataclass
class JointController:
    """Independent-joint PD + acceleration feed-forward on a motor/link model
    I*qdd + b*qd = tau - g(q), with torque saturation."""

    inertia: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.04, 0.01]))
    damping: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.06, 0.02]))
    kp: np.ndarray = field(default_factory=lambda: np.array([90.0, 60.0, 20.0]))
    kd: np.ndarray = field(default_factory=lambda: np.array([6.0, 3.5, 0.9]))
    tau_max: np.ndarray = field(default_factory=lambda: np.array([12.0, 8.0, 3.0]))
    gravity_scale: float = 1.0
    links: tuple = (0.35, 0.30, 0.12)

    def gravity(self, q, payload: float = 0.0) -> np.ndarray:
        # lumped link masses (kg) at link centres; payload at the gripper tip
        m = np.array([0.6, 0.4, 0.15])
        L = np.array(self.links)
        g = 9.81 * self.gravity_scale
        q1, q12, q123 = q[0], q[0] + q[1], q[0] + q[1] + q[2]
        c = np.cos([q1, q12, q123])
        t3 = g * (m[2] * L[2] / 2 + payload * L[2]) * c[2]
        t2 = g * (m[1] * L[1] / 2 + (m[2] + payload) * L[1]) * c[1] + t3
        t1 = g * (m[0] * L[0] / 2 + (m[1] + m[2] + payload) * L[0]) * c[0] + t2
        return np.array([t1, t2, t3])

    def track(self, q_ref, qd_ref, qdd_ref, q0, dt: float, payload: float = 0.0, gravity_comp: bool = True):
        """Simulates tracking of a reference trajectory. Returns (q, qd, tau) histories."""
        q, qd = np.asarray(q0, float).copy(), np.zeros(3)
        Q, QD, TAU = [], [], []
        for qr, vr, ar in zip(q_ref, qd_ref, qdd_ref):
            g = self.gravity(q, payload)
            tau = self.inertia * ar + self.kp * wrap(qr - q) + self.kd * (vr - qd)
            if gravity_comp:
                tau = tau + self.gravity(q, payload)
            tau = np.clip(tau, -self.tau_max, self.tau_max)
            qdd = (tau - self.damping * qd - g) / self.inertia
            qd = qd + qdd * dt
            q = q + qd * dt
            Q.append(q.copy())
            QD.append(qd.copy())
            TAU.append(tau)
        return np.array(Q), np.array(QD), np.array(TAU)
