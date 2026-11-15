"""Manipulation labs: Pick-n-Place and Object Assembly, plus an IK solver study.

* IK study: analytic vs damped-least-squares IK over random reachable poses
  (accuracy, iterations, runtime), manipulability map of the workspace
* Pick-n-Place: randomised block / target positions, success = block ends at target
* Object assembly: build 3-block towers in a random colour order
* Robustness: success vs. joint-sensor noise injected into the controller

    python scripts/run_manipulation.py --trials 100
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from edgebot.manipulation.arm import JointController, PlanarArm  # noqa: E402
from edgebot.manipulation.workcell import Recorder, Workcell  # noqa: E402
from edgebot.utils import RESULTS, save_json  # noqa: E402

SLOTS = np.array([0.22, 0.30, 0.38, 0.46, 0.54])


def ik_study(n=2000, seed=0):
    arm = PlanarArm()
    rng = np.random.default_rng(seed)
    a_err, a_time, d_err, d_time, d_iter, fails = [], [], [], [], [], 0
    for _ in range(n):
        q = rng.uniform(arm.q_min, arm.q_max)
        target = arm.fk(q)
        t = time.perf_counter()
        qa = arm.ik_analytic(*target)
        a_time.append(time.perf_counter() - t)
        if qa is None:
            fails += 1
        else:
            a_err.append(np.linalg.norm(arm.fk(qa)[:2] - target[:2]))
        t = time.perf_counter()
        qd, it, _ = arm.ik_dls(target)
        d_time.append(time.perf_counter() - t)
        d_iter.append(it)
        d_err.append(np.linalg.norm(arm.fk(qd)[:2] - target[:2]))
    xs, zs = np.linspace(0.0, 0.7, 71), np.linspace(-0.1, 0.6, 71)
    manip = np.full((len(zs), len(xs)), np.nan)
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            q = arm.ik_analytic(x, z, -np.pi / 2)
            if q is not None:
                manip[i, j] = arm.manipulability(q)
    return {
        "samples": n,
        "analytic": {"mean_err_m": float(np.mean(a_err)), "max_err_m": float(np.max(a_err)),
                     "mean_time_us": float(np.mean(a_time) * 1e6), "limit_failures": fails},
        "dls": {"mean_err_m": float(np.mean(d_err)), "p99_err_m": float(np.percentile(d_err, 99)),
                "mean_iterations": float(np.mean(d_iter)), "mean_time_us": float(np.mean(d_time) * 1e6),
                "success_rate_1mm": float(np.mean(np.array(d_err) < 1e-3))},
        "manipulability": {"x": xs.tolist(), "z": zs.tolist(), "grid": np.nan_to_num(manip, nan=-1).tolist()},
    }


class NoisyController(JointController):
    """Adds Gaussian joint-encoder noise to the feedback signal."""

    def __init__(self, sigma_deg: float, seed: int):
        super().__init__()
        self.sigma = np.radians(sigma_deg)
        self.rng = np.random.default_rng(seed)

    def track(self, q_ref, qd_ref, qdd_ref, q0, dt, payload=0.0, gravity_comp=True):
        q, qd = np.asarray(q0, float).copy(), np.zeros(3)
        Q, QD, TAU = [], [], []
        for qr, vr, ar in zip(q_ref, qd_ref, qdd_ref):
            q_meas = q + self.rng.normal(0, self.sigma, 3)
            g = self.gravity(q, payload)
            tau = self.inertia * ar + self.kp * (qr - q_meas) + self.kd * (vr - qd) + self.gravity(q_meas, payload)
            tau = np.clip(tau, -self.tau_max, self.tau_max)
            qdd = (tau - self.damping * qd - g) / self.inertia
            qd = qd + qdd * dt
            q = q + qd * dt
            Q.append(q.copy())
            QD.append(qd.copy())
            TAU.append(tau)
        return np.array(Q), np.array(QD), np.array(TAU)


def pick_place_trials(trials, seed=0, sigma_deg=0.0):
    rng = np.random.default_rng(seed)
    ok, place_err, durations, rmse = [], [], [], []
    for k in range(trials):
        ctrl = NoisyController(sigma_deg, seed + k) if sigma_deg > 0 else JointController()
        cell = Workcell(ctrl=ctrl)
        src, dst = rng.choice(len(SLOTS), 2, replace=False)
        x_src = SLOTS[src] + rng.uniform(-0.02, 0.02)
        x_dst = SLOTS[dst] + rng.uniform(-0.02, 0.02)
        cell.add_block("b", "red", x_src)
        success = cell.pick_and_place("b", x_dst)
        b = cell.blocks["b"]
        err = abs(b.x - x_dst)
        success = success and err < 0.01 and abs(b.z) < 1e-6
        ok.append(success)
        place_err.append(err)
        durations.append(cell.t)
        rmse.append(np.degrees(cell.tracking_rmse()).mean())
    return {"trials": trials, "success_rate": float(np.mean(ok)), "mean_place_err_mm": float(np.mean(place_err) * 1000),
            "mean_duration_s": float(np.mean(durations)), "mean_tracking_rmse_deg": float(np.mean(rmse))}


def assembly_trials(trials, seed=1, sigma_deg=0.0):
    rng = np.random.default_rng(seed)
    colors = ["red", "green", "blue", "yellow"]
    ok, durations = [], []
    for k in range(trials):
        ctrl = NoisyController(sigma_deg, seed + k) if sigma_deg > 0 else JointController()
        cell = Workcell(ctrl=ctrl)
        slots = rng.permutation(SLOTS)
        names = list(rng.permutation(colors)[:3])
        for name, x in zip(names, slots[:3]):
            cell.add_block(name, name, float(x))
        target = float(slots[3])
        order = list(rng.permutation(names))
        success = cell.stack(order, target) and cell.tower_ok(order, target)
        ok.append(success)
        durations.append(cell.t)
    return {"trials": trials, "success_rate": float(np.mean(ok)), "mean_duration_s": float(np.mean(durations))}


def make_demo_gif(path: Path):
    import imageio.v2 as imageio

    cell = Workcell()
    for name, x in (("red", 0.24), ("green", 0.34), ("blue", 0.54)):
        cell.add_block(name, name, x)
    rec = Recorder(cell, every=5)
    cell.pick_and_place("red", 0.44)
    cell.stack(["blue", "green", "red"], 0.30)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, rec.frames, duration=0.05, loop=0)
    log = {k: np.array(v).tolist() for k, v in cell.log.items() if k in ("t", "q", "q_ref", "tau")}
    return len(rec.frames), log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=100)
    args = p.parse_args()
    out = {"ik": ik_study()}
    print("IK", {k: v for k, v in out["ik"].items() if k != "manipulability"})
    out["pick_place"] = pick_place_trials(args.trials)
    print("pick-n-place", out["pick_place"])
    out["assembly"] = assembly_trials(args.trials)
    print("assembly", out["assembly"])
    out["noise_robustness"] = []
    for sigma in (0.0, 0.5, 1.0, 2.0, 4.0):
        r = {"sigma_deg": sigma, "pick_place": pick_place_trials(max(args.trials // 2, 10), 100, sigma)["success_rate"],
             "assembly": assembly_trials(max(args.trials // 4, 10), 200, sigma)["success_rate"]}
        out["noise_robustness"].append(r)
        print("noise", r)
    n, log = make_demo_gif(RESULTS / "media" / "manipulation_demo.gif")
    out["demo_frames"] = n
    save_json(log, RESULTS / "manipulation_trace.json")
    save_json(out, RESULTS / "manipulation.json")


if __name__ == "__main__":
    main()
