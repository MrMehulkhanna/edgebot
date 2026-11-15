import numpy as np

from edgebot.manipulation.arm import PlanarArm, quintic
from edgebot.manipulation.workcell import Workcell


def test_analytic_ik_inverts_fk():
    arm = PlanarArm()
    rng = np.random.default_rng(0)
    for _ in range(300):
        q = rng.uniform(arm.q_min, arm.q_max)
        p = arm.fk(q)
        qi = arm.ik_analytic(*p)
        if qi is not None:
            assert np.allclose(arm.fk(qi)[:2], p[:2], atol=1e-9)


def test_jacobian_matches_finite_differences():
    arm = PlanarArm()
    q = np.array([0.7, -0.9, 0.2])
    eps = 1e-6
    Jn = np.stack([(arm.fk(q + eps * e) - arm.fk(q - eps * e)) / (2 * eps) for e in np.eye(3)], 1)
    assert np.allclose(arm.jacobian(q), Jn, atol=1e-6)


def test_quintic_boundary_conditions():
    _, q, qd, qdd = quintic([0, 0, 0], [1, -1, 0.5], 2.0, 0.01)
    assert np.allclose(q[0], 0) and np.allclose(q[-1], [1, -1, 0.5])
    assert np.allclose(qd[[0, -1]], 0) and np.allclose(qdd[[0, -1]], 0, atol=1e-9)


def test_pick_place_and_stack():
    cell = Workcell()
    for name, x in (("r", 0.22), ("g", 0.32), ("b", 0.54)):
        cell.add_block(name, name, x)
    assert cell.pick_and_place("r", 0.42)
    assert abs(cell.blocks["r"].x - 0.42) < 0.01 and cell.blocks["r"].z == 0.0
    assert cell.stack(["b", "g", "r"], 0.22) and cell.tower_ok(["b", "g", "r"], 0.22)
