import numpy as np

from edgebot.envs.jetbot_nav import JetBotNavEnv, Scenario
from edgebot.robot import warehouse as wh
from edgebot.robot.planner import plan_path


def test_observation_bounds_and_shapes():
    env = JetBotNavEnv()
    obs, _ = env.reset(seed=3)
    assert obs.shape == (29,) and obs.min() >= -1 and obs.max() <= 1
    pix = JetBotNavEnv(obs_mode="pixels", action_mode="discrete")
    o, _ = pix.reset(seed=3)
    assert o.shape == (2, 64, 64)


def test_lidar_matches_geometry():
    scn = Scenario(4.0, 4.0, np.array([[3.0, 0.0, 3.5, 4.0]]), np.zeros((0, 3)),
                   start=np.array([1.0, 2.0, 0.0]), goal=np.array([0.5, 0.5]))
    env = JetBotNavEnv(fixed_scenario=scn)
    env.reset(seed=0)
    scan = env.lidar()
    forward = scan[np.argmin(np.abs(env.ray_angles))]
    backward = scan[np.argmin(np.abs(np.abs(env.ray_angles) - np.pi))]
    assert abs(forward - 2.0) < 1e-6 and abs(backward - 1.0) < 1e-6


def test_collision_terminates():
    scn = Scenario(4.0, 4.0, np.array([[1.3, 0.0, 1.6, 4.0]]), np.zeros((0, 3)),
                   start=np.array([1.0, 2.0, 0.0]), goal=np.array([3.5, 2.0]))
    env = JetBotNavEnv(fixed_scenario=scn)
    env.reset(seed=0)
    for _ in range(50):
        _, r, term, _, info = env.step(np.array([1.0, 0.0]))
        if term:
            break
    assert info["collision"] and r < -5


def test_every_station_pair_is_connected():
    scn = wh.scenario()
    names = list(wh.STATIONS)
    for a in names:
        for b in names:
            if a != b:
                assert plan_path(scn, np.array(wh.STATIONS[a]), np.array(wh.STATIONS[b]), inflate=0.24) is not None


def test_shield_blocks_driving_into_wall():
    import numpy as np

    from edgebot.envs.jetbot_nav import JetBotNavEnv
    from edgebot.rl.shield import LidarShield

    env = JetBotNavEnv()
    env.reset(seed=0)
    env.pose = np.array([0.2, 2.5, np.pi])  # facing the left wall, 8 cm of clearance
    shield = LidarShield(env)
    safe = shield(np.array([1.0, 0.0], dtype=np.float32))  # full speed ahead
    assert shield.interventions == 1
    assert env._to_cmd(safe)[0] < 1.0


def test_robust_shield_ignores_single_spurious_beam():
    import numpy as np

    from edgebot.envs.jetbot_nav import JetBotNavEnv
    from edgebot.rl.shield import LidarShield

    env = JetBotNavEnv()
    env.reset(seed=0)
    env.pose = np.array([3.0, 2.5, 0.0])
    env._scan_key = (env.t, *env.pose)
    env._scan = np.full(env.n_rays, env.max_range)
    env._scan[np.argmin(np.abs(env.ray_angles))] = 0.1  # one fake return straight ahead
    plain, robust = LidarShield(env), LidarShield(env, median_filter=True)
    plain(np.array([1.0, 0.0], dtype=np.float32))
    robust(np.array([1.0, 0.0], dtype=np.float32))
    assert plain.interventions == 1 and robust.interventions == 0
