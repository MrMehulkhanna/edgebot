import numpy as np

from edgebot.envs.warehouse_grid import WarehouseGridEnv
from edgebot.rl import tabular as T


def test_transition_model_is_normalised():
    env = WarehouseGridEnv(slip=0.2)
    for s in range(env.n_states):
        for a in range(4):
            assert abs(sum(p for p, *_ in env.P[s][a]) - 1.0) < 1e-12


def test_value_and_policy_iteration_agree():
    env = WarehouseGridEnv(slip=0.1)
    V_vi, pi_vi, _, _ = T.value_iteration(env.P, env.n_states, 4, 0.99)
    V_pi, _, _ = T.policy_iteration(env.P, env.n_states, 4, 0.99)
    assert np.max(np.abs(V_vi - V_pi)) < 1e-5


def test_optimal_deterministic_path_length():
    env = WarehouseGridEnv()
    V, pi, _, _ = T.value_iteration(env.P, env.n_states, 4, 1.0)
    assert V[env.start] == -11.0
    G, path = T.rollout_greedy(env, pi)
    assert G == -11.0 and path[-1] == env.goal


def test_q_learning_finds_optimal_route_and_sarsa_is_safe():
    q = T.q_learning(WarehouseGridEnv(), 500, seed=0)
    assert T.rollout_greedy(WarehouseGridEnv(), q.policy)[0] == -11.0
    s = T.sarsa(WarehouseGridEnv(), 500, seed=1)
    G, path = T.rollout_greedy(WarehouseGridEnv(), s.policy)
    env = WarehouseGridEnv()
    # SARSA's greedy route never walks along the row next to the spill (row 3)
    assert G < -11.0 and all(env.to_rc(x)[0] != 3 or env.to_rc(x)[1] in (0, 9) for x in path)
