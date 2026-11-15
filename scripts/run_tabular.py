"""Finite-MDP experiments on the warehouse grid world.

* Dynamic Programming: value iteration vs policy iteration (deterministic + slippery floor)
* Model-free control: Monte Carlo, SARSA, Expected SARSA, Q-learning, Double Q-learning, n-step SARSA
* The classic on-policy vs off-policy result: SARSA learns the safe route away from the
  oil spill, Q-learning learns the optimal edge route but earns less online reward
* TD(0) prediction converging to v_pi computed by DP

    python scripts/run_tabular.py --runs 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from edgebot.envs.warehouse_grid import WarehouseGridEnv  # noqa: E402
from edgebot.rl import tabular as T  # noqa: E402
from edgebot.utils import RESULTS, ensure_dirs, save_json  # noqa: E402


def run_control(name, fn, runs, episodes, **kw):
    curves, final_greedy, paths = [], [], []
    for seed in range(runs):
        env = WarehouseGridEnv()
        res = fn(env, episodes, seed=seed, **kw)
        curves.append(res.returns)
        G, path = T.rollout_greedy(WarehouseGridEnv(), res.policy)
        final_greedy.append(G)
        paths.append(path)
    curves = np.array(curves, dtype=float)
    return {
        "name": name,
        "mean_curve": curves.mean(0).tolist(),
        "online_return_last100": float(curves[:, -100:].mean()),
        "greedy_return_mean": float(np.mean(final_greedy)),
        "greedy_return_median": float(np.median(final_greedy)),
        "greedy_reaches_goal": float(np.mean(np.array(final_greedy) > -200)),
        "greedy_return_all": [float(g) for g in final_greedy],
        "example_path": [int(s) for s in paths[0]],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--episodes", type=int, default=500)
    args = p.parse_args()
    ensure_dirs()
    out = {"layout": list(WarehouseGridEnv().grid.tolist())}

    # ---------------- Dynamic programming
    dp = {}
    for slip in (0.0, 0.1, 0.2):
        env = WarehouseGridEnv(slip=slip)
        gamma = 1.0 if slip == 0 else 0.99
        V_vi, pi_vi, it_vi, deltas = T.value_iteration(env.P, env.n_states, 4, gamma)
        V_pi, pi_pi, it_pi = T.policy_iteration(env.P, env.n_states, 4, gamma)
        dp[f"slip_{slip}"] = {
            "gamma": gamma, "vi_iterations": it_vi, "pi_improvements": it_pi,
            "V_start": float(V_vi[env.start]), "max_value_gap": float(np.max(np.abs(V_vi - V_pi))),
            "V": V_vi.reshape(env.nrow, env.ncol).tolist(), "policy": pi_vi.tolist(),
            "vi_deltas": deltas, "policy_ascii": env.ansi(pi_vi),
        }
        print(f"DP slip={slip}: VI {it_vi} sweeps, PI {it_pi} improvements, V*(S)={V_vi[env.start]:.2f}")
        print(env.ansi(pi_vi))
    out["dp"] = dp

    # ---------------- Model-free control (deterministic floor)
    E = args.episodes
    algos = [
        ("Monte Carlo (first-visit)", T.mc_control, dict(gamma=1.0, eps=0.1)),
        ("SARSA", T.sarsa, dict(alpha=0.5, eps=0.1)),
        ("Expected SARSA", T.expected_sarsa, dict(alpha=0.5, eps=0.1)),
        ("Q-learning", T.q_learning, dict(alpha=0.5, eps=0.1)),
        ("Double Q-learning", T.double_q_learning, dict(alpha=0.5, eps=0.1)),
        ("4-step SARSA", T.n_step_sarsa, dict(n=4, alpha=0.1, eps=0.1)),
    ]
    control = []
    for name, fn, kw in algos:
        r = run_control(name, fn, args.runs, E if "Monte" not in name else 4 * E, **kw)
        control.append(r)
        print(f"{name:<26s} online(last100) {r['online_return_last100']:8.2f} | greedy median "
              f"{r['greedy_return_median']:7.2f} | greedy reaches goal {r['greedy_reaches_goal']:.2f}")
    out["control"] = control

    # ---------------- n-step / step-size sweep
    sweep = []
    for n in (1, 2, 4, 8):
        for alpha in (0.05, 0.1, 0.2, 0.4):
            vals = [np.mean(T.n_step_sarsa(WarehouseGridEnv(), 300, n=n, alpha=alpha, seed=s).returns[-100:])
                    for s in range(max(5, args.runs // 4))]
            sweep.append({"n": n, "alpha": alpha, "return": float(np.mean(vals))})
    out["n_step_sweep"] = sweep

    # ---------------- TD(0) prediction of v_pi for the DP-optimal policy on a slippery floor
    env = WarehouseGridEnv(slip=0.1)
    V_star, pi_star, _, _ = T.value_iteration(env.P, env.n_states, 4, 0.99)
    v_pi = T.policy_evaluation(env.P, pi_star, env.n_states, 0.99)
    valid = [s for s in env.valid_states() if env.cell(s) in (".", "S")]
    rmse = {}
    for alpha in (0.02, 0.05, 0.1):
        errs = []
        for k in (25, 50, 100, 250, 500, 1000, 2000):
            V = T.td0_prediction(WarehouseGridEnv(slip=0.1), pi_star, k, alpha=alpha, gamma=0.99, seed=1,
                                 starts=valid)
            errs.append(float(np.sqrt(np.mean((V[valid] - v_pi[valid]) ** 2))))
        rmse[str(alpha)] = errs
    out["td0_prediction"] = {"episodes": [25, 50, 100, 250, 500, 1000, 2000], "rmse_by_alpha": rmse}

    save_json(out, RESULTS / "tabular.json")
    print("saved", RESULTS / "tabular.json")


if __name__ == "__main__":
    main()
