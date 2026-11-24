"""Tabular RL: Dynamic Programming, Monte Carlo and Temporal-Difference control.

Every algorithm follows the pseudo-code of Sutton & Barto (2nd ed.):

* ``value_iteration``      - Sec. 4.4, Bellman optimality backup
* ``policy_iteration``     - Sec. 4.3, evaluation + greedy improvement
* ``mc_control``           - Sec. 5.4, on-policy first-visit MC with epsilon-soft policy
* ``td0_prediction``       - Sec. 6.1, TD(0) state-value prediction
* ``sarsa``                - Sec. 6.4, on-policy TD control
* ``q_learning``           - Sec. 6.5, off-policy TD control
* ``expected_sarsa``       - Sec. 6.6
* ``n_step_sarsa``         - Sec. 7.2
* ``double_q_learning``    - Sec. 6.7, removes maximisation bias
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TabularResult:
    Q: np.ndarray
    returns: list = field(default_factory=list)
    lengths: list = field(default_factory=list)

    @property
    def policy(self) -> np.ndarray:
        return greedy(self.Q)

    @property
    def V(self) -> np.ndarray:
        return self.Q.max(axis=1)


def greedy(Q: np.ndarray) -> np.ndarray:
    return np.argmax(Q, axis=1)


def epsilon_greedy(Q_s: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(len(Q_s)))
    best = np.flatnonzero(Q_s == Q_s.max())
    return int(rng.choice(best))


# ----------------------------------------------------------------------- DP
def q_from_v(P, V: np.ndarray, n_states: int, n_actions: int, gamma: float) -> np.ndarray:
    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = sum(p * (r + gamma * V[ns] * (not d)) for p, ns, r, d in P[s][a])
    return Q


def value_iteration(P, n_states: int, n_actions: int, gamma: float = 0.99, theta: float = 1e-8,
                    max_iter: int = 10_000):
    """Returns (V*, pi*, iterations, deltas)."""
    V = np.zeros(n_states)
    deltas = []
    for it in range(1, max_iter + 1):
        Q = q_from_v(P, V, n_states, n_actions, gamma)
        V_new = Q.max(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        deltas.append(delta)
        V = V_new
        if delta < theta:
            break
    Q = q_from_v(P, V, n_states, n_actions, gamma)
    return V, greedy(Q), it, deltas


def policy_evaluation(P, pi: np.ndarray, n_states: int, gamma: float, theta: float = 1e-8,
                      max_iter: int = 100_000) -> np.ndarray:
    V = np.zeros(n_states)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(n_states):
            v = sum(p * (r + gamma * V[ns] * (not d)) for p, ns, r, d in P[s][int(pi[s])])
            delta = max(delta, abs(v - V[s]))
            V[s] = v  # in-place (Gauss-Seidel) sweep
        if delta < theta:
            break
    return V


def policy_iteration(P, n_states: int, n_actions: int, gamma: float = 0.99, theta: float = 1e-8,
                     max_iter: int = 1_000):
    """Returns (V*, pi*, number of improvement steps)."""
    pi = np.zeros(n_states, dtype=int)
    for it in range(1, max_iter + 1):
        V = policy_evaluation(P, pi, n_states, gamma, theta)
        Q = q_from_v(P, V, n_states, n_actions, gamma)
        new_pi = greedy(Q)
        # keep the old action on ties so the loop terminates
        stable_mask = np.isclose(Q[np.arange(n_states), pi], Q.max(axis=1))
        new_pi = np.where(stable_mask, pi, new_pi)
        if np.array_equal(new_pi, pi):
            return V, pi, it
        pi = new_pi
    return V, pi, max_iter


# ----------------------------------------------------------- Monte Carlo
def mc_control(env, episodes: int, gamma: float = 0.99, eps: float = 0.1, seed: int = 0,
               eps_decay: float = 1.0, min_eps: float = 0.01) -> TabularResult:
    """On-policy first-visit Monte Carlo control with an epsilon-soft policy."""
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    N = np.zeros((nS, nA))
    res = TabularResult(Q)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 100_003 + ep)
        traj = []
        done = trunc = False
        while not (done or trunc):
            a = epsilon_greedy(Q[s], eps, rng)
            ns, r, done, trunc, _ = env.step(a)
            traj.append((s, a, r))
            s = ns
        G = 0.0
        first_visit = {}
        for t, (s_t, a_t, _) in enumerate(traj):
            first_visit.setdefault((s_t, a_t), t)
        for t in range(len(traj) - 1, -1, -1):
            s_t, a_t, r_t = traj[t]
            G = gamma * G + r_t
            if first_visit[(s_t, a_t)] == t:
                N[s_t, a_t] += 1
                Q[s_t, a_t] += (G - Q[s_t, a_t]) / N[s_t, a_t]
        res.returns.append(sum(x[2] for x in traj))
        res.lengths.append(len(traj))
        eps = max(min_eps, eps * eps_decay)
    return res


# ------------------------------------------------------------------- TD
def td0_prediction(env, policy: np.ndarray, episodes: int, alpha: float = 0.1, gamma: float = 0.99,
                   seed: int = 0, starts=None) -> np.ndarray:
    """TD(0) estimate of v_pi. ``starts`` (optional list of states) gives exploring starts."""
    rng = np.random.default_rng(seed)
    V = np.zeros(env.observation_space.n)
    for ep in range(episodes):
        opts = {"start": int(rng.choice(starts))} if starts is not None else None
        s, _ = env.reset(seed=seed * 100_003 + ep, options=opts)
        done = trunc = False
        while not (done or trunc):
            ns, r, done, trunc, _ = env.step(int(policy[s]))
            V[s] += alpha * (r + gamma * V[ns] * (not done) - V[s])
            s = ns
    return V


def _td_control(env, episodes, alpha, gamma, eps, seed, target: str) -> TabularResult:
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    res = TabularResult(Q)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 100_003 + ep)
        a = epsilon_greedy(Q[s], eps, rng)
        done = trunc = False
        G, T = 0.0, 0
        while not (done or trunc):
            ns, r, done, trunc, _ = env.step(a)
            G += r
            T += 1
            na = epsilon_greedy(Q[ns], eps, rng)
            if done:
                bootstrap = 0.0
            elif target == "sarsa":
                bootstrap = Q[ns, na]
            elif target == "q":
                bootstrap = Q[ns].max()
            else:  # expected sarsa under the epsilon-greedy behaviour policy
                probs = np.full(nA, eps / nA)
                best = np.flatnonzero(Q[ns] == Q[ns].max())
                probs[best] += (1 - eps) / len(best)
                bootstrap = float(probs @ Q[ns])
            Q[s, a] += alpha * (r + gamma * bootstrap - Q[s, a])
            s, a = ns, na
        res.returns.append(G)
        res.lengths.append(T)
    return res


def sarsa(env, episodes: int, alpha=0.5, gamma=1.0, eps=0.1, seed=0) -> TabularResult:
    return _td_control(env, episodes, alpha, gamma, eps, seed, "sarsa")


def q_learning(env, episodes: int, alpha=0.5, gamma=1.0, eps=0.1, seed=0) -> TabularResult:
    return _td_control(env, episodes, alpha, gamma, eps, seed, "q")


def expected_sarsa(env, episodes: int, alpha=0.5, gamma=1.0, eps=0.1, seed=0) -> TabularResult:
    return _td_control(env, episodes, alpha, gamma, eps, seed, "expected")


def double_q_learning(env, episodes: int, alpha=0.5, gamma=1.0, eps=0.1, seed=0) -> TabularResult:
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q1, Q2 = np.zeros((nS, nA)), np.zeros((nS, nA))
    res = TabularResult(Q1)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 100_003 + ep)
        done = trunc = False
        G, T = 0.0, 0
        while not (done or trunc):
            a = epsilon_greedy(Q1[s] + Q2[s], eps, rng)
            ns, r, done, trunc, _ = env.step(a)
            G += r
            T += 1
            if rng.random() < 0.5:
                best = int(np.argmax(Q1[ns]))
                Q1[s, a] += alpha * (r + gamma * Q2[ns, best] * (not done) - Q1[s, a])
            else:
                best = int(np.argmax(Q2[ns]))
                Q2[s, a] += alpha * (r + gamma * Q1[ns, best] * (not done) - Q2[s, a])
            s = ns
        res.returns.append(G)
        res.lengths.append(T)
    res.Q = (Q1 + Q2) / 2
    return res


def n_step_sarsa(env, episodes: int, n: int = 4, alpha=0.5, gamma=1.0, eps=0.1, seed=0) -> TabularResult:
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    res = TabularResult(Q)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 100_003 + ep)
        S, A, R = [s], [epsilon_greedy(Q[s], eps, rng)], [0.0]
        T = np.inf
        t = 0
        G_ep = 0.0
        while True:
            if t < T:
                ns, r, done, trunc, _ = env.step(A[t])
                S.append(ns)
                R.append(r)
                G_ep += r
                if done or trunc:
                    T = t + 1
                    terminal = done
                else:
                    A.append(epsilon_greedy(Q[ns], eps, rng))
            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma ** (i - tau - 1) * R[i] for i in range(tau + 1, int(min(tau + n, T)) + 1))
                if tau + n < T:
                    G += gamma ** n * Q[S[tau + n], A[tau + n]]
                elif not terminal:  # truncated: bootstrap from the last state
                    G += gamma ** (T - tau) * Q[S[int(T)]].max()
                Q[S[tau], A[tau]] += alpha * (G - Q[S[tau], A[tau]])
            if tau == T - 1:
                break
            t += 1
        res.returns.append(G_ep)
        res.lengths.append(int(T))
    return res


def rollout_greedy(env, policy: np.ndarray, max_steps: int = 200, seed: int = 0):
    """Executes the greedy policy once; returns (return, path of states)."""
    s, _ = env.reset(seed=seed)
    path, G = [s], 0.0
    for _ in range(max_steps):
        s, r, done, trunc, _ = env.step(int(policy[s]))
        G += r
        path.append(s)
        if done or trunc:
            break
    return G, path
