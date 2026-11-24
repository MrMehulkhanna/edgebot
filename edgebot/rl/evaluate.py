"""Policy evaluation on held-out navigation scenarios."""

from __future__ import annotations

import numpy as np

from edgebot.envs.jetbot_nav import JetBotNavEnv

EVAL_SEED_OFFSET = 1_000_000  # evaluation layouts never overlap training seeds


def evaluate_nav(policy_fn, episodes: int = 200, seed: int = 0, record: int = 0, shield: bool | str = False,
                 **env_kwargs) -> dict:
    """policy_fn(obs) -> action. Returns aggregate metrics (+ optional frames of first episodes).

    shield=True filters every action through edgebot.rl.shield.LidarShield;
    shield="robust" uses the median-filtered shield.
    """
    env = JetBotNavEnv(**env_kwargs)
    guard = None
    if shield:
        from edgebot.rl.shield import LidarShield

        guard = LidarShield(env, median_filter=(shield == "robust"))
    succ, coll, timeout, returns, lengths, eff = [], [], [], [], [], []
    frames = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED_OFFSET + seed * 10_000 + ep)
        straight = np.linalg.norm(env.pose[:2] - env.scn.goal) - env.goal_tol
        G = 0.0
        while True:
            obs, r, term, trunc, info = env.step(guard(policy_fn(obs)) if guard else policy_fn(obs))
            G += r
            if ep < record:
                frames.append(env.render(scale=80))
            if term or trunc:
                break
        trail = np.array(env.trail)
        path_len = float(np.sum(np.linalg.norm(np.diff(trail, axis=0), axis=1)))
        succ.append(info["success"])
        coll.append(info["collision"])
        timeout.append(trunc)
        returns.append(G)
        lengths.append(env.t)
        if info["success"]:
            eff.append(straight / max(path_len, 1e-6))
    out = {
        "episodes": episodes,
        "success_rate": float(np.mean(succ)),
        "collision_rate": float(np.mean(coll)),
        "timeout_rate": float(np.mean(timeout)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(lengths)),
        "path_efficiency": float(np.mean(eff)) if eff else 0.0,
    }
    if guard:
        out["shield_intervention_rate"] = guard.intervention_rate
    if record:
        out["frames"] = frames
    return out
