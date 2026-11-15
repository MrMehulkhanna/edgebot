"""Task learning for the manipulator: PPO learns joint-velocity control for reaching,
then is compared with the analytic IK + trajectory pipeline on the same targets.

    python scripts/train_arm_reach.py --steps 1500000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from edgebot.manipulation.arm import PlanarArm, quintic  # noqa: E402
from edgebot.manipulation.reach_env import ArmReachEnv, ArmVecEnv  # noqa: E402
from edgebot.rl.ppo import PPO, PPOConfig  # noqa: E402
from edgebot.utils import CHECKPOINTS, RESULTS, ensure_dirs, save_json, seed_everything  # noqa: E402


def evaluate(agent, episodes=500, seed=123):
    env = ArmReachEnv()
    arm = PlanarArm()
    rl_err, rl_succ, rl_steps, ik_err = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        q0, target = env.q.copy(), env.target.copy()
        while True:
            obs, _, term, trunc, info = env.step(agent.predict(obs, deterministic=True))
            if term or trunc:
                break
        rl_err.append(info["pos_err"])
        rl_succ.append(info["success"])
        rl_steps.append(env.t)
        # analytic baseline: IK + min-jerk joint trajectory over the same horizon
        q_goal = arm.ik_analytic(target[0], target[1], -np.pi / 2)
        _, q_traj, _, _ = quintic(q0, q_goal, 2.0, 0.05)
        ik_err.append(float(np.linalg.norm(arm.fk(q_traj[-1])[:2] - target)))
    return {
        "episodes": episodes,
        "rl_success_rate": float(np.mean(rl_succ)),
        "rl_mean_error_mm": float(np.mean(rl_err) * 1000),
        "rl_median_error_mm": float(np.median(rl_err) * 1000),
        "rl_mean_steps": float(np.mean(rl_steps)),
        "ik_mean_error_mm": float(np.mean(ik_err) * 1000),
        "ik_success_rate": float(np.mean(np.array(ik_err) < env.tol)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_500_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    venv = ArmVecEnv(args.n_envs, seed=args.seed * 1000 + 77)
    cfg = PPOConfig(total_steps=args.steps, n_envs=args.n_envs, n_steps=128, seed=args.seed, hidden=(128, 128))
    agent = PPO(venv.observation_space.shape[0], 3, True, cfg, "cpu")
    history = agent.train(venv, log_every=10)
    agent.save(str(CHECKPOINTS / "arm_reach_ppo.pt"))
    save_json(history, RESULTS / "arm_reach_history.json")
    metrics = evaluate(agent)
    metrics["train_time_s"] = history["time"][-1]
    save_json(metrics, RESULTS / "arm_reach_eval.json")
    print(metrics)


if __name__ == "__main__":
    main()
