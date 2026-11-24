"""Train the JetBot navigation policy with PPO (continuous or discrete actions).

    python scripts/train_nav_ppo.py --steps 3000000
    python scripts/train_nav_ppo.py --action discrete --steps 2000000 --tag ppo_discrete
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from edgebot.envs.jetbot_nav import make_vec_envs  # noqa: E402
from edgebot.rl.evaluate import evaluate_nav  # noqa: E402
from edgebot.rl.ppo import PPO, PPOConfig  # noqa: E402
from edgebot.utils import CHECKPOINTS, RESULTS, ensure_dirs, save_json, seed_everything  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--action", choices=["continuous", "discrete"], default="continuous")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default=None)
    p.add_argument("--eval-episodes", type=int, default=300)
    p.add_argument("--arch", choices=["mlp", "simba"], default="mlp", help="simba = SimBa network (ICLR 2025)")
    p.add_argument("--lidar-noise", type=float, default=0.0, help="training-time relative range noise")
    p.add_argument("--lidar-mask", type=float, default=0.0, help="training-time fraction of spurious beams")
    args = p.parse_args()

    ensure_dirs()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    tag = args.tag or f"ppo_{args.action}"
    sensor = {"lidar_noise": args.lidar_noise, "lidar_mask": args.lidar_mask}
    venv = make_vec_envs(args.n_envs, seed=args.seed * 1000, action_mode=args.action, **sensor)
    continuous = args.action == "continuous"
    act_dim = venv.action_space.shape[0] if continuous else venv.action_space.n
    cfg = PPOConfig(total_steps=args.steps, n_envs=args.n_envs, seed=args.seed,
                    ent_coef=0.0 if continuous else 0.01, arch=args.arch)
    agent = PPO(venv.observation_space.shape[0], act_dim, continuous, cfg, args.device)

    ckpt = CHECKPOINTS / f"nav_{tag}.pt"

    def checkpoint(agent, history):
        agent.save(str(ckpt), {"action_mode": args.action, "arch": args.arch, **sensor})
        save_json(history, RESULTS / f"nav_{tag}_history.json")

    history = agent.train(venv, log_every=10, callback=checkpoint)
    checkpoint(agent, history)

    metrics = evaluate_nav(lambda o: agent.predict(o, deterministic=True), args.eval_episodes,
                           action_mode=args.action)
    metrics.update({"algorithm": "PPO", "arch": args.arch, **sensor, "action_mode": args.action, "train_steps": args.steps,
                    "train_time_s": history["time"][-1] if history["time"] else None})
    save_json(metrics, RESULTS / f"nav_{tag}_eval.json")
    print(metrics)


if __name__ == "__main__":
    main()
