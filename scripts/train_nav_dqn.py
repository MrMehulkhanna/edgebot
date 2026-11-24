"""Train a DQN navigation agent on lidar or egocentric pixel observations.

    python scripts/train_nav_dqn.py --obs lidar  --steps 1000000
    python scripts/train_nav_dqn.py --obs pixels --steps 1500000 --device cuda
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from edgebot.envs.jetbot_nav import make_vec_envs  # noqa: E402
from edgebot.rl.dqn import DQN, DQNConfig  # noqa: E402
from edgebot.rl.evaluate import evaluate_nav  # noqa: E402
from edgebot.utils import CHECKPOINTS, RESULTS, ensure_dirs, get_device, save_json, seed_everything  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", choices=["lidar", "pixels"], default="lidar")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-double", action="store_true")
    p.add_argument("--no-dueling", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--eval-episodes", type=int, default=300)
    args = p.parse_args()

    ensure_dirs()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = get_device(args.device)
    tag = args.tag or f"dqn_{args.obs}"
    venv = make_vec_envs(args.n_envs, seed=args.seed * 1000 + 500, obs_mode=args.obs, action_mode="discrete")
    pixels = args.obs == "pixels"
    cfg = DQNConfig(total_steps=args.steps, n_envs=args.n_envs, seed=args.seed,
                    buffer_size=100_000 if pixels else 300_000, double=not args.no_double,
                    dueling=not args.no_dueling, lr=2.5e-4 if pixels else 5e-4)
    agent = DQN(venv.observation_space.shape, venv.action_space.n, cfg, device)
    history = agent.train(venv, log_every_steps=20_000)
    agent.save(str(CHECKPOINTS / f"nav_{tag}.pt"), {"obs_mode": args.obs})
    save_json(history, RESULTS / f"nav_{tag}_history.json")

    metrics = evaluate_nav(lambda o: agent.predict(o), args.eval_episodes, obs_mode=args.obs,
                           action_mode="discrete")
    metrics.update({"algorithm": "Double Dueling DQN" if cfg.double and cfg.dueling else "DQN",
                    "obs_mode": args.obs, "train_steps": args.steps, "train_time_s": history["time"][-1]})
    save_json(metrics, RESULTS / f"nav_{tag}_eval.json")
    print(metrics)


if __name__ == "__main__":
    main()
