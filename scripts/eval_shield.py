"""Novelty experiment: does the lidar safety shield make FP32 and INT8 policies safer?

Evaluates every exported navigation policy on the same unseen layouts with and
without edgebot.rl.shield.LidarShield and writes results/shield_eval.json.

    python scripts/eval_shield.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgebot.edge.export import OnnxPolicy  # noqa: E402
from edgebot.rl.evaluate import evaluate_nav  # noqa: E402
from edgebot.utils import CHECKPOINTS, RESULTS, save_json  # noqa: E402

ONNX = CHECKPOINTS / "onnx"
MODELS = [
    ("PPO lidar FP32", "nav_ppo_actor.onnx", {}),
    ("PPO lidar INT8", "nav_ppo_actor_int8.onnx", {}),
    ("DQN pixels FP32", "nav_dqn_pixels.onnx", {"obs_mode": "pixels", "action_mode": "discrete"}),
    ("DQN pixels INT8 (dynamic)", "nav_dqn_pixels_int8.onnx", {"obs_mode": "pixels", "action_mode": "discrete"}),
    ("DQN pixels INT8 (static)", "nav_dqn_pixels_int8_static.onnx",
     {"obs_mode": "pixels", "action_mode": "discrete"}),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    args = p.parse_args()
    out = {"episodes": args.episodes, "results": []}
    for name, f, kw in MODELS:
        pol = OnnxPolicy(ONNX / f, discrete=kw.get("action_mode") == "discrete")
        for sh in (False, True):
            m = evaluate_nav(pol, args.episodes, shield=sh, **kw)
            row = {"model": name, "shield": sh, **{k: m[k] for k in (
                "success_rate", "collision_rate", "timeout_rate", "path_efficiency")}}
            row["intervention_rate"] = m.get("shield_intervention_rate", 0.0)
            out["results"].append(row)
            print(row, flush=True)
    save_json(out, RESULTS / "shield_eval.json")


if __name__ == "__main__":
    main()
