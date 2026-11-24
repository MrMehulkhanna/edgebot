"""Robustness of the navigation policies to a corrupted lidar, with no shield, the original
safety shield, and the robust (median-filtered) shield.

The corruption follows the LiDAR randomisation used by Omni-Perception (Wang et al., 2025):
10% of beams replaced by a spurious short return in [0, 0.3] m, and up to 10% range noise.
Policies trained on clean data (the deployed PPO and its INT8 copy) are compared with a PPO
trained with that corruption (domain randomisation) and a PPO with a SimBa network.

    python scripts/eval_robustness.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgebot.edge.export import OnnxPolicy  # noqa: E402
from edgebot.envs.jetbot_nav import JetBotNavEnv  # noqa: E402
from edgebot.rl.evaluate import evaluate_nav  # noqa: E402
from edgebot.rl.ppo import PPO  # noqa: E402
from edgebot.utils import CHECKPOINTS, RESULTS, save_json  # noqa: E402

CONDITIONS = {
    "clean": {},
    "noise 10%": {"lidar_noise": 0.1},
    "spurious beams 10%": {"lidar_mask": 0.1},
    "noise + spurious (Omni-Perception)": {"lidar_noise": 0.1, "lidar_mask": 0.1},
}


def load_policies():
    pols = [("PPO (deployed, FP32)", OnnxPolicy(CHECKPOINTS / "onnx" / "nav_ppo_actor.onnx")),
            ("PPO (deployed, INT8)", OnnxPolicy(CHECKPOINTS / "onnx" / "nav_ppo_actor_int8.onnx"))]
    obs_dim = JetBotNavEnv().observation_space.shape[0]
    for tag, label in (("ppo_dr", "PPO + domain randomisation"), ("ppo_simba", "PPO + SimBa network")):
        ck = CHECKPOINTS / f"nav_{tag}.pt"
        if ck.exists():
            agent = PPO.load(str(ck), obs_dim, 2)
            pols.append((label, lambda o, a=agent: a.predict(o, deterministic=True)))
    return pols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    args = p.parse_args()
    out = {"episodes": args.episodes, "results": []}
    for name, pol in load_policies():
        for cond, kw in CONDITIONS.items():
            for sh in (False, True, "robust"):
                m = evaluate_nav(pol, args.episodes, shield=sh, **kw)
                row = {"model": name, "condition": cond, "shield": sh,
                       **{k: m[k] for k in ("success_rate", "collision_rate", "timeout_rate")},
                       "intervention_rate": m.get("shield_intervention_rate", 0.0)}
                out["results"].append(row)
                print(row, flush=True)
    save_json(out, RESULTS / "robustness_eval.json")


if __name__ == "__main__":
    main()
