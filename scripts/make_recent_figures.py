"""Figures for the safety-shield and recent-techniques chapters.

    python scripts/make_recent_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from edgebot.utils import FIGURES, RESULTS, load_json  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "legend.frameon": False})


def shield_figure():
    r = load_json(RESULTS / "robustness_eval.json")["results"]
    get = lambda m, c, s, k: next(x[k] for x in r if x["model"] == m and x["condition"] == c and x["shield"] == s)
    m = "PPO (deployed, FP32)"
    conds = [("clean", "Clean lidar"), ("noise + spurious (Omni-Perception)", "Corrupted lidar")]
    shields = [(False, "No shield"), (True, "Original shield"), ("robust", "Robust shield")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
    x = np.arange(len(conds))
    for ax, key, title in zip(axes, ("success_rate", "collision_rate"), ("Success rate", "Collision rate")):
        for k, (s, lab) in enumerate(shields):
            vals = [get(m, c, s, key) for c, _ in conds]
            bars = ax.bar(x + (k - 1) * 0.26, vals, 0.24, color=C[k], label=lab)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{100 * v:.1f}%", ha="center", fontsize=7)
        ax.set_xticks(x, [lab for _, lab in conds])
        ax.set_title(title, fontweight="bold")
    axes[0].set_ylim(0, 1.12)
    axes[1].set_ylim(0, 0.05)
    axes[0].legend(fontsize=8, loc="center left")
    fig.suptitle("Deployed PPO on 300 unseen layouts: effect of the safety shield", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "shield_robustness.png", bbox_inches="tight")


def recent_curves():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for k, (tag, lab) in enumerate((("nav_ppo_continuous", "PPO, plain MLP (deployed)"),
                                    ("nav_ppo_simba", "PPO, SimBa network (ICLR 2025)"),
                                    ("nav_ppo_dr", "PPO, domain randomisation"))):
        h = load_json(RESULTS / f"{tag}_history.json")
        ax.plot(np.array(h["steps"]) / 1e6, h["success"], color=C[k], label=lab, lw=1.6)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Training success rate")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 5)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Learning speed: SimBa reaches 90% about 4x sooner", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "recent_learning_curves.png", bbox_inches="tight")


if __name__ == "__main__":
    shield_figure()
    recent_curves()
    print("wrote shield_robustness.png, recent_learning_curves.png")
