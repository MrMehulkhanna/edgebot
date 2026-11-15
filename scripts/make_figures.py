"""Generate every figure used in the README, report and slides from results/*.json.

    python scripts/make_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from edgebot.utils import FIGURES, RESULTS, load_json, moving_average  # noqa: E402

# categorical palette (fixed order, validated for colour-vision deficiency)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.facecolor": "white",
})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


def exists(name):
    return (RESULTS / name).exists()


# --------------------------------------------------------------- tabular
def fig_tabular():
    r = load_json(RESULTS / "tabular.json")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k, c in enumerate(r["control"]):
        curve = moving_average(c["mean_curve"], 20)
        x = np.arange(len(curve)) + 20
        if "Monte" in c["name"]:
            x = x / 4  # MC ran 4x more episodes; plot on the same budget axis
            curve, x = curve[::4], x[::4]
        ax.plot(x, curve, color=C[k], label=c["name"])
    ax.set_ylim(-120, 0)
    ax.set_xlabel("Episode (Monte Carlo: per 4 episodes)")
    ax.set_ylabel("Return per episode (20-ep. avg)")
    ax.set_title("Model-free control on the warehouse grid (mean of 20 runs)")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "tabular_learning_curves")

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, key, title in zip(axes, ("slip_0.0", "slip_0.1"), ("Deterministic floor", "Slippery floor (slip = 0.1)")):
        V = np.array(r["dp"][key]["V"])
        layout = np.array(r["layout"])
        Vm = np.ma.masked_where((layout == "#") | (layout == "x"), V)
        im = ax.imshow(Vm, cmap="Blues_r")
        pol = np.array(r["dp"][key]["policy"]).reshape(V.shape)
        arrows = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                if layout[i, j] in (".", "S"):
                    dx, dy = arrows[int(pol[i, j])]
                    ax.arrow(j - dx / 2, i - dy / 2, dx, dy, head_width=0.18, color=INK, length_includes_head=True)
                elif layout[i, j] == "x":
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#3a3a3a"))
                elif layout[i, j] == "#":
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#8b6a46"))
                elif layout[i, j] == "G":
                    ax.text(j, i, "G", ha="center", va="center", color="white", weight="bold")
        ax.set_title(f"{title}: V*(S) = {r['dp'][key]['V_start']:.1f}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.025)
    save(fig, "dp_value_policy")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    eps = r["td0_prediction"]["episodes"]
    for k, (alpha, errs) in enumerate(r["td0_prediction"]["rmse_by_alpha"].items()):
        ax.plot(eps, errs, marker="o", color=C[k], label=f"alpha = {alpha}")
    ax.set_xscale("log")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMSE vs. v_pi from DP")
    ax.set_title("TD(0) prediction converges to v_pi")
    ax.legend()
    save(fig, "td0_prediction")


# --------------------------------------------------------------- navigation
def fig_navigation():
    runs = [("nav_ppo_continuous", "PPO, continuous (lidar)"), ("nav_ppo_discrete", "PPO, discrete (lidar)"),
            ("nav_dqn_lidar", "Double Dueling DQN (lidar)"), ("nav_dqn_pixels", "Double Dueling DQN (pixels, CNN)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, (name, label) in enumerate(runs):
        if not exists(f"{name}_history.json"):
            continue
        h = load_json(RESULTS / f"{name}_history.json")
        steps = np.array(h["steps"]) / 1e6
        axes[0].plot(steps, h["success"], color=C[k], label=label)
        axes[1].plot(steps, h["collision"], color=C[k], label=label)
    axes[0].set_title("Success rate (last 100 training episodes)")
    axes[1].set_title("Collision rate")
    for ax in axes:
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8, loc="lower right")
    save(fig, "nav_learning_curves")

    labels, succ, coll, tout = [], [], [], []
    for name, label in runs:
        if exists(f"{name}_eval.json"):
            e = load_json(RESULTS / f"{name}_eval.json")
            labels.append(label.replace(" (", "\n("))
            succ.append(e["success_rate"])
            coll.append(e["collision_rate"])
            tout.append(e["timeout_rate"])
    if labels:
        fig, ax = plt.subplots(figsize=(8, 3.4))
        x = np.arange(len(labels))
        w = 0.26
        for k, (vals, lab) in enumerate(((succ, "Success"), (coll, "Collision"), (tout, "Timeout"))):
            bars = ax.bar(x + (k - 1) * w, vals, w - 0.02, color=C[k], label=lab)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title("Held-out evaluation: 300 unseen random warehouse layouts")
        ax.legend(ncol=3, loc="upper right")
        save(fig, "nav_eval_bars")


def fig_nav_trajectories():
    from edgebot.envs.jetbot_nav import JetBotNavEnv
    from edgebot.rl.evaluate import EVAL_SEED_OFFSET
    from edgebot.rl.ppo import PPO
    from edgebot.utils import CHECKPOINTS

    ck = CHECKPOINTS / "nav_ppo_continuous.pt"
    if not ck.exists():
        return
    env = JetBotNavEnv()
    agent = PPO.load(str(ck), env.observation_space.shape[0], 2)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for k, ax in enumerate(axes.flat):
        obs, _ = env.reset(seed=EVAL_SEED_OFFSET + 7 + k)
        while True:
            obs, _, term, trunc, info = env.step(agent.predict(obs))
            if term or trunc:
                break
        ax.imshow(env.render(scale=60, show_lidar=False))
        ax.set_title("success" if info["success"] else ("collision" if info["collision"] else "timeout"), fontsize=9)
        ax.axis("off")
    fig.suptitle("PPO navigation policy on unseen layouts (blue = driven path, green ring = goal)", fontsize=10)
    save(fig, "nav_trajectories")


# --------------------------------------------------------------- manipulation
def fig_manipulation():
    r = load_json(RESULTS / "manipulation.json")
    m = r["ik"]["manipulability"]
    grid = np.array(m["grid"])
    grid = np.ma.masked_less(grid, 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    im = axes[0].imshow(grid, origin="lower", extent=[m["x"][0], m["x"][-1], m["z"][0], m["z"][-1]], cmap="Blues",
                        aspect="auto")
    axes[0].set_title("Manipulability (gripper pointing down)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0])
    nr = r["noise_robustness"]
    s = [x["sigma_deg"] for x in nr]
    axes[1].plot(s, [x["pick_place"] for x in nr], marker="o", color=C[0], label="Pick-n-place")
    axes[1].plot(s, [x["assembly"] for x in nr], marker="o", color=C[1], label="3-block assembly")
    axes[1].set_xlabel("Joint encoder noise sigma (deg)")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Robustness to sensor noise")
    axes[1].legend()
    save(fig, "manipulation_analysis")

    if exists("manipulation_trace.json"):
        tr = load_json(RESULTS / "manipulation_trace.json")
        t = np.array(tr["t"])
        q, qr, tau = np.degrees(tr["q"]), np.degrees(tr["q_ref"]), np.array(tr["tau"])
        fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
        for j in range(3):
            axes[0].plot(t, qr[:, j], color=C[j], lw=1.2, ls="--")
            axes[0].plot(t, q[:, j], color=C[j], lw=1.6, label=f"joint {j + 1}")
            axes[1].plot(t, tau[:, j], color=C[j], lw=1.2, label=f"joint {j + 1}")
        axes[0].set_ylabel("angle (deg)")
        axes[0].set_title("Joint tracking during pick-n-place + assembly (dashed = reference)")
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].set_ylabel("torque (N m)")
        axes[1].set_xlabel("time (s)")
        save(fig, "joint_tracking")


# --------------------------------------------------------------- NLU
def fig_nlu():
    r = load_json(RESULTS / "nlu_eval.json")
    models = r["models"]
    names = [("bow_logreg", "BoW + LogReg"), ("jointbert_mini_scratch", "JointBERT-mini\n(scratch)"),
             ("jointbert_distilbert", "JointBERT\n(DistilBERT)"),
             ("jointbert_mini_distilled", "JointBERT-mini\n(distilled)")]
    names = [(k, v) for k, v in names if k in models]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(names))
    w = 0.38
    for k, split in enumerate(("test", "test_noisy")):
        vals = []
        for key, _ in names:
            m = models[key]
            vals.append(m["intent_acc"][split] if key == "bow_logreg" else m[split]["intent_acc"])
        bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.03, color=C[k],
                      label="unseen templates" if split == "test" else "unseen templates + ASR noise")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [v for _, v in names], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Intent accuracy")
    ax.set_title("Intent detection generalisation")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    save(fig, "nlu_comparison")

    key = r.get("deployed", "jointbert_mini_scratch")
    M = np.array(models[key]["test"]["confusion"], dtype=float)
    M = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(r["intents"])), r["intents"], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(r["intents"])), r["intents"], fontsize=8)
    for i in range(len(M)):
        for j in range(len(M)):
            if M[i, j] >= 0.005:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.6 else INK)
    ax.grid(False)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Deployed NLU: intent confusion (unseen templates)")
    save(fig, "nlu_confusion")


# --------------------------------------------------------------- edge
def fig_edge():
    r = load_json(RESULTS / "edge_benchmark.json")
    candidates = (("nav_ppo", "PPO actor"), ("nav_dqn_pixels", "DQN CNN"), ("nlu", "JointBERT-mini"))
    models = [(k, lab) for k, lab in candidates if k in r]
    variants = [("torch_cpu_1thread", "PyTorch CPU"), ("torch_cuda", "PyTorch CUDA"),
                ("onnx_fp32_cpu", "ONNX RT FP32"), ("onnx_int8_cpu", "ONNX RT INT8")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(models))
    w = 0.2
    for k, (v, lab) in enumerate(variants):
        vals = [r[m].get(v, {}).get("p50_ms", np.nan) for m, _ in models]
        bars = axes[0].bar(x + (k - 1.5) * w, vals, w - 0.02, color=C[k], label=lab)
        for b, val in zip(bars, vals):
            if not np.isnan(val):
                axes[0].text(b.get_x() + b.get_width() / 2, val * 1.08, f"{val:.2f}", ha="center", fontsize=6,
                             color=INK2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [lab for _, lab in models])
    axes[0].set_ylabel("median latency, batch 1 (ms, log)")
    axes[0].set_title("Inference latency (1 CPU thread)")
    axes[0].legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for k, (key, lab) in enumerate((("size_kb_fp32", "FP32"), ("size_kb_int8", "INT8"))):
        vals = [r[m][key] for m, _ in models]
        bars = axes[1].bar(x + (k - 0.5) * 0.38, vals, 0.35, color=C[k + 2], label=lab)
        for b, val in zip(bars, vals):
            axes[1].text(b.get_x() + b.get_width() / 2, val * 1.02, f"{val:.0f}", ha="center", fontsize=7, color=INK2)
    axes[1].set_xticks(x, [lab for _, lab in models])
    axes[1].set_ylabel("ONNX file size (KB)")
    axes[1].set_title("Model size after INT8 quantisation")
    axes[1].legend()
    save(fig, "edge_benchmark")


def main():
    for name, fn, need in ((
            "tabular", fig_tabular, "tabular.json"), ("navigation", fig_navigation, None),
            ("trajectories", fig_nav_trajectories, None), ("manipulation", fig_manipulation, "manipulation.json"),
            ("nlu", fig_nlu, "nlu_eval.json"), ("edge", fig_edge, "edge_benchmark.json")):
        if need is None or exists(need):
            fn()
        else:
            print(f"skipping {name}: results/{need} not found")


if __name__ == "__main__":
    main()
