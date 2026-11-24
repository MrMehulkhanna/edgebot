"""LaTeX tables for the "recent techniques (2024-2026)" chapter, built from results JSON.
Only writes new files (tab_robust.tex, tab_simba.tex); the original tables are not touched.

    python scripts/build_recent_tables.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgebot.utils import RESULTS, ROOT, load_json  # noqa: E402

OUT = ROOT / "docs" / "generated"
SHIELD = {False: "none", True: "original", "robust": "robust"}


def pct(x):
    return f"{100 * x:.1f}\\%"


def robust_table():
    r = load_json(RESULTS / "robustness_eval.json")
    rows, last = [], None
    for row in r["results"]:
        corrupted = row["condition"] == "noise + spurious (Omni-Perception)"
        if not (corrupted or (row["condition"] == "clean" and row["model"] == "PPO (deployed, FP32)")):
            continue
        model = row["model"] if row["model"] != last else ""
        last = row["model"]
        cond = "clean" if row["condition"] == "clean" else "corrupted"
        rows.append(f"{model} & {cond} & {SHIELD[row['shield']]} & {pct(row['success_rate'])} & "
                    f"{pct(row['collision_rate'])} & {pct(row['timeout_rate'])} \\\\")
    body = ("\\begin{tabular}{lllrrr}\n\\toprule\nPolicy & Lidar & Shield & Success & Collision & Timeout \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "tab_robust.tex").write_text(body)


def simba_table():
    rows = []
    for tag, label in (("nav_ppo_continuous", "PPO, plain MLP (deployed)"), ("nav_ppo_simba", "PPO, SimBa network")):
        e = load_json(RESULTS / f"{tag}_eval.json")
        h = load_json(RESULTS / f"{tag}_history.json")
        first = next((s for s, v in zip(h["steps"], h["success"]) if v >= 0.9), None)
        first = f"{first / 1e6:.2f}M" if first else "--"
        rows.append(f"{label} & {first} & {pct(e['success_rate'])} & {pct(e['collision_rate'])} \\\\")
    body = ("\\begin{tabular}{lrrr}\n\\toprule\nNetwork & Steps to 90\\% (training) & Success & Collision \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "tab_simba.tex").write_text(body)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    robust_table()
    simba_table()
    print("wrote", OUT / "tab_robust.tex", OUT / "tab_simba.tex")
