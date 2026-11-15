"""Render LaTeX tables for the report and slides directly from results/*.json, so
every number in the documents is the number the experiments produced.

    python scripts/build_tables.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgebot.utils import RESULTS, ROOT, load_json  # noqa: E402

OUT = ROOT / "docs" / "generated"


def pct(x):
    return f"{100 * x:.1f}\\%"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", OUT / f"{name}.tex")


def tabular_tables():
    r = load_json(RESULTS / "tabular.json")
    rows = []
    for c in r["control"]:
        rows.append(f"{c['name']} & {c['online_return_last100']:.1f} & {c['greedy_return_median']:.0f} & "
                    f"{pct(c['greedy_reaches_goal'])} \\\\")
    write("tab_control", "\\begin{tabular}{lrrr}\n\\toprule\nAlgorithm & Online return & Greedy return & "
          "Greedy reaches goal \\\\\n & (last 100 ep.) & (median) & \\\\\n\\midrule\n" + "\n".join(rows) +
          "\n\\bottomrule\n\\end{tabular}\n")
    rows = []
    for key, d in r["dp"].items():
        rows.append(f"{key.split('_')[1]} & {d['gamma']} & {d['vi_iterations']} & {d['pi_improvements']} & "
                    f"{d['V_start']:.2f} & {d['max_value_gap']:.1e} \\\\")
    write("tab_dp", "\\begin{tabular}{rrrrrr}\n\\toprule\nSlip & $\\gamma$ & VI sweeps & PI improvements & "
          "$V^*(S)$ & $\\max|V_{VI}-V_{PI}|$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def nav_table():
    runs = [("nav_ppo_continuous", "PPO", "lidar", "continuous"), ("nav_ppo_discrete", "PPO", "lidar", "discrete"),
            ("nav_dqn_lidar", "Double Dueling DQN", "lidar", "discrete"),
            ("nav_dqn_pixels", "Double Dueling DQN", "pixels (CNN)", "discrete")]
    rows = []
    for name, algo, obs, act in runs:
        p = RESULTS / f"{name}_eval.json"
        if not p.exists():
            continue
        e = load_json(p)
        steps = e["train_steps"] / 1e6
        rows.append(f"{algo} & {obs} & {act} & {steps:.1f}M & {pct(e['success_rate'])} & {pct(e['collision_rate'])} & "
                    f"{pct(e['timeout_rate'])} & {e['path_efficiency']:.2f} \\\\")
    write("tab_nav", "\\begin{tabular}{lllrrrrr}\n\\toprule\nAlgorithm & Obs. & Actions & Steps & Success & "
          "Collision & Timeout & Path eff. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def manip_tables():
    m = load_json(RESULTS / "manipulation.json")
    ik = m["ik"]
    write("tab_ik", "\\begin{tabular}{lrrr}\n\\toprule\nSolver & Mean error & Mean time & Notes \\\\\n\\midrule\n"
          f"Analytic (closed form) & {ik['analytic']['mean_err_m']:.1e} m & {ik['analytic']['mean_time_us']:.0f} "
          f"$\\mu$s & exact, {ik['analytic']['limit_failures']} limit rejections / {ik['samples']} \\\\\n"
          f"Damped least squares & {1000 * ik['dls']['mean_err_m']:.1f} mm & {ik['dls']['mean_time_us']:.0f} $\\mu$s & "
          f"{ik['dls']['mean_iterations']:.0f} iter., {pct(ik['dls']['success_rate_1mm'])} within 1 mm \\\\\n"
          "\\bottomrule\n\\end{tabular}\n")
    pp, asm = m["pick_place"], m["assembly"]
    rows = [f"Pick-n-place & {pp['trials']} & {pct(pp['success_rate'])} & {pp['mean_duration_s']:.1f} s & "
            f"{pp['mean_tracking_rmse_deg']:.2f}$^\\circ$ \\\\",
            f"3-block assembly & {asm['trials']} & {pct(asm['success_rate'])} & "
            f"{asm['mean_duration_s']:.1f} s & -- \\\\"]
    arm = RESULTS / "arm_reach_eval.json"
    if arm.exists():
        a = load_json(arm)
        rows.append(f"RL reaching (PPO) & {a['episodes']} & {pct(a['rl_success_rate'])} & "
                    f"{a['rl_mean_steps'] * 0.05:.2f} s & {a['rl_mean_error_mm']:.1f} mm err. \\\\")
    write("tab_manip", "\\begin{tabular}{lrrrr}\n\\toprule\nTask & Trials & Success & Duration & Tracking RMSE "
          "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def nlu_table():
    r = load_json(RESULTS / "nlu_eval.json")
    M = r["models"]
    rows = []
    b = M["bow_logreg"]["intent_acc"]
    rows.append(f"BoW + logistic regression & -- & {pct(b['test'])} & -- & -- & {pct(b['test_noisy'])} \\\\")
    for key, label in (("jointbert_mini_scratch", "JointBERT-mini (scratch)"),
                       ("jointbert_distilbert", "JointBERT (DistilBERT fine-tuned)"),
                       ("jointbert_mini_distilled", "JointBERT-mini (distilled, deployed)")):
        if key not in M:
            continue
        m = M[key]
        params = f"{m['n_params'] / 1e6:.2f}M"
        rows.append(f"{label} & {params} & {pct(m['test']['intent_acc'])} & {pct(m['test']['slot_f1'])} & "
                    f"{pct(m['test']['frame_acc'])} & {pct(m['test_noisy']['intent_acc'])} \\\\")
    write("tab_nlu", "\\begin{tabular}{lrrrrr}\n\\toprule\nModel & Params & Intent acc. & Slot F1 & Frame acc. & "
          "Intent (noisy) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def voice_table():
    p = RESULTS / "voice_eval.json"
    if not p.exists():
        return
    v = load_json(p)
    rows = []
    for cond, label in (("clean", "Clean"), ("snr20", "20 dB SNR"), ("snr10", "10 dB SNR")):
        c = v["conditions"][cond]
        rows.append(f"{label} & {pct(c['wer'])} & {pct(c['intent_acc'])} & {pct(c['slot_f1'])} & "
                    f"{pct(c['frame_acc'])} \\\\")
    write("tab_voice", "\\begin{tabular}{lrrrr}\n\\toprule\nAudio & WER & Intent acc. & Slot-value F1 & Frame acc. "
          "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    rtf = v["asr_rtf"]
    rows = [f"Whisper-{v['asr_model'].split('-')[-1]} {k.replace('_', ' ')} & {d['rtf']:.3f} & "
            f"{1000 * d['mean_latency_s']:.0f} ms \\\\" for k, d in rtf.items()]
    rows.append(f"Piper TTS (VITS, ONNX) CPU & {v['tts_rtf_cpu']:.3f} & -- \\\\")
    write("tab_speech_rtf", "\\begin{tabular}{lrr}\n\\toprule\nModel / device & Real-time factor & Latency per "
          "command \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def edge_table():
    p = RESULTS / "edge_benchmark.json"
    if not p.exists():
        return
    e = load_json(p)
    rows = []
    for key, label in (("nav_ppo", "PPO actor"), ("nav_dqn_pixels", "DQN (CNN)"), ("nlu", "JointBERT-mini")):
        if key not in e:
            continue
        m = e[key]
        cuda = m.get("torch_cuda", {}).get("p50_ms")
        rows.append(f"{label} & {m['params'] / 1e3:.0f}k & {m['size_kb_fp32']:.0f} / {m['size_kb_int8']:.0f} KB & "
                    f"{m['torch_cpu_1thread']['p50_ms']:.3f} & {cuda:.3f} & {m['onnx_fp32_cpu']['p50_ms']:.3f} & "
                    f"{m['onnx_int8_cpu']['p50_ms']:.3f} \\\\")
    write("tab_edge", "\\begin{tabular}{lrrrrrr}\n\\toprule\nModel & Params & ONNX FP32 / INT8 & PyTorch CPU & "
          "PyTorch CUDA & ORT FP32 & ORT INT8 \\\\\n & & & (ms) & (ms) & (ms) & (ms) \\\\\n\\midrule\n" +
          "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    rows = []
    if "nav_ppo" in e:
        m = e["nav_ppo"]
        rows.append(f"PPO navigation success & {pct(m['nav_success_fp32'])} & {pct(m['nav_success_int8'])} \\\\")
    if "nav_dqn_pixels" in e:
        m = e["nav_dqn_pixels"]
        rows.append(f"DQN (pixels) navigation success & {pct(m['nav_success_fp32'])} & "
                    f"{pct(m['nav_success_int8'])} \\\\")
    if "nlu" in e:
        m = e["nlu"]
        rows.append(f"NLU frame accuracy & {pct(m['onnx_fp32_test']['frame_acc'])} & "
                    f"{pct(m['onnx_int8_test']['frame_acc'])} \\\\")
    if "nav_dqn_pixels" in e and "nav_success_int8_static" in e["nav_dqn_pixels"]:
        m = e["nav_dqn_pixels"]
        drows = [f"FP32 & {m['size_kb_fp32']:.0f} & {m['onnx_fp32_cpu']['p50_ms']:.3f} & "
                 f"{pct(m['nav_success_fp32'])} \\\\",
                 f"INT8 dynamic, per-tensor & {m['size_kb_int8']:.0f} & {m['onnx_int8_cpu']['p50_ms']:.3f} & "
                 f"{pct(m['nav_success_int8'])} \\\\",
                 f"INT8 dynamic, per-channel & -- & -- & {pct(m['nav_success_int8_perchannel'])} \\\\",
                 f"INT8 static QDQ, per-channel, calibrated & {m['size_kb_int8_static']:.0f} & "
                 f"{m['onnx_int8_static_cpu']['p50_ms']:.3f} & {pct(m['nav_success_int8_static'])} \\\\"]
        write("tab_dqn_quant", "\\begin{tabular}{lrrr}\n\\toprule\nDQN (pixels) variant & Size (KB) & "
              "ORT latency (ms) & Nav. success \\\\\n\\midrule\n" + "\n".join(drows) +
              "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_quant", "\\begin{tabular}{lrr}\n\\toprule\nTask metric & ONNX FP32 & ONNX INT8 \\\\\n\\midrule\n" +
          "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    for fn in (tabular_tables, nav_table, manip_tables, nlu_table, voice_table, edge_table):
        fn()
