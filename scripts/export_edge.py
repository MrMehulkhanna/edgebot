"""Export every deployable model to ONNX, quantise to INT8 and benchmark it.

Measures, for each model: file size, batch-1 latency (PyTorch CPU / CUDA, ONNX
Runtime FP32 / INT8 on a single CPU thread - the constrained edge setting),
numerical fidelity versus PyTorch and the task metric after quantisation.

    python scripts/export_edge.py
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from edgebot.edge.export import (  # noqa: E402
    OnnxNLU,
    OnnxPolicy,
    collect_observations,
    export_dqn,
    export_nlu,
    export_ppo_actor,
    quantize_int8,
    quantize_int8_static,
    session,
)
from edgebot.envs.jetbot_nav import JetBotNavEnv  # noqa: E402
from edgebot.nlu.data import build_splits  # noqa: E402
from edgebot.nlu.engine import NLU, encode_batch  # noqa: E402
from edgebot.rl.dqn import DQN  # noqa: E402
from edgebot.rl.evaluate import evaluate_nav  # noqa: E402
from edgebot.rl.ppo import PPO  # noqa: E402
from edgebot.utils import CHECKPOINTS, RESULTS, save_json  # noqa: E402

EXPORT = CHECKPOINTS / "onnx"


def latency(fn, n=300, warmup=30):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}


def size_kb(p):
    return Path(p).stat().st_size / 1024


def bench_torch(module, inputs):
    out = {}
    torch.set_num_threads(1)
    module = module.cpu().eval()
    with torch.no_grad():
        out["torch_cpu_1thread"] = latency(lambda: module(*inputs))
        if torch.cuda.is_available():
            m = module.cuda()
            gi = [x.cuda() for x in inputs]

            def run():
                m(*gi)
                torch.cuda.synchronize()

            out["torch_cuda"] = latency(run)
            module.cpu()
    return out


def bench_onnx(path, feeds):
    s = session(path, threads=1)
    return latency(lambda: s.run(None, feeds))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nav-episodes", type=int, default=200)
    args = p.parse_args()
    report = {"device": {"torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                         else None}}
    obs_dim = JetBotNavEnv().observation_space.shape[0]

    # ------------------------------------------------ PPO navigation policy
    ppo = PPO.load(str(CHECKPOINTS / "nav_ppo_continuous.pt"), obs_dim, 2)
    f32 = export_ppo_actor(ppo, obs_dim, EXPORT / "nav_ppo_actor.onnx")
    i8 = quantize_int8(f32)
    from edgebot.rl.networks import DeterministicActor

    actor = DeterministicActor(ppo.ac).eval()
    x = torch.as_tensor(JetBotNavEnv().reset(seed=0)[0])[None]
    entry = {"params": sum(p.numel() for p in actor.parameters()), "size_kb_fp32": size_kb(f32),
             "size_kb_int8": size_kb(i8)}
    entry.update(bench_torch(actor, [x]))
    feeds = {"obs": x.numpy()}
    entry["onnx_fp32_cpu"] = bench_onnx(f32, feeds)
    entry["onnx_int8_cpu"] = bench_onnx(i8, feeds)
    real_obs = collect_observations(OnnxPolicy(f32), episodes=20)
    with torch.no_grad():
        ref = actor(torch.as_tensor(real_obs)).numpy()
    for name, path in (("fp32", f32), ("int8", i8)):
        out = session(path).run(None, {"obs": real_obs})[0]
        entry[f"mean_abs_action_err_{name}"] = float(np.abs(out - ref).mean())
    for name, path in (("fp32", f32), ("int8", i8)):
        pol = OnnxPolicy(path)
        m = evaluate_nav(pol, args.nav_episodes)
        entry[f"nav_success_{name}"] = m["success_rate"]
        entry[f"nav_collision_{name}"] = m["collision_rate"]
    report["nav_ppo"] = entry
    print("PPO", {k: v for k, v in entry.items()})

    # ------------------------------------------------ DQN (pixels) policy
    dqn_path = CHECKPOINTS / "nav_dqn_pixels.pt"
    if dqn_path.exists():
        dqn = DQN.load(str(dqn_path))
        f32 = export_dqn(dqn, EXPORT / "nav_dqn_pixels.onnx")
        i8 = quantize_int8(f32)
        x = torch.rand((1,) + tuple(dqn.obs_shape))
        entry = {"params": sum(p.numel() for p in dqn.q.parameters()), "size_kb_fp32": size_kb(f32),
                 "size_kb_int8": size_kb(i8)}
        entry.update(bench_torch(dqn.q, [x]))
        feeds = {"obs": x.numpy()}
        entry["onnx_fp32_cpu"] = bench_onnx(f32, feeds)
        entry["onnx_int8_cpu"] = bench_onnx(i8, feeds)
        # quantisation study: per-tensor dynamic -> per-channel dynamic -> calibrated static (QDQ)
        i8_pc = quantize_int8(f32, f32.with_name("nav_dqn_pixels_int8_perchannel.onnx"), per_channel=True)
        calib = collect_observations(OnnxPolicy(f32, discrete=True), obs_mode="pixels", action_mode="discrete")
        i8_st = quantize_int8_static(f32, calib)
        entry["onnx_int8_static_cpu"] = bench_onnx(i8_st, feeds)
        entry["size_kb_int8_static"] = size_kb(i8_st)
        for name, path in (("fp32", f32), ("int8", i8), ("int8_perchannel", i8_pc), ("int8_static", i8_st)):
            pol = OnnxPolicy(path, discrete=True)
            m = evaluate_nav(pol, args.nav_episodes // 2, obs_mode="pixels", action_mode="discrete")
            entry[f"nav_success_{name}"] = m["success_rate"]
        report["nav_dqn_pixels"] = entry
        print("DQN", entry)

    # ------------------------------------------------ NLU
    nlu = NLU.load(str(CHECKPOINTS / "nlu_jointbert_mini.pt"))
    f32 = export_nlu(nlu, EXPORT / "nlu_jointbert_mini.onnx")
    i8 = quantize_int8(f32)
    toks = [["move", "the", "red", "box", "from", "shelf", "a", "to", "the", "conveyor"]]
    w, c, m = encode_batch(toks, nlu.vocab)
    entry = {"params": sum(p.numel() for p in nlu.model.parameters()), "size_kb_fp32": size_kb(f32),
             "size_kb_int8": size_kb(i8)}
    entry.update(bench_torch(nlu.model, [w, c, m]))
    feeds = {"words": w.numpy(), "chars": c.numpy(), "mask": m.numpy()}
    entry["onnx_fp32_cpu"] = bench_onnx(f32, feeds)
    entry["onnx_int8_cpu"] = bench_onnx(i8, feeds)
    splits = build_splits()
    entry["torch_test"] = {k: v for k, v in nlu.evaluate(splits["test"]).items() if k != "confusion"}
    for name, path in (("fp32", f32), ("int8", i8)):
        res = OnnxNLU(path, nlu.vocab).evaluate(splits["test"])
        entry[f"onnx_{name}_test"] = {k: v for k, v in res.items() if k != "confusion"}
    report["nlu"] = entry
    print("NLU", entry)

    save_json(report, RESULTS / "edge_benchmark.json")
    print("saved", RESULTS / "edge_benchmark.json")


if __name__ == "__main__":
    main()
