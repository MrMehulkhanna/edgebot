"""Edge deployment: ONNX export, INT8 post-training quantisation and ONNX Runtime
inference wrappers.

ONNX is the interchange format consumed by TensorRT on NVIDIA Jetson
(``trtexec --onnx=model.onnx --fp16``) and by ONNX Runtime on any ARM/x86 CPU, so
the same artefacts produced here run on the robot. Dynamic (weight-only) INT8
quantisation shrinks the linear layers 4x without any calibration data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from edgebot.rl.networks import DeterministicActor


def _export(model: torch.nn.Module, args: tuple, path: Path, input_names, output_names, dynamic_axes):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, args, str(path), input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes, opset_version=17, dynamo=False)
    import onnx

    onnx.checker.check_model(onnx.load(str(path)))
    return path


def export_ppo_actor(agent, obs_dim: int, path: Path) -> Path:
    actor = DeterministicActor(agent.ac).cpu()
    return _export(actor, (torch.zeros(1, obs_dim),), path, ["obs"], ["action"],
                   {"obs": {0: "batch"}, "action": {0: "batch"}})


def export_dqn(agent, path: Path) -> Path:
    q = agent.q.cpu()
    return _export(q, (torch.zeros((1,) + tuple(agent.obs_shape)),), path, ["obs"], ["q_values"],
                   {"obs": {0: "batch"}, "q_values": {0: "batch"}})


class _NLUExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, words, chars, mask):
        intent, slots = self.model(words, chars, mask)
        return torch.softmax(intent, -1), slots.argmax(-1)


def export_nlu(nlu, path: Path) -> Path:
    """Exports with the torch.export-based exporter, which keeps batch and sequence
    length symbolic through the attention reshapes (the TorchScript tracer bakes them in)."""
    import onnx

    from edgebot.nlu.engine import encode_batch

    nlu.model.cpu().eval()
    # the fused "fast path" kernel of nn.TransformerEncoderLayer has no ONNX symbolic
    torch.backends.mha.set_fastpath_enabled(False)
    words, chars, mask = encode_batch([["go", "to", "the", "conveyor", "now"], ["hi"]], nlu.vocab, "cpu")
    batch = torch.export.Dim("batch", min=1, max=512)
    seq = torch.export.Dim("seq", min=1, max=48)
    shapes = {"words": {0: batch, 1: seq}, "chars": {0: batch, 1: seq}, "mask": {0: batch, 1: seq}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(_NLUExport(nlu.model), (words, chars, mask), str(path), dynamo=True,
                          input_names=["words", "chars", "mask"], output_names=["intent_probs", "slot_ids"],
                          dynamic_shapes=shapes, opset_version=18, external_data=False)
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def quantize_int8(src: Path, dst: Path | None = None, per_channel: bool = False) -> Path:
    """Dynamic quantisation: INT8 weights, activations quantised at run time."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst = dst or src.with_name(src.stem + "_int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=per_channel)
    return dst


def quantize_int8_static(src: Path, samples: np.ndarray, input_name: str = "obs", dst: Path | None = None) -> Path:
    """Static post-training quantisation (QDQ, per-channel weights) with activation
    ranges calibrated on real observations - the format TensorRT consumes for INT8."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{input_name: x[None].astype(np.float32)} for x in samples])

        def get_next(self):
            return next(self.it, None)

    dst = dst or src.with_name(src.stem + "_int8_static.onnx")
    quantize_static(str(src), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    return dst


def collect_observations(policy, episodes: int = 40, every: int = 3, seed: int = 5000, **env_kwargs) -> np.ndarray:
    """Rolls out a policy and returns visited observations (calibration / fidelity data)."""
    from edgebot.envs.jetbot_nav import JetBotNavEnv

    env = JetBotNavEnv(**env_kwargs)
    obs = []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        while True:
            obs.append(o)
            o, _, term, trunc, _ = env.step(policy(o))
            if term or trunc:
                break
    return np.array(obs[::every], dtype=np.float32)


def session(path: str | Path, threads: int = 1, provider: str = "cpu"):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), so, providers=providers)


class OnnxPolicy:
    """Callable navigation policy backed by ONNX Runtime (what runs on the robot)."""

    def __init__(self, path: str | Path, discrete: bool = False, threads: int = 1):
        self.sess = session(path, threads)
        self.inp = self.sess.get_inputs()[0].name
        self.discrete = discrete

    def __call__(self, obs: np.ndarray):
        out = self.sess.run(None, {self.inp: np.asarray(obs, np.float32)[None]})[0][0]
        return int(np.argmax(out)) if self.discrete else out


class OnnxNLU:
    """Drop-in replacement for edgebot.nlu.engine.NLU using an ONNX graph."""

    def __init__(self, path: str | Path, vocab, threads: int = 1):
        self.sess = session(path, threads)
        self.vocab = vocab

    def predict_tokens(self, token_lists):
        from edgebot.nlu.data import INTENTS, TAGS
        from edgebot.nlu.engine import encode_batch

        w, c, m = encode_batch(token_lists, self.vocab, "cpu")
        probs, slots = self.sess.run(None, {"words": w.numpy(), "chars": c.numpy(), "mask": m.numpy()})
        intents = [INTENTS[i] for i in probs.argmax(-1)]
        tags = [[TAGS[t] for t in slots[k, :len(toks)]] for k, toks in enumerate(token_lists)]
        return intents, tags, probs.max(-1)

    def parse(self, text: str):
        from edgebot.nlu.data import tokenize
        from edgebot.nlu.engine import Frame, tags_to_slots

        toks = tokenize(text)
        if not toks:
            return Frame("unknown", 0.0)
        intents, tags, conf = self.predict_tokens([toks])
        return Frame(intents[0], float(conf[0]), tags_to_slots(toks, tags[0]), toks, tags[0])

    def evaluate(self, data, bs: int = 256):
        from edgebot.nlu.engine import confusion, evaluate_predictions

        pi, pt = [], []
        for i in range(0, len(data), bs):
            a, b, _ = self.predict_tokens([e.tokens for e in data[i:i + bs]])
            pi += a
            pt += b
        m = evaluate_predictions(data, pi, pt)
        m["confusion"] = confusion(data, pi).tolist()
        return m
