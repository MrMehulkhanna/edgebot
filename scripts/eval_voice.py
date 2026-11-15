"""End-to-end speech evaluation: TTS -> (noise) -> ASR -> NLU.

Held-out test commands (sentence templates never seen by the NLU) are spoken by
the Piper TTS voice, optionally corrupted with additive white noise at a given
SNR, transcribed by Whisper and parsed by the deployed NLU. Reported: word error
rate, intent accuracy, slot-value F1 and full-frame accuracy, plus real-time
factors of TTS and ASR on CPU (INT8) and GPU (FP16).

    python scripts/eval_voice.py --n 150
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from edgebot.nlu.data import build_splits  # noqa: E402
from edgebot.nlu.engine import NLU, tags_to_slots  # noqa: E402
from edgebot.speech.asr import ASR, word_error_rate  # noqa: E402
from edgebot.speech.tts import TTS, write_wav  # noqa: E402
from edgebot.utils import CHECKPOINTS, RESULTS, save_json  # noqa: E402


def slot_pairs(slots: dict) -> set:
    out = set()
    for k, v in slots.items():
        if isinstance(v, list):
            out |= {(k, i, x) for i, x in enumerate(v)}
        else:
            out.add((k, v))
    return out


def add_noise(x: np.ndarray, snr_db: float, rng) -> np.ndarray:
    p = np.mean(x ** 2)
    noise = rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), x.shape)
    return (x + noise).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model", default="base")
    ap.add_argument("--rtf-n", type=int, default=25)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    tts = TTS()
    nlu = NLU.load(str(CHECKPOINTS / "nlu_jointbert_mini.pt"))
    asr_gpu = ASR(args.model, device="auto")
    test = build_splits()["test"]
    idx = rng.choice(len(test), size=min(args.n, len(test)), replace=False)
    samples = [test[i] for i in idx]

    audio_dir = RESULTS / "audio"
    clips, tts_rtf = [], []
    for k, e in enumerate(samples):
        wav, t = tts.synthesize(e.text)
        tts_rtf.append(t / (len(wav) / tts.sample_rate))
        clips.append(resample_poly(wav, 160, tts.sample_rate // 100).astype(np.float32))
        if k < 6:
            write_wav(audio_dir / f"command_{k}.wav", wav, tts.sample_rate)

    report = {"n": len(samples), "asr_model": f"whisper-{args.model}", "asr_device": asr_gpu.device,
              "asr_compute_type": asr_gpu.compute_type, "tts_rtf_cpu": float(np.mean(tts_rtf)), "conditions": {}}
    examples = []
    for cond in ("clean", "snr20", "snr10"):
        wers, intent_ok, frame_ok = [], [], []
        tp = fp = fn = 0
        for k, (e, clip) in enumerate(zip(samples, clips)):
            if cond != "clean":
                clip = add_noise(clip, float(cond[3:]), rng)
            hyp, _ = asr_gpu.transcribe(clip)
            wers.append(word_error_rate(e.text, hyp))
            f = nlu.parse(hyp)
            gold = slot_pairs(tags_to_slots(e.tokens, e.tags))
            pred = slot_pairs(f.slots)
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
            intent_ok.append(f.intent == e.intent)
            frame_ok.append(f.intent == e.intent and gold == pred)
            if cond == "clean" and k < 12:
                examples.append({"reference": e.text, "asr": hyp, "intent": f.intent, "slots": f.slots,
                                 "correct": bool(frame_ok[-1])})
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        report["conditions"][cond] = {
            "wer": float(np.mean(wers)), "intent_acc": float(np.mean(intent_ok)),
            "slot_f1": 2 * prec * rec / max(prec + rec, 1e-9), "frame_acc": float(np.mean(frame_ok)),
        }
        print(cond, report["conditions"][cond], flush=True)
    # text-only upper bound on the same samples (no ASR)
    text_frames = [nlu.parse(e.text) for e in samples]
    report["text_only_frame_acc"] = float(np.mean([
        f.intent == e.intent and slot_pairs(f.slots) == slot_pairs(tags_to_slots(e.tokens, e.tags))
        for f, e in zip(text_frames, samples)]))
    report["examples"] = examples

    # real-time factor: processing time / audio duration
    sub = clips[:args.rtf_n]
    dur = sum(len(c) for c in sub) / 16_000
    rtf = {}
    for label, asr in ((f"{asr_gpu.device}_{asr_gpu.compute_type}", asr_gpu),
                       ("cpu_int8", ASR(args.model, device="cpu", compute_type="int8"))):
        asr.transcribe(sub[0])
        total = sum(asr.transcribe(c)[1] for c in sub)
        rtf[label] = {"rtf": total / dur, "mean_latency_s": total / len(sub)}
    report["asr_rtf"] = rtf
    report["mean_utterance_s"] = dur / len(sub)
    print(rtf)
    save_json(report, RESULTS / "voice_eval.json")


if __name__ == "__main__":
    main()
