"""Run EdgeBot end to end.

Scripted demo (the default) - a multi-turn warehouse conversation that exercises
navigation, pick-n-place, clarification dialogue, assembly and status queries,
rendered to an MP4 with spoken responses:

    python scripts/run_demo.py --voice --video results/media/edgebot_demo.mp4

Other modes:

    python scripts/run_demo.py --text "move the red box from shelf a to the conveyor"
    python scripts/run_demo.py --wav path/to/command.wav
    python scripts/run_demo.py --interactive            # type commands
    python scripts/run_demo.py --mic                    # speak commands (needs sounddevice)

--voice turns every scripted command into speech with the TTS voice and feeds the
audio through ASR, so the full speech pipeline runs without a microphone.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from edgebot.robot.agent import EdgeBot, FrameSink  # noqa: E402
from edgebot.utils import CHECKPOINTS, RESULTS, save_json  # noqa: E402

SCRIPT = [
    "hello jetbot",
    "move the red box from shelf a to the conveyor",
    "pick up the box",
    "the blue one",
    "put it on the packing station",
    "stack the red green and blue blocks at the workbench",
    "what is your status",
    "go back to the charging dock",
]


def load_stack(use_onnx: bool = True):
    from edgebot.nlu.engine import NLU

    nlu_torch = NLU.load(str(CHECKPOINTS / "nlu_jointbert_mini.pt"))
    onnx_dir = CHECKPOINTS / "onnx"
    if use_onnx and (onnx_dir / "nav_ppo_actor.onnx").exists() and (onnx_dir / "nlu_jointbert_mini.onnx").exists():
        from edgebot.edge.export import OnnxNLU, OnnxPolicy

        return OnnxNLU(onnx_dir / "nlu_jointbert_mini.onnx", nlu_torch.vocab), \
            OnnxPolicy(onnx_dir / "nav_ppo_actor.onnx"), "onnxruntime"
    from edgebot.envs.jetbot_nav import JetBotNavEnv
    from edgebot.rl.ppo import PPO

    obs_dim = JetBotNavEnv().observation_space.shape[0]
    ppo = PPO.load(str(CHECKPOINTS / "nav_ppo_continuous.pt"), obs_dim, 2)
    return nlu_torch, (lambda o: ppo.predict(o)), "pytorch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="*", help="one or more text commands")
    ap.add_argument("--wav", nargs="*", help="one or more 16 kHz+ WAV files with spoken commands")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--voice", action="store_true", help="synthesise scripted commands and run them through ASR")
    ap.add_argument("--speak", action="store_true", help="synthesise the robot's replies to WAV")
    ap.add_argument("--video", default=None, help="write an MP4/GIF of the run")
    ap.add_argument("--torch", action="store_true", help="use PyTorch models instead of ONNX Runtime")
    ap.add_argument("--asr-model", default="base")
    args = ap.parse_args()

    nlu, policy, runtime = load_stack(not args.torch)
    needs_asr = args.voice or args.wav or args.mic
    asr = tts = None
    if needs_asr:
        from edgebot.speech.asr import ASR

        asr = ASR(args.asr_model)
    if args.voice or args.speak or args.mic:
        from edgebot.speech.tts import TTS

        tts = TTS()
    sink = FrameSink(enabled=args.video is not None)
    bot = EdgeBot(nlu, policy, asr, tts, sink)
    out_dir = RESULTS / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"EdgeBot ready (inference runtime: {runtime}, ASR: {'whisper-' + args.asr_model if asr else 'off'})\n")

    def show(turn, k):
        heard = f"   (ASR heard: \"{turn.asr_text}\")" if turn.asr_text is not None else ""
        print(f"User   : {turn.user}{heard}")
        print(f"NLU    : {turn.frame}")
        print(f"EdgeBot: {turn.response}\n")
        if args.speak or args.voice:
            bot.speak(turn.response, out_dir / f"reply_{k:02d}.wav")

    k = 0
    if args.wav:
        for path in args.wav:
            show(bot.hear(path), k)
            k += 1
    elif args.mic:
        from edgebot.speech.asr import record_microphone
        from edgebot.speech.tts import play

        print("Press Enter to speak a command (Ctrl+C to quit).")
        while True:
            try:
                input()
            except (KeyboardInterrupt, EOFError):
                break
            turn = bot.hear(record_microphone(4.0))
            show(turn, k)
            audio, _ = tts.synthesize(turn.response)
            play(audio, tts.sample_rate)
            k += 1
    elif args.interactive:
        print("Type a command (empty line to quit).")
        while True:
            try:
                text = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not text:
                break
            show(bot.handle(text), k)
            k += 1
    else:
        commands = args.text or SCRIPT
        for text in commands:
            if args.voice:
                from scipy.signal import resample_poly

                wav, _ = tts.synthesize(text)
                from edgebot.speech.tts import write_wav

                write_wav(out_dir / f"user_{k:02d}.wav", wav, tts.sample_rate)
                audio16 = resample_poly(wav, 160, tts.sample_rate // 100).astype(np.float32)
                turn = bot.hear(audio16)
                turn.user = text
            else:
                turn = bot.handle(text)
            show(turn, k)
            k += 1

    summary = {"runtime": runtime, "stats": bot.stats, "distance_m": bot.distance_travelled,
               "battery": bot.battery, "inventory": bot.inventory,
               "turns": [t.__dict__ for t in bot.history]}
    save_json(summary, out_dir / "transcript.json")
    print("Stats:", bot.stats)
    if args.video and sink.frames:
        import imageio.v2 as imageio

        path = Path(args.video)
        path.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        if path.suffix == ".gif":
            imageio.mimsave(path, sink.frames[::3], duration=0.1, loop=0)
        else:
            with imageio.get_writer(path, fps=20, codec="libx264", quality=7, macro_block_size=4) as w:
                for f in sink.frames:
                    w.append_data(f)
        print(f"Video: {path} ({len(sink.frames)} frames, {time.time() - t:.0f}s to encode)")


if __name__ == "__main__":
    main()
