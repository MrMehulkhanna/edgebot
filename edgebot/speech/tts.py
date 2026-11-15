"""Text-to-Speech with Piper (VITS acoustic model + vocoder, exported to ONNX).

Piper is designed for embedded devices (Raspberry Pi, Jetson) and runs through
ONNX Runtime, the same runtime used for the robot's policy and NLU models. The
voice model is downloaded once from the Hugging Face Hub and cached locally.
"""

from __future__ import annotations

import io
import time
import wave
from pathlib import Path

import numpy as np

VOICE_REPO = "rhasspy/piper-voices"
VOICE = "en/en_US/lessac/medium/en_US-lessac-medium"


def voice_paths(voice: str = VOICE) -> tuple[str, str]:
    from huggingface_hub import hf_hub_download

    onnx = hf_hub_download(VOICE_REPO, f"{voice}.onnx")
    cfg = hf_hub_download(VOICE_REPO, f"{voice}.onnx.json")
    return onnx, cfg


class TTS:
    def __init__(self, voice: str = VOICE, use_cuda: bool = False):
        from piper import PiperVoice

        onnx, cfg = voice_paths(voice)
        self.voice = PiperVoice.load(onnx, config_path=cfg, use_cuda=use_cuda)
        self.sample_rate = self.voice.config.sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        """Returns (float32 waveform, seconds taken)."""
        t0 = time.perf_counter()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize_wav(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0, time.perf_counter() - t0

    def save(self, text: str, path: str | Path) -> float:
        audio, _ = self.synthesize(text)
        write_wav(path, audio, self.sample_rate)
        return len(audio) / self.sample_rate


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def play(audio: np.ndarray, sr: int) -> None:
    """Plays audio on the default output device (requires ``sounddevice``)."""
    import sounddevice as sd

    sd.play(audio, sr)
    sd.wait()
