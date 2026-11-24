"""Automatic Speech Recognition with Whisper (CTranslate2 backend via faster-whisper).

Whisper is an encoder-decoder Transformer trained on 680k hours of audio. The
CTranslate2 runtime supports INT8 / FP16 inference, which is how speech models are
deployed on edge GPUs such as the Jetson family. A domain prompt biases decoding
towards the warehouse vocabulary (location names, colours, objects).
"""

from __future__ import annotations

import time

import numpy as np

DOMAIN_PROMPT = ("JetBot warehouse robot commands. Go to shelf A, shelf B, shelf C, the conveyor, the packing "
                 "station, the charging dock, the loading bay or the workbench. Pick up the red, green, blue, "
                 "yellow, orange or purple box, block, crate, parcel or cube. Stack the blocks.")


def load_audio(path: str, sr: int = 16_000) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(int(file_sr), sr)
        audio = resample_poly(audio, sr // g, int(file_sr) // g).astype(np.float32)
    return audio


def _preload_cuda_libs() -> None:
    """Loads cuBLAS / cuDNN shipped as pip wheels (nvidia-*-cu12) so CTranslate2 can find them."""
    import ctypes
    import glob
    import importlib.util

    for pkg, patterns in (("nvidia.cublas", ("libcublasLt.so.*", "libcublas.so.*")),
                          ("nvidia.cudnn", ("libcudnn*.so.*",))):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        for root in spec.submodule_search_locations:
            for pat in patterns:
                for lib in sorted(glob.glob(f"{root}/lib/{pat}")):
                    try:
                        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass


class ASR:
    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str | None = None,
                 domain_prompt: bool = True):
        from faster_whisper import WhisperModel

        if device in ("auto", "cuda"):
            try:
                import ctranslate2

                _preload_cuda_libs()
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        self.prompt = DOMAIN_PROMPT if domain_prompt else None
        self.model_size = model_size
        self._load(WhisperModel, device, compute_type)
        if self.device == "cuda":
            try:  # verify the GPU kernels actually load; otherwise fall back to CPU INT8
                list(self.model.transcribe(np.zeros(16_000, np.float32), language="en")[0])
            except RuntimeError:
                self._load(WhisperModel, "cpu", None)

    def _load(self, cls, device, compute_type):
        self.device = device
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.model = cls(self.model_size, device=device, compute_type=self.compute_type)

    def transcribe(self, audio, sr: int = 16_000) -> tuple[str, float]:
        """audio: path or float32 mono array at 16 kHz. Returns (text, seconds)."""
        if isinstance(audio, str):
            audio = load_audio(audio, sr)
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(audio, language="en", beam_size=5, initial_prompt=self.prompt,
                                            vad_filter=False, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, time.perf_counter() - t0


def word_error_rate(ref: str, hyp: str) -> float:
    """Levenshtein distance over words divided by reference length."""
    from edgebot.nlu.data import tokenize

    r, h = tokenize(ref), tokenize(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(len(r), 1)


def record_microphone(seconds: float = 4.0, sr: int = 16_000) -> np.ndarray:
    """Records from the default microphone (requires the optional ``sounddevice`` package)."""
    import sounddevice as sd

    print(f"Listening for {seconds:.0f} s ...", flush=True)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]
