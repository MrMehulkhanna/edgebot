"""Small shared helpers: seeding, device selection, JSON I/O and paths."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
FIGURES = RESULTS / "figures"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "auto"):
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def moving_average(x, window: int):
    x = np.asarray(x, dtype=float)
    if len(x) < window or window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def ensure_dirs() -> None:
    for d in (RESULTS, CHECKPOINTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
