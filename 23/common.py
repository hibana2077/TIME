from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_inf_epsilon(epsilon: float) -> bool:
    return math.isinf(epsilon) or str(epsilon).lower() in {"inf", "infty", "infinity"}


def parse_epsilons(values: Sequence[str]) -> List[float]:
    eps: List[float] = []
    for v in values:
        if str(v).lower() in {"inf", "infty", "infinity"}:
            eps.append(float("inf"))
        else:
            eps.append(float(v))
    return eps


def write_csv(path: str | Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model: str
    epsilon: float
    delta: float
    seed: int


def flatten_feature_names(prefix: str, d: int) -> List[str]:
    return [f"{prefix}{i}" for i in range(d)]
