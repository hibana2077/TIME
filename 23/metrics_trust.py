from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np


def rankdata_desc(x: np.ndarray) -> np.ndarray:
    """Ranks features by descending score.

    Returns ranks in [0..D-1] where 0 = highest score.
    Ties are handled by stable sort order.
    """
    order = np.argsort(-x, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(x.shape[0])
    return ranks


def spearman_rho_from_ranks(r1: np.ndarray, r2: np.ndarray) -> float:
    """Spearman correlation computed as Pearson correlation on ranks."""
    r1 = r1.astype(np.float64)
    r2 = r2.astype(np.float64)
    r1 = r1 - r1.mean()
    r2 = r2 - r2.mean()
    denom = (np.linalg.norm(r1) * np.linalg.norm(r2))
    if denom == 0:
        return 0.0
    return float(np.dot(r1, r2) / denom)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class StabilitySummary:
    epsilon: float
    mean_rho: float
    std_rho: float


def compute_stability(attribs_by_seed: Dict[int, np.ndarray]) -> Tuple[List[dict], StabilitySummary]:
    """Pairwise Spearman on attribution rankings across seeds."""
    seeds = sorted(attribs_by_seed.keys())
    rows: List[dict] = []
    rhos: List[float] = []

    ranks = {s: rankdata_desc(attribs_by_seed[s]) for s in seeds}

    for si, sj in combinations(seeds, 2):
        rho = spearman_rho_from_ranks(ranks[si], ranks[sj])
        rows.append({"seed_i": si, "seed_j": sj, "spearman_rho": rho})
        rhos.append(rho)

    mean_rho = float(np.mean(rhos)) if rhos else 0.0
    std_rho = float(np.std(rhos, ddof=0)) if rhos else 0.0

    # epsilon will be filled by caller
    return rows, StabilitySummary(epsilon=float("nan"), mean_rho=mean_rho, std_rho=std_rho)


def compute_credibility(attrib: np.ndarray, baseline_attrib: np.ndarray) -> float:
    return cosine_similarity(attrib, baseline_attrib)


def normalize_utility_to_unit_interval(u: float, u_min: float = 0.0, u_max: float = 1.0) -> float:
    # Accuracy is already in [0,1] but keep the function for clarity/extensibility.
    if u_max <= u_min:
        return float(u)
    return float(np.clip((u - u_min) / (u_max - u_min), 0.0, 1.0))


def trustworthiness_index(u: float, s: float, c: float, alpha: float = 1 / 3, beta: float = 1 / 3, gamma: float = 1 / 3) -> float:
    return float(alpha * u + beta * s + gamma * c)
