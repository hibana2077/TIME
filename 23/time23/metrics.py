from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _ci95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size <= 1:
        return float("nan")
    return 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)


def format_ci95(mean: float, ci95: float) -> str:
    if mean is None or (isinstance(mean, float) and (math.isnan(mean) or math.isinf(mean))):
        return "n/a"
    if ci95 is None or (isinstance(ci95, float) and math.isnan(ci95)):
        return f"{mean:.4f}"
    return f"{mean:.4f}±{ci95:.4f}"


def compute_utility_metrics(df_runs: pd.DataFrame, epsilon: float) -> Dict[str, float]:
    rows = df_runs[df_runs["epsilon"].astype(float) == float(epsilon)]
    acc = rows["acc"].astype(float).to_numpy()
    loss = rows["loss"].astype(float).to_numpy() if "loss" in rows.columns else np.array([np.nan])
    out = {"acc_mean": float(np.nanmean(acc)), "acc_ci95": float(_ci95(acc))}
    if loss.size:
        out.update({"loss_mean": float(np.nanmean(loss)), "loss_ci95": float(_ci95(loss))})
    return out


def _topk_indices(v: np.ndarray, k: int) -> np.ndarray:
    if k >= v.size:
        return np.argsort(-v)
    # partial argpartition then sort
    idx = np.argpartition(-v, kth=k - 1)[:k]
    idx = idx[np.argsort(-v[idx])]
    return idx


def compute_stability_metrics(
    attribs_by_seed: Dict[int, np.ndarray],
    topk: int,
) -> Dict[str, float]:
    """Stability across seeds for a fixed epsilon.

    attribs_by_seed[seed] has shape [Q, M].
    We compute pairwise seed correlations per query, then average.
    """

    seeds = sorted(attribs_by_seed.keys())
    if len(seeds) < 2:
        return {
            "stability_spearman_mean": float("nan"),
            "stability_spearman_ci95": float("nan"),
            "stability_jaccard_mean": float("nan"),
            "stability_jaccard_ci95": float("nan"),
        }

    Q = attribs_by_seed[seeds[0]].shape[0]

    spear_values = []
    jac_values = []

    for q in range(Q):
        # all pairs
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a = attribs_by_seed[seeds[i]][q]
                b = attribs_by_seed[seeds[j]][q]

                rho = spearmanr(a, b).correlation
                spear_values.append(rho)

                ta = set(_topk_indices(a, topk).tolist())
                tb = set(_topk_indices(b, topk).tolist())
                jac = len(ta & tb) / max(1, len(ta | tb))
                jac_values.append(jac)

    spear = np.asarray(spear_values, dtype=np.float64)
    jac = np.asarray(jac_values, dtype=np.float64)

    return {
        "stability_spearman_mean": float(np.nanmean(spear)),
        "stability_spearman_ci95": float(_ci95(spear)),
        "stability_jaccard_mean": float(np.nanmean(jac)),
        "stability_jaccard_ci95": float(_ci95(jac)),
    }


def compute_credibility_agreement(
    dp_attribs: Dict[int, np.ndarray],
    baseline_attribs: Dict[int, np.ndarray],
) -> Dict[str, float]:
    """Agreement with non-private baseline.

    Uses same-seed pairing when possible; otherwise uses all combinations.
    Returns mean Spearman correlation across queries.
    """

    dp_seeds = set(dp_attribs.keys())
    base_seeds = set(baseline_attribs.keys())

    pairs = []
    for s in sorted(dp_seeds & base_seeds):
        pairs.append((s, s))
    if not pairs:
        # fallback: all pairs
        for sd in sorted(dp_seeds):
            for sb in sorted(base_seeds):
                pairs.append((sd, sb))

    Q = next(iter(dp_attribs.values())).shape[0]

    values = []
    for q in range(Q):
        for sd, sb in pairs:
            a = dp_attribs[sd][q]
            b = baseline_attribs[sb][q]
            rho = spearmanr(a, b).correlation
            values.append(rho)

    vals = np.asarray(values, dtype=np.float64)
    return {
        "agreement_spearman_mean": float(np.nanmean(vals)),
        "agreement_spearman_ci95": float(_ci95(vals)),
    }


def compute_counterfactual_proxy_delta(
    attribution_scores: np.ndarray,
    topk: int,
    repeats: int,
    rng_seed: int,
) -> float:
    """Fast proxy for counterfactual removal test.

    attribution_scores: [Q, M] TracIn scores.
    For each query q:
      effect_top = |sum_{i in TopK} score[q,i]|
      effect_rand = mean_r |sum_{i in RandK_r} score[q,i]|
      delta_q = effect_top - effect_rand
    Returns mean(delta_q) across queries.

    This is not full retraining-based removal; it's a lightweight sanity check
    consistent with TracIn's gradient-dot-product interpretation.
    """

    rng = np.random.default_rng(int(rng_seed))
    Q, M = attribution_scores.shape
    k = min(int(topk), int(M))
    r = max(1, int(repeats))

    deltas = []
    for q in range(Q):
        v = attribution_scores[q]
        top_idx = _topk_indices(v, k)
        effect_top = float(np.abs(v[top_idx].sum()))

        rand_effects = []
        for _ in range(r):
            rand_idx = rng.choice(M, size=k, replace=False)
            rand_effects.append(float(np.abs(v[rand_idx].sum())))

        effect_rand = float(np.mean(rand_effects))
        deltas.append(effect_top - effect_rand)

    return float(np.mean(deltas))


def add_trustworthiness_index(
    df_summary: pd.DataFrame,
    baseline_epsilon: float,
    use_counterfactual: bool,
    wp: float = 1.0,
    wt: float = 1.0,
    wu: float = 1.0,
) -> pd.DataFrame:
    """Adds normalized components and TWI column to summary.

    - Utility U: acc normalized by baseline acc
    - Attribution trust T: average(stability, credibility)
      stability uses stability_spearman mapped to [0,1]
      credibility uses agreement_spearman mapped to [0,1], optionally mixed with counterfactual
    - Privacy strength P: based on 1/(1+epsilon), normalized across finite eps.
    """

    df = df_summary.copy()
    if df.empty:
        return df

    base_rows = df[np.isinf(df["epsilon"].astype(float))]
    if base_rows.empty:
        base_acc = float(df["acc_mean"].max())
    else:
        base_acc = float(base_rows.iloc[0]["acc_mean"])
    base_acc = max(base_acc, 1e-12)

    # U in [0,1]
    df["U"] = (df["acc_mean"].astype(float) / base_acc).clip(0.0, 1.0)

    # Stability Spearman -> [0,1]
    stab = df.get("stability_spearman_mean", pd.Series([np.nan] * len(df))).astype(float)
    df["S"] = ((stab + 1.0) / 2.0).clip(0.0, 1.0)

    # Agreement Spearman -> [0,1]
    agree = df.get("agreement_spearman_mean", pd.Series([np.nan] * len(df))).astype(float)
    df["A"] = ((agree + 1.0) / 2.0).clip(0.0, 1.0)

    if use_counterfactual and "counterfactual_delta_mean" in df.columns:
        cf = df["counterfactual_delta_mean"].astype(float)
        finite = cf[~np.isinf(df["epsilon"].astype(float)) & ~np.isnan(cf)]
        if finite.size >= 2:
            lo = float(finite.min())
            hi = float(finite.max())
            if hi - lo > 1e-12:
                df["C"] = ((cf - lo) / (hi - lo)).clip(0.0, 1.0)
            else:
                df["C"] = 0.0
        else:
            df["C"] = np.nan

        df["credibility"] = np.nanmean(np.stack([df["A"].to_numpy(), df["C"].to_numpy()]), axis=0)
    else:
        df["credibility"] = df["A"]

    df["trust"] = np.nanmean(np.stack([df["S"].to_numpy(), df["credibility"].to_numpy()]), axis=0)
    df["trust"] = np.clip(df["trust"], 0.0, 1.0)

    # Privacy strength from epsilon
    eps = df["epsilon"].astype(float)
    P_raw = np.where(np.isinf(eps), 0.0, 1.0 / (1.0 + eps))
    finite_p = P_raw[~np.isinf(eps)]
    if finite_p.size >= 2 and (finite_p.max() - finite_p.min()) > 1e-12:
        P = (P_raw - finite_p.min()) / (finite_p.max() - finite_p.min())
    else:
        P = np.zeros_like(P_raw)
    df["P"] = np.clip(P, 0.0, 1.0)

    # Weighted geometric mean
    def _safe(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 1e-12, 1.0)

    denom = float(wp + wt + wu)
    df["TWI"] = (_safe(df["P"]) ** wp * _safe(df["trust"]) ** wt * _safe(df["U"]) ** wu) ** (1.0 / denom)

    return df
