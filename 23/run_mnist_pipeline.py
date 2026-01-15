from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from attribution_ig import integrated_gradients_average
from common import RunSpec, ensure_dir, flatten_feature_names, parse_epsilons, set_global_seed
from datasets import load_mnist_torch
from metrics_trust import (
    compute_credibility,
    compute_stability,
    normalize_utility_to_unit_interval,
    trustworthiness_index,
)
from models import MLP2
from train_dp import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilons", nargs="+", default=["0.1", "0.5", "1.0", "5.0", "inf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ig-batches", type=int, default=10)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default=str(Path("results") / "mnist"))
    args = parser.parse_args()

    epsilons = parse_epsilons(args.epsilons)
    seeds: List[int] = list(args.seeds)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    out_dir = ensure_dir(args.out)
    ensure_dir(Path("plots") / "mnist")

    # Data
    train_loader, test_loader = load_mnist_torch(batch_size=args.batch_size, num_workers=2)

    # Storage (CSV-first)
    accuracy_rows: List[dict] = []
    attrib_rows: List[dict] = []

    d = 28 * 28
    feat_cols = flatten_feature_names("feat_", d)

    # First: run all trainings + attribution extractions
    for epsilon in epsilons:
        for seed in seeds:
            set_global_seed(seed)

            model = MLP2(input_dim=d, hidden_dim=256, num_classes=10)
            tr = train_model(
                model,
                train_loader,
                test_loader,
                device,
                epochs=args.epochs,
                lr=args.lr,
                epsilon=epsilon,
                delta=args.delta,
                max_grad_norm=args.max_grad_norm,
            )

            accuracy_rows.append(
                {
                    "epsilon": float(epsilon) if not math.isinf(epsilon) else float("inf"),
                    "seed": int(seed),
                    "delta": float(args.delta),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "max_grad_norm": float(args.max_grad_norm),
                    "noise_multiplier": tr.noise_multiplier if tr.noise_multiplier is not None else "",
                    "test_accuracy": float(tr.test_accuracy),
                }
            )

            ig = integrated_gradients_average(
                tr.model,
                test_loader,
                device,
                n_batches=args.ig_batches,
                steps=args.ig_steps,
                baseline=0.0,
                target_mode="true",
                use_abs=True,
            )

            row = {
                "epsilon": float(epsilon) if not math.isinf(epsilon) else float("inf"),
                "seed": int(seed),
                "ig_batches": int(args.ig_batches),
                "ig_steps": int(args.ig_steps),
                "n_ig_batches_used": int(ig.n_samples),
            }
            row.update({c: float(v) for c, v in zip(feat_cols, ig.attribution_vector.tolist())})
            attrib_rows.append(row)

    acc_path = out_dir / "accuracy_by_seed.csv"
    att_path = out_dir / "attributions_by_seed.csv"

    pd.DataFrame(accuracy_rows).to_csv(acc_path, index=False)
    pd.DataFrame(attrib_rows).to_csv(att_path, index=False)

    # Second: compute metrics from CSVs (so plotting data is definitely saved)
    acc_df = pd.read_csv(acc_path)
    att_df = pd.read_csv(att_path)

    # Baseline attribution reference: mean across baseline (epsilon=inf) seeds
    baseline_df = att_df[np.isinf(att_df["epsilon"])].copy()
    if baseline_df.empty:
        raise RuntimeError("No baseline (epsilon=inf) runs found; cannot compute credibility")

    baseline_vec = baseline_df[feat_cols].to_numpy(dtype=np.float64).mean(axis=0)

    stability_pairs_rows: List[dict] = []
    stability_summary_rows: List[dict] = []
    credibility_rows: List[dict] = []
    credibility_summary_rows: List[dict] = []
    ti_rows: List[dict] = []

    for epsilon in sorted(acc_df["epsilon"].unique(), key=lambda x: (np.isinf(x), x)):
        # Accuracy summary
        eps_acc = acc_df[acc_df["epsilon"] == epsilon]
        acc_mean = float(eps_acc["test_accuracy"].mean())
        acc_std = float(eps_acc["test_accuracy"].std(ddof=0))

        # Attribution vectors for this epsilon
        eps_att = att_df[att_df["epsilon"] == epsilon]
        attribs_by_seed: Dict[int, np.ndarray] = {}
        for _, r in eps_att.iterrows():
            attribs_by_seed[int(r["seed"])] = r[feat_cols].to_numpy(dtype=np.float64)

        # Stability
        pair_rows, summary = compute_stability(attribs_by_seed)
        for pr in pair_rows:
            stability_pairs_rows.append(
                {
                    "epsilon": float(epsilon),
                    **pr,
                }
            )
        stability_summary_rows.append(
            {
                "epsilon": float(epsilon),
                "stability_mean": float(summary.mean_rho),
                "stability_std": float(summary.std_rho),
            }
        )

        # Credibility per seed (vs baseline mean)
        cred_vals: List[float] = []
        for seed, vec in attribs_by_seed.items():
            c = compute_credibility(vec, baseline_vec)
            credibility_rows.append({"epsilon": float(epsilon), "seed": int(seed), "credibility_cosine": float(c)})
            cred_vals.append(float(c))
        cred_mean = float(np.mean(cred_vals)) if cred_vals else 0.0
        cred_std = float(np.std(cred_vals, ddof=0)) if cred_vals else 0.0
        credibility_summary_rows.append(
            {"epsilon": float(epsilon), "credibility_mean": cred_mean, "credibility_std": cred_std}
        )

        # Trustworthiness Index (use mean values)
        u_norm = normalize_utility_to_unit_interval(acc_mean)
        s = float(summary.mean_rho)
        c = cred_mean
        t = trustworthiness_index(u_norm, s, c)
        ti_rows.append(
            {
                "epsilon": float(epsilon),
                "utility_mean": acc_mean,
                "utility_std": acc_std,
                "utility_norm": u_norm,
                "stability_mean": s,
                "credibility_mean": c,
                "trustworthiness_index": t,
            }
        )

    pd.DataFrame(stability_pairs_rows).to_csv(out_dir / "stability_pairs.csv", index=False)
    pd.DataFrame(stability_summary_rows).to_csv(out_dir / "stability_summary.csv", index=False)
    pd.DataFrame(credibility_rows).to_csv(out_dir / "credibility_by_seed.csv", index=False)
    pd.DataFrame(credibility_summary_rows).to_csv(out_dir / "credibility_summary.csv", index=False)
    pd.DataFrame(ti_rows).to_csv(out_dir / "trustworthiness_index.csv", index=False)

    # Third: plot from saved CSVs
    from plot_results import plot_mnist_from_csv

    plot_mnist_from_csv(results_dir=out_dir, plots_dir=Path("plots") / "mnist")


if __name__ == "__main__":
    main()
