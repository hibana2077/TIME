import argparse
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from time23.cifar10_data import build_cifar10_dataloaders
from time23.config import ExperimentConfig
from time23.dp_train import train_one_run
from time23.metrics import (
    add_trustworthiness_index,
    compute_counterfactual_proxy_delta,
    compute_credibility_agreement,
    compute_stability_metrics,
    compute_utility_metrics,
    format_ci95,
)
from time23.tracin_last_layer import compute_tracin_attributions_last_layer
from time23.utils import ensure_dir, set_global_seed


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_eps_list(s: str) -> List[Optional[float]]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    eps: List[Optional[float]] = []
    for p in parts:
        if p.lower() in {"inf", "infty", "infinite", "none"}:
            eps.append(None)
        else:
            eps.append(float(p))
    return eps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 + timm ResNet-18 + DP-SGD (Opacus) + TracIn (last-layer) sweep runner"
    )
    parser.add_argument("--out_dir", type=str, default="runs", help="Output root directory")
    parser.add_argument(
        "--epsilons",
        type=str,
        default="inf,10,5,2,1,0.5,0.2",
        help="Comma-separated epsilons; use inf for non-private baseline",
    )
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--train_subset", type=int, default=5000, help="Train subset size for TracIn")
    parser.add_argument("--query_subset", type=int, default=100, help="Test query subset size for TracIn")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument(
        "--tracin_checkpoints",
        type=str,
        default="last5",
        help="Which checkpoints to use: last5 | all | comma-separated epoch indices (e.g., 1,3,5)",
    )

    parser.add_argument(
        "--run_counterfactual",
        action="store_true",
        help="Run counterfactual removal test (can be slow).",
    )
    parser.add_argument("--counterfactual_steps", type=int, default=200)
    parser.add_argument("--counterfactual_repeats", type=int, default=5)

    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing")

    args = parser.parse_args()

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    eps_list = _parse_eps_list(args.epsilons)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_group = f"cifar10_resnet18_tracin_{_timestamp()}"
    group_dir = out_root / run_group
    ensure_dir(group_dir)

    config = ExperimentConfig(
        dataset="cifar10",
        model_name="resnet18",
        num_classes=10,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        delta=args.delta,
        device=args.device,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        query_subset=args.query_subset,
        topk=args.topk,
        tracin_checkpoints=args.tracin_checkpoints,
        run_counterfactual=args.run_counterfactual,
        counterfactual_steps=args.counterfactual_steps,
        counterfactual_repeats=args.counterfactual_repeats,
        download=args.download,
    )

    _save_json(group_dir / "config.json", asdict(config))

    # Shared dataset indices so seeds are comparable
    set_global_seed(12345)
    trainloader_full, testloader_full, train_ds, test_ds = build_cifar10_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        download=config.download,
    )

    train_indices = np.random.permutation(len(train_ds))[: config.train_subset].tolist()
    query_indices = np.random.permutation(len(test_ds))[: config.query_subset].tolist()

    _save_json(
        group_dir / "subset_indices.json",
        {"train_indices": train_indices, "query_indices": query_indices},
    )

    records: List[Dict] = []
    per_run_paths: Dict[str, Dict[str, str]] = {}

    # 1) Train + compute TracIn per (epsilon, seed)
    for eps in eps_list:
        for seed in seeds:
            run_id = f"eps_{'inf' if eps is None else str(eps).replace('.', 'p')}_seed_{seed}"
            run_dir = group_dir / run_id
            ensure_dir(run_dir)

            set_global_seed(seed)

            trainloader, testloader, _, _ = build_cifar10_dataloaders(
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                download=config.download,
            )

            run_result = train_one_run(
                run_dir=run_dir,
                trainloader=trainloader,
                testloader=testloader,
                config=config,
                epsilon=eps,
                seed=seed,
            )

            # TracIn on fixed subsets
            train_subset = Subset(train_ds, train_indices)
            query_subset = Subset(test_ds, query_indices)
            train_subset_loader = DataLoader(
                train_subset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
            )
            query_loader = DataLoader(
                query_subset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
            )

            tracin = compute_tracin_attributions_last_layer(
                checkpoints_dir=run_dir / "checkpoints",
                train_subset_loader=train_subset_loader,
                query_loader=query_loader,
                device=config.device,
                model_name=config.model_name,
                num_classes=config.num_classes,
                which_checkpoints=config.tracin_checkpoints,
            )

            np.savez_compressed(run_dir / "tracin_attributions.npz", scores=tracin)

            per_run_paths[run_id] = {
                "run_dir": str(run_dir),
                "tracin_attributions": str(run_dir / "tracin_attributions.npz"),
                "checkpoints_dir": str(run_dir / "checkpoints"),
            }

            records.append(
                {
                    "run_id": run_id,
                    "epsilon": float("inf") if eps is None else float(eps),
                    "seed": seed,
                    **run_result,
                }
            )

    df_runs = pd.DataFrame.from_records(records)
    df_runs.to_csv(group_dir / "per_run_metrics.csv", index=False)

    # 2) Aggregate stability + credibility vs baseline
    # Load all attribution arrays
    attribs: Dict[Tuple[float, int], np.ndarray] = {}
    for row in df_runs.itertuples(index=False):
        run_dir = group_dir / row.run_id
        data = np.load(run_dir / "tracin_attributions.npz")
        attribs[(float(row.epsilon), int(row.seed))] = data["scores"]

    # Optional: per-run counterfactual proxy metric
    if config.run_counterfactual:
        cf_values = []
        for row in df_runs.itertuples(index=False):
            scores = attribs[(float(row.epsilon), int(row.seed))]
            delta = compute_counterfactual_proxy_delta(
                attribution_scores=scores,
                topk=config.topk,
                repeats=config.counterfactual_repeats,
                rng_seed=int(row.seed) + 1000,
            )
            cf_values.append(delta)

        df_runs["counterfactual_delta"] = cf_values
        df_runs.to_csv(group_dir / "per_run_metrics.csv", index=False)

    eps_values = sorted(set(df_runs["epsilon"].astype(float).tolist()))
    baseline_eps = float("inf")

    summary_rows: List[Dict] = []
    for eps_val in eps_values:
        # Utility metrics from per-run table
        util = compute_utility_metrics(df_runs, eps_val)

        # Stability within eps
        same_eps = {seed: attribs[(eps_val, seed)] for seed in seeds if (eps_val, seed) in attribs}
        stab = compute_stability_metrics(same_eps, topk=config.topk)

        # Credibility agreement vs baseline
        if eps_val != baseline_eps:
            baseline = {seed: attribs[(baseline_eps, seed)] for seed in seeds if (baseline_eps, seed) in attribs}
            cred = compute_credibility_agreement(dp_attribs=same_eps, baseline_attribs=baseline)
        else:
            cred = {"agreement_spearman_mean": np.nan, "agreement_spearman_ci95": np.nan}

        # Counterfactual proxy aggregated across seeds
        if config.run_counterfactual and "counterfactual_delta" in df_runs.columns:
            rows = df_runs[df_runs["epsilon"].astype(float) == float(eps_val)]
            vals = rows["counterfactual_delta"].astype(float).to_numpy()
            cf = {"counterfactual_delta_mean": float(np.nanmean(vals)), "counterfactual_delta_ci95": float(1.96 * np.nanstd(vals, ddof=1) / max(1.0, np.sqrt(np.sum(~np.isnan(vals))))) if np.sum(~np.isnan(vals)) > 1 else float("nan")}
        else:
            cf = {"counterfactual_delta_mean": np.nan, "counterfactual_delta_ci95": np.nan}

        row = {
            "epsilon": eps_val,
            **util,
            **stab,
            **cred,
            **cf,
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # Trustworthiness Index
    df_summary = add_trustworthiness_index(
        df_summary,
        baseline_epsilon=baseline_eps,
        use_counterfactual=bool(config.run_counterfactual),
    )
    df_summary.to_csv(group_dir / "summary.csv", index=False)

    _save_json(
        group_dir / "summary.json",
        {
            "run_group": run_group,
            "config": asdict(config),
            "per_run_paths": per_run_paths,
            "per_run_metrics_csv": str(group_dir / "per_run_metrics.csv"),
            "summary_csv": str(group_dir / "summary.csv"),
            "summary": df_summary.to_dict(orient="records"),
        },
    )

    # Pretty print to console
    print("\nSummary (mean ± 95% CI):")
    if not df_summary.empty:
        for row in df_summary.itertuples(index=False):
            eps_display = "inf" if math.isinf(row.epsilon) else f"{row.epsilon:g}"
            print(
                f"eps={eps_display:>4} | acc={format_ci95(row.acc_mean, row.acc_ci95)} | "
                f"stab_spear={format_ci95(row.stability_spearman_mean, row.stability_spearman_ci95)} | "
                f"stab_jac={format_ci95(row.stability_jaccard_mean, row.stability_jaccard_ci95)} | "
                f"agree={format_ci95(row.agreement_spearman_mean, row.agreement_spearman_ci95)} | "
                f"TWI={getattr(row, 'TWI', float('nan')):.4f}"
            )

    print(f"\nWrote outputs to: {group_dir}")


if __name__ == "__main__":
    main()
