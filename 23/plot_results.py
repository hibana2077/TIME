from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_mnist_from_csv(results_dir: Path, plots_dir: Path) -> None:
    """Generates Figure 1/2/3 from CSVs under results_dir."""
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ti = pd.read_csv(results_dir / "trustworthiness_index.csv")

    # For log-scale plotting: remove inf for x-axis, but keep as a separate marker if desired.
    finite = ti[np.isfinite(ti["epsilon"])].copy()

    # Figure 1: Accuracy / Stability / Credibility vs epsilon (log scale)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(finite["epsilon"], finite["utility_mean"], marker="o", label="Accuracy (Utility)")
    ax.plot(finite["epsilon"], finite["stability_mean"], marker="o", label="Stability")
    ax.plot(finite["epsilon"], finite["credibility_mean"], marker="o", label="Credibility")
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy budget $\epsilon$ (log scale)")
    ax.set_ylabel("Metric value")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "figure1_privacy_transparency_utility.png", dpi=200)
    plt.close(fig)

    # Figure 2: Trustworthiness Index vs epsilon
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(finite["epsilon"], finite["trustworthiness_index"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy budget $\epsilon$ (log scale)")
    ax.set_ylabel(r"Trustworthiness Index $T(\epsilon)$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(plots_dir / "figure2_trustworthiness_index.png", dpi=200)
    plt.close(fig)

    # Figure 3: Utility–Stability trade-off (colored by epsilon)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sc = ax.scatter(
        finite["utility_mean"],
        finite["stability_mean"],
        c=np.log10(finite["epsilon"].to_numpy()),
        cmap="viridis",
        s=60,
    )
    ax.set_xlabel("Accuracy (Utility)")
    ax.set_ylabel("Stability")
    ax.grid(True, linestyle=":", linewidth=0.7)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log_{10}(\epsilon)$")
    fig.tight_layout()
    fig.savefig(plots_dir / "figure3_utility_stability_tradeoff.png", dpi=200)
    plt.close(fig)
