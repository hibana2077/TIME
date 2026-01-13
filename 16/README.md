# MedMNIST Saliency Sanity Study (timm + multi-seed + CSV/plots)

This folder contains a lightweight, paper-oriented empirical study that evaluates whether saliency maps provide reliable *evidence*.

## Main script

- `experiment_sanity_medmnist.py`

## Install

```bash
pip install -r 16/requirements.txt
```

## Run

Run *all* MedMNIST 2D classification datasets with 5 seeds:

```bash
python 16/experiment_sanity_medmnist.py --datasets all --seeds 0,1,2,3,4 --pretrained
```

Run a single dataset (example):

```bash
python 16/experiment_sanity_medmnist.py --datasets pathmnist --seeds 0,1,2
```

## Outputs

A run directory is created under `16/outputs/sanity_<timestamp>/`:

- `tables/per_seed_metrics.csv`: per-dataset per-seed metrics
- `tables/agg_mean_std.csv`: mean ± std across seeds (per dataset)
- `tables/paired_ttests_gradcam_vs_ig.csv`: paired t-tests across seeds
- `tables/deletion_curves_raw.json`: raw deletion curves (for figure regeneration)
- `plots/*.png`: reliability diagrams, robustness curves, deletion curves, stability boxplots, qualitative saliency overlays
- `repro/`: `config.yaml`, `environment.txt`, `pip_freeze.txt`, and optional `git_info.txt`

## What is measured

- **Accuracy** on the clean test split
- **Robustness**: accuracy under Gaussian noise and Gaussian blur (multiple severities)
- **Calibration**: expected calibration error (ECE) + reliability diagram
- **Faithfulness**: deletion AUC (confidence drop when masking top-saliency pixels)
- **Stability**: similarity (SSIM, Spearman) of attributions under small perturbations

## Paper-ready text

See `paper_ready_sanity_section.md` for an English Methods/Results-ready section you can paste and adapt.
