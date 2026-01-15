# TIME/23 Experiments (MNIST + DP + Attribution Trust)

This folder contains runnable scripts that instantiate the experimental illustration described in `inst.md`:

- Train non-DP baseline and DP-SGD models for an epsilon sweep
- Compute Integrated Gradients feature attributions
- Compute attribution trust metrics:
  - Stability: average pairwise Spearman correlation across seeds
  - Credibility: cosine similarity against the non-DP baseline attribution
- Aggregate into a Trustworthiness Index
- Save **all plotting data to CSV** and generate plots from CSV

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run MNIST end-to-end

This will:
1) train models (baseline + DP)
2) compute attribution vectors
3) write CSVs under `results/`
4) compute metrics + Trustworthiness Index
5) write plots under `plots/`

```bash
python run_mnist_pipeline.py --device cpu
```

You can customize epsilons/seeds/epochs:

```bash
python run_mnist_pipeline.py --epsilons 0.1 0.5 1.0 5.0 inf --seeds 0 1 2 3 4 --epochs 5 --device cuda
```

## Outputs

- `results/mnist/accuracy_by_seed.csv`
- `results/mnist/attributions_by_seed.csv` (flattened feature vector per run)
- `results/mnist/stability_pairs.csv`
- `results/mnist/stability_summary.csv`
- `results/mnist/credibility_by_seed.csv`
- `results/mnist/credibility_summary.csv`
- `results/mnist/trustworthiness_index.csv`

Plots (generated **from CSV**):
- `plots/mnist/figure1_privacy_transparency_utility.png`
- `plots/mnist/figure2_trustworthiness_index.png`
- `plots/mnist/figure3_utility_stability_tradeoff.png`

## Tabular datasets (sklearn)

The code includes a tabular loader (e.g. `breast_cancer`, `iris`, `wine`) for later extension.
MNIST is the default pipeline in this folder.
