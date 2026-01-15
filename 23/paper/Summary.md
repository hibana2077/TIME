
# Summary: Experimental Setup, Parameters, and Results (CSV-based)

This document summarizes the experimental instantiation described in `inst.md` and implemented in this folder’s runnable pipeline (see `README.md`). It provides (i) a concrete, reproducible experimental setup with explicit parameters, and (ii) an English report of the empirical findings computed from the CSV artifacts saved under `results/`.

## 1. Experimental Goal

We study how Differential Privacy (DP) training (via DP-SGD) affects:

1) **Utility** (test accuracy)
2) **Transparency/attribution behavior** via feature-level Integrated Gradients (IG)
3) **Attribution trust metrics** (Stability, Credibility)
4) A single aggregated **Trustworthiness Index** that is explicitly computable from saved metrics

The overall objective is not to “solve” DP attribution, but to make the privacy–utility–transparency trade-off measurable, auditable, and decision-relevant.

## 2. Experimental Setup (Reproducible Specification)

### 2.1 Dataset

- **Dataset:** MNIST (10-class classification)
- **Input representation:** 28×28 grayscale image, flattened to a 784-dimensional vector for the model
- **Preprocessing:** `ToTensor()` only (no normalization beyond [0,1] scaling performed by `ToTensor()`)

### 2.2 Model

- **Architecture:** 2-layer MLP (`MLP2`)
  - Input: 784
  - Hidden: 256
  - Output: 10 logits
  - Activation: ReLU

### 2.3 Training (Non-DP baseline + DP-SGD)

The pipeline trains one model per `(epsilon, seed)`.

- **Privacy mechanism:** DP-SGD via Opacus `PrivacyEngine.make_private_with_epsilon`
- **Delta (fixed):** δ = 1e−5
- **Epsilon sweep:** ε ∈ {0.1, 0.5, 1.0, 5.0, ∞}
  - ε = ∞ corresponds to the **non-DP baseline**
- **Random seeds:** 0, 1, 2, 3, 4 (R = 5)

**Optimization and hyperparameters (as recorded in `results/accuracy_by_seed.csv`):**

- Epochs: 5
- Batch size: 256
- Learning rate: 0.2
- Optimizer: SGD with momentum = 0.9
- Loss: Cross-entropy
- Gradient clipping (DP): max_grad_norm = 1.0

For DP runs, Opacus chooses the **noise multiplier** to meet the target privacy budget (ε, δ) for the given number of epochs.

### 2.4 Feature Attribution (Integrated Gradients)

After each model is trained, the pipeline computes a single **feature attribution vector** using Integrated Gradients (IG), averaged across test batches.

- **Method:** Integrated Gradients
- **Attribution target:** true label (per-sample)
- **Baseline input:** all zeros
- **Absolute attributions:** enabled (uses absolute IG magnitude)
- **IG steps:** 32
- **Number of test batches used:** 10 (average attribution across these batches)
- **Output:** a single vector a ∈ R^784 per `(epsilon, seed)`

## 3. Metrics (Instantiation)

All metrics are computed from CSV artifacts, ensuring that the reported results are reproducible and can be recomputed without re-running training.

### 3.1 Utility

- **Utility U(ε):** test accuracy (mean across seeds)
- Accuracy is already in [0,1], so utility normalization is identity in this instantiation.

### 3.2 Attribution Stability (across seeds)

For each ε, let a^(seed,ε) be the IG attribution vector for a trained model. Define each vector’s feature ranking by descending attribution score. Stability is the **average pairwise Spearman correlation** over all seed pairs:

\[
S(\epsilon) = \frac{2}{R(R-1)}\sum_{i<j}\rho\Big(\mathrm{rank}(\mathbf{a}^{(i,\epsilon)}),\,\mathrm{rank}(\mathbf{a}^{(j,\epsilon)})\Big).
\]

### 3.3 Attribution Credibility (vs non-DP baseline)

Let the baseline reference attribution vector be the **mean attribution over baseline seeds** at ε = ∞:

\[
\mathbf{a}^{(\infty)} = \frac{1}{R}\sum_{r=1}^R \mathbf{a}^{(r,\infty)}.
\]

Credibility is the cosine similarity between a DP attribution vector and the baseline reference:

\[
C(\epsilon) = \cos\Big(\mathbf{a}^{(\epsilon)},\, \mathbf{a}^{(\infty)}\Big).
\]

In the implementation, credibility is computed per seed and then aggregated as mean ± std for reporting.

### 3.4 Trustworthiness Index (aggregate)

The Trustworthiness Index aggregates utility, stability, and credibility using equal weights:

\[
T(\epsilon)=\tfrac{1}{3}U(\epsilon)+\tfrac{1}{3}S(\epsilon)+\tfrac{1}{3}C(\epsilon).
\]

This aggregation is meant to be **auditable and computable** (not a claim of universal optimality). Other weightings can be substituted depending on application needs.

## 4. Artifacts (What was saved)

The pipeline is CSV-first: all plotting data and intermediate metrics are saved.

- `results/accuracy_by_seed.csv`: one row per `(epsilon, seed)` with test accuracy and DP noise multiplier
- `results/attributions_by_seed.csv`: one row per `(epsilon, seed)` with the flattened 784-d IG vector
- `results/stability_pairs.csv`: Spearman ρ for every seed pair at each ε
- `results/stability_summary.csv`: stability mean ± std per ε
- `results/credibility_by_seed.csv`: cosine similarity per `(epsilon, seed)`
- `results/credibility_summary.csv`: credibility mean ± std per ε
- `results/trustworthiness_index.csv`: per-ε summary + Trustworthiness Index

## 5. Results Report (computed from CSVs)

### 5.1 Run coverage

- Epsilons: 5 values (0.1, 0.5, 1, 5, inf)
- Seeds: 5 values (0–4)
- Total trained models / attribution vectors: 5 × 5 = 25

### 5.2 DP noise levels (Opacus-chosen noise multipliers)

Lower ε requires stronger privacy and therefore larger injected noise.

| epsilon | noise multiplier (mean ± std) |
| --- | --- |
| 0.1 | 5.3125 ± 0.0000 |
| 0.5 | 1.3086 ± 0.0000 |
| 1 | 0.9082 ± 0.0000 |
| 5 | 0.5621 ± 0.0000 |

### 5.3 Main quantitative summary (Utility / Stability / Credibility / Trustworthiness)

The table below is taken from `results/trustworthiness_index.csv` (utility is reported as mean ± std across seeds; stability/credibility are computed from attribution vectors).

| epsilon | utility_mean | utility_std | stability_mean | credibility_mean | trustworthiness_index |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.7159 | 0.0059 | 0.9856 | 0.9512 | 0.8842 |
| 0.5 | 0.8807 | 0.0052 | 0.9838 | 0.9528 | 0.9391 |
| 1 | 0.9082 | 0.0033 | 0.9840 | 0.9559 | 0.9494 |
| 5 | 0.9270 | 0.0015 | 0.9853 | 0.9621 | 0.9582 |
| inf | 0.9779 | 0.0017 | 0.9973 | 0.9966 | 0.9906 |

For completeness, the reported variability for attribution trust metrics from `results/stability_summary.csv` and `results/credibility_summary.csv` is:

| epsilon | stability (mean ± std) | credibility (mean ± std) |
| --- | --- | --- |
| 0.1 | 0.9856 ± 0.0008 | 0.9512 ± 0.0038 |
| 0.5 | 0.9838 ± 0.0009 | 0.9528 ± 0.0028 |
| 1 | 0.9840 ± 0.0011 | 0.9559 ± 0.0025 |
| 5 | 0.9853 ± 0.0010 | 0.9621 ± 0.0033 |
| inf | 0.9973 ± 0.0003 | 0.9966 ± 0.0004 |

### 5.4 Key observations

1) **Utility improves monotonically with ε.**
   - The strongest privacy setting ε = 0.1 reduces accuracy to ~0.716.
   - As ε increases, accuracy recovers; the non-DP baseline reaches ~0.978.

2) **Attribution stability is high across all ε, but highest at ε = ∞.**
   - Even at ε = 0.1, stability remains ~0.986, suggesting that the *ranking of features by IG* is relatively consistent across random seeds.
   - The baseline has near-perfect stability (~0.997).

3) **Credibility against baseline increases with ε.**
   - Credibility is lowest under strong privacy (ε = 0.1: ~0.951) and steadily increases as ε relaxes.
   - The baseline is nearly perfectly self-consistent (~0.997).

4) **Trustworthiness Index provides a computable decision signal.**
   - Among finite ε values, the best Trustworthiness Index occurs at **ε = 5** (T ≈ 0.9582).
   - The baseline (ε = ∞) yields the best overall index (T ≈ 0.9906) but provides no DP protection.
   - Therefore, ε = 5 is a reasonable “high trust” DP operating point in this run, balancing accuracy, stability, and credibility.

## 6. Notes and Limitations

- This instantiation uses **a single attribution method** (IG) and a **single dataset/model** (MNIST + MLP). Results should not be over-generalized without additional datasets and architectures.
- Stability is computed on **rankings** (Spearman), which is robust to monotonic transformations but may hide changes in absolute attribution magnitudes.
- Credibility is defined relative to the **non-DP baseline mean attribution**; if the baseline itself is unstable for a different dataset/model, credibility should be interpreted carefully.

## 7. Reproducibility checklist

- Pipeline entry point: `run_mnist_pipeline.py`
- Default sweep (unless overridden by CLI): ε ∈ {0.1, 0.5, 1, 5, inf}, seeds {0–4}
- Key hyperparameters: epochs=5, batch_size=256, lr=0.2, δ=1e−5, max_grad_norm=1.0
- IG parameters: ig_batches=10, ig_steps=32, baseline=0, target=true, abs=true

