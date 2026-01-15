# A Minimal Empirical Study on Faithfulness under Shortcut Learning

Date: 2026-01-15

## Goal
This small study tests whether post-hoc explanations remain *faithful* when a classifier is exposed to a spurious, shortcut feature (a “logo/watermark”) that is correlated with the label during training.

## Experimental Setup
**Dataset (synthetic, medical-like):** 64×64 grayscale images with Gaussian noise.
- **Causal feature:** a bright circular “lesion” (ground-truth lesion mask available). Label is defined by lesion presence.
- **Spurious feature (shortcut):** a bright square “logo” in the bottom-right corner.

**Splits:**
- **ID (in-distribution):** logo is present with high correlation to the label (shortcut available).
- **OOD (out-of-distribution):** logo is randomized or removed (shortcut broken).

**Models:** a small 3-layer CNN (`SmallCNN`).
- **Baseline:** trained on the correlated (shortcut) training set.
- **Mitigated:** same CNN but trained with a shortcut-breaking augmentation applied to the logo region:
  - **Cutout:** zero the bottom-right corner.
  - **Randomize:** replace the logo region with random noise.

**Explainers:**
- **Grad-CAM**
- **Input Gradient**

**Metrics:**
- **Accuracy (ID/OOD):** verifies shortcut sensitivity by checking performance under distribution shift.
- **Localization IoU:** overlap between thresholded saliency and the *lesion* mask (higher = explanation focuses on causal region).
- **Deletion AUC (faithfulness):** probability/confidence AUC as top-salient pixels are progressively removed.
  - Reported for **saliency-guided deletion** and a **random deletion baseline**.
  - Interpretation: if the explanation identifies truly important pixels, saliency-guided deletion should reduce confidence faster than random deletion, yielding a *lower* AUC.

## Results & Interpretation
### Quantitative summary (Grad-CAM)
Values copied from the experiment outputs in `16/exp_*/metrics_summary.csv`.

| Setting | acc_id | acc_ood | id IoU | id del AUC (sal) | id del AUC (rand) | ood IoU | ood del AUC (sal) | ood del AUC (rand) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (none) | 0.994 | 0.997 | 0.099 | 0.494 | 0.767 | 0.099 | 0.452 | 0.669 |
| Mitigated (cutout) | 0.994 | 1.000 | 0.099 | 0.491 | 0.785 | 0.097 | 0.427 | 0.698 |
| Baseline (none, alt run) | 0.994 | 0.990 | 0.099 | 0.494 | 0.759 | 0.099 | 0.464 | 0.665 |
| Mitigated (randomize) | 0.997 | 0.995 | 0.099 | 0.498 | 0.748 | 0.095 | 0.454 | 0.655 |

### Additional summary (Input Gradient)
- Baseline: `acc_id=0.994`, `acc_ood=0.979`, `id IoU=0.055`, `ood IoU=0.083`
- Mitigated (cutout): `acc_id=0.975`, `acc_ood=1.000`, `id IoU=0.063`, `ood IoU=0.089`

### What these numbers suggest
1. **OOD accuracy is high overall, but some shortcut sensitivity appears depending on the run/explainer.**
   - With Input Gradient, baseline OOD accuracy drops to ~0.979, while the cutout-mitigated model recovers to 1.0. This is consistent with some reliance on the shortcut in the baseline.
   - With Grad-CAM runs shown above, OOD accuracy remains very high (≈0.99–1.0), suggesting the model also learned the causal lesion feature strongly enough to remain accurate even when the shortcut is weakened.

2. **Localization to the causal lesion is poor for both models and explainers (IoU ≈ 0.05–0.10).**
   - Despite strong classification performance, saliency maps have low overlap with the lesion mask.
   - This highlights a key pitfall for “looks plausible” heatmaps: high accuracy does not imply explanations are focusing on the true causal region.

3. **Deletion AUC indicates saliency-guided deletion is more disruptive than random deletion (AUC lower than random), but mitigation does not substantially improve this.**
   - Across Grad-CAM runs, saliency-guided deletion AUC (≈0.45–0.50) is consistently below random deletion AUC (≈0.65–0.79), which is *directionally consistent* with faithfulness.
   - However, the differences between baseline and mitigated models are small, suggesting the chosen mitigation (cutout/randomize) did not strongly shift what the explainer highlights—or that the current metrics are not sensitive enough to capture the expected shift.

## Limitations (minimal study caveats)
- **Single synthetic setting + limited runs:** results may vary with random seeds, shortcut correlation strength, and lesion size/contrast.
- **Lesion-only IoU may miss shortcut focusing:** a more direct test would also compute overlap with the **logo mask** (e.g., “logo IoU”) to quantify shortcut attribution explicitly.
- **Deletion metric dependence:** deletion AUC can be sensitive to masking strategy, step schedule, and model calibration.