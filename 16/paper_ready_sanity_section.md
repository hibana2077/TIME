# Lightweight Empirical Sanity Check: Evaluating Saliency as Evidence (Paper-ready, English)

## Motivation (1–2 paragraphs)

Clinical trust is often supported by visually plausible post-hoc saliency maps. However, plausibility alone is not sufficient evidence: explanations can be unstable under minor perturbations, weakly coupled to model decision-making, and misleading under distribution shift. To operationalize *transparent and trustworthy* ML as an **evidence chain**, we complement qualitative saliency visualization with quantitative criteria that test whether explanations (i) reflect model reliance on the highlighted regions, (ii) remain stable under benign perturbations, and (iii) remain meaningful when the model is stressed by simple corruptions.

## Experimental setup (drop-in Methods text)

We conduct a lightweight sanity study across the MedMNIST benchmark suite (2D medical imaging datasets with fixed train/validation/test splits). For each dataset, we train a standard convolutional backbone from **timm** (ResNet-18 by default) with an image size of 224 and cross-entropy objective. We repeat all experiments over multiple random seeds (default: 5) and report mean ± standard deviation across seeds.

### Explanation methods

We compare two widely used attribution methods:

- **Grad-CAM**: produces class-discriminative heatmaps from the final convolutional activations.
- **Integrated Gradients (IG)**: computes path-integrated attributions from a zero baseline.

### Metrics (paper-ready definitions)

We evaluate explanations under four complementary criteria:

1. **Faithfulness (Deletion AUC)**. For each test image, we rank pixels by attribution magnitude and progressively mask the top fraction of pixels. We then track the model’s confidence for the original predicted class as a function of the masked fraction. We summarize faithfulness by the area under this deletion curve (AUC); lower AUC indicates a steeper confidence drop and stronger coupling between highlighted regions and the model decision.

2. **Stability (Attribution similarity under small perturbations)**. For each image, we apply small input perturbations (Gaussian noise and brightness shifts) and recompute attributions. We quantify stability using similarity between attribution maps, including **SSIM** and **Spearman rank correlation**. Higher values indicate more stable explanations.

3. **Robustness (Performance under simple corruptions)**. We evaluate classification accuracy under controlled corruptions (Gaussian noise and Gaussian blur at multiple severity levels) and summarize the accuracy drop relative to clean test performance.

4. **Calibration (ECE)**. We compute expected calibration error (ECE) with fixed binning to quantify the alignment between predicted confidence and empirical accuracy. We additionally visualize reliability diagrams.

## Reporting format (what we produce for the paper)

For each dataset, we produce:

- **Tables**: per-seed metrics and aggregated mean ± std (accuracy, ECE, robustness accuracy at each severity, deletion AUC, stability metrics).
- **Statistical validation**: paired t-tests across seeds comparing Grad-CAM vs IG for deletion AUC and stability metrics.
- **Figures**:
  - deletion curves (mean over seeds),
  - stability boxplots, 
  - robustness curves (accuracy vs severity),
  - reliability diagrams, 
  - qualitative examples (input + overlay heatmaps).

## Expected takeaway (Results narrative template)

Across datasets, we observe that attribution methods can produce visually plausible heatmaps while exhibiting (i) non-trivial instability under small perturbations and (ii) substantially different faithfulness scores under deletion-based tests. These findings illustrate why post-hoc explanations should not be treated as standalone evidence for trust. Instead, explanation artifacts should be integrated into a multi-dimensional evaluation protocol aligned with deployment risks—linking interpretability outputs to faithfulness, stability, robustness, and calibration.

---

## How to run (reproducible commands)

From the workspace root:

- Install dependencies:
  - `pip install -r 16/requirements.txt`

- Run all MedMNIST 2D classification datasets with 5 seeds:
  - `python 16/experiment_sanity_medmnist.py --datasets all --seeds 0,1,2,3,4 --pretrained`

Outputs are saved under `16/outputs/sanity_<timestamp>/` including CSV tables, plots, and reproducibility metadata.
