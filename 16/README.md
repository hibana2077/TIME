# Shortcut Learning & XAI Faithfulness Experiment

This project implements a controlled experiment to analyze how "shortcut learning" affects the faithfulness of Explainable AI (XAI) methods. It uses a synthetic medical-like dataset where a model may latch onto a spurious artifact (a "logo") instead of the true causal feature (a "lesion").

## Implementation Summary

### 1. Synthetic Dataset (`make_synthetic_shortcut_dataset`)
Generates 64x64 grayscale images representing tissue samples.
- **Causal Feature (Lesion)**: A bright circular disk. Presence indicates `label=1`.
- **Spurious Feature (Logo)**: A bright square in the bottom-right corner.
- **Modes**:
  - **Correlated**: The logo is highly correlated with the label (e.g., 95% of `label=1` have a logo). This induces shortcut learning.
  - **Random**: The logo appears randomly (50% prob) regardless of label. Used for OOD testing.
  - **Removed**: No logos present.

### 2. Models (`SmallCNN`)
A simple 3-layer Convolutional Neural Network (CNN) is used as the classifier.
- **Baseline**: Trained on the correlated dataset, expected to learn the "logo" shortcut.
- **Mitigated**: Trained with a mitigation strategy to force the model to look at the lesion.

### 3. Mitigation Strategies
Strategies to break the shortcut during training:
- **Cutout**: Zeros out the bottom-right corner where the logo usually appears.
- **Randomize**: Replaces the logo region with random noise to destroy the correlation.

### 4. XAI & Evaluation
The script evaluates how well saliency maps identify the true lesion vs. the shortcut.
- **Methods**: `GradCAM`, `Input Gradient`.
- **Metrics**:
  - **IoU (Intersection over Union)**: Measures overlap between the saliency map and the ground-truth lesion mask.
  - **Deletion AUC**: A faithfulness metric. Measures how quickly confidence drops as important pixels (determined by saliency) are removed.
  - **Accuracy**: Reported for ID (In-Distribution, correlated) and OOD (Out-Of-Distribution, random logo) sets.

---

## Usage Guide

### Requirements
Ensure you have the following Python packages installed:
```bash
pip install torch torchvision numpy pandas matplotlib tqdm
```

### Running the Experiment
Run the script directly using Python. By default, it runs a quick experiment with standard settings.

```bash
python shortcut_xai_experiment.py
```

### Key Command Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--out-dir` | `results_shortcut_xai` | Directory to save results and plots. |
| `--spurious-corr` | `0.95` | Strength of the shortcut correlation (0.0 to 1.0). Higher means stronger shortcut. |
| `--mitigation` | `cutout` | Mitigation strategy: `cutout` or `randomize`. |
| `--explainer` | `gradcam` | XAI method to evaluate: `gradcam` or `input_grad`. |
| `--explain-target` | `pred` | Whether to explain the `pred` (predicted) or `true` label. |
| `--ood-mode` | `random` | Test set behavior: `random` (logo independent) or `removed` (no logo). |
| `--regen-data` | `False` | Add this flag to force regeneration of synthetic data instead of using cached `.npz`. |

### Examples

**1. Run with Input Gradient explainer instead of GradCAM:**
```bash
python shortcut_xai_experiment.py --explainer input_grad
```

**2. Test with a weaker shortcut correlation (0.8):**
```bash
python shortcut_xai_experiment.py --spurious-corr 0.8
```

**3. Use "randomize" mitigation instead of "cutout":**
```bash
python shortcut_xai_experiment.py --mitigation randomize
```

### Outputs
The script creates a directory (default `results_shortcut_xai/`) containing:
- **`metrics_summary.csv`**: Quantitative results including Accuracy (ID/OOD), Mean IoU, and Deletion AUC scores.
- **`qual_baseline_id.png`**: Visualization of images, saliency maps, and masks for the baseline model on ID data.
- **`qual_mitigated_ood.png`**: Visualization for the mitigated model on OOD data.
- **`deletion_curve_*.png`**: Plots of the deletion metric curves.
