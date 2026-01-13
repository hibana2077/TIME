#!/usr/bin/env python3
"""Sanity study for saliency as "evidence" on MedMNIST datasets.

Outputs (per run):
- CSV tables: per-seed metrics, aggregated mean/std, paired t-tests
- Plots: deletion curves, stability boxplots, robustness curves, reliability diagrams, qualitative examples
- Reproducibility: config.yaml, environment.txt, pip_freeze.txt, (optional) git_info.txt

Designed to be paper-friendly: logs everything needed to cite settings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import timm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import yaml

import scipy.stats as stats

from skimage.metrics import structural_similarity as ssim

from torchmetrics.classification import (
    BinaryCalibrationError,
    MulticlassCalibrationError,
)

from captum.attr import IntegratedGradients

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import medmnist
from medmnist import INFO


# -------------------------
# Utilities / reproducibility
# -------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_pref: str) -> torch.device:
    if device_pref != "auto":
        return torch.device(device_pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def safe_run(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.STDOUT)
        return 0, out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        return int(e.returncode), e.output.decode("utf-8", errors="replace")
    except FileNotFoundError as e:
        return 127, str(e)


def save_reproducibility_bundle(out_dir: Path, config: Dict[str, Any]) -> None:
    ensure_dir(out_dir)

    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    env_lines = [
        f"timestamp: {datetime.utcnow().isoformat()}Z",
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"timm: {timm.__version__}",
        f"medmnist: {medmnist.__version__}",
    ]
    try:
        import captum  # type: ignore

        env_lines.append(f"captum: {captum.__version__}")
    except Exception:
        pass

    write_text(out_dir / "environment.txt", "\n".join(env_lines) + "\n")

    # pip freeze-like list (no shell)
    try:
        from importlib.metadata import distributions

        pkgs = sorted([f"{d.metadata['Name']}=={d.version}" for d in distributions()])
        write_text(out_dir / "pip_freeze.txt", "\n".join(pkgs) + "\n")
    except Exception as e:
        write_text(out_dir / "pip_freeze.txt", f"Failed to collect packages: {e}\n")

    # optional git info
    code, out = safe_run(["git", "rev-parse", "--show-toplevel"], cwd=out_dir)
    if code == 0:
        repo_root = Path(out.strip())
        _, head = safe_run(["git", "rev-parse", "HEAD"], cwd=repo_root)
        _, status = safe_run(["git", "status", "--porcelain"], cwd=repo_root)
        write_text(out_dir / "git_info.txt", f"HEAD: {head.strip()}\n\nSTATUS:\n{status}\n")


# -------------------------
# Data
# -------------------------


def list_medmnist_2d_classification_datasets() -> List[str]:
    names: List[str] = []
    for name, meta in INFO.items():
        # skip 3D datasets
        if meta.get("n_channels") not in (1, 3):
            continue
        task = meta.get("task")
        if task not in ("multi-class", "binary-class"):
            continue
        # Some datasets are multi-label; skip for simplicity
        if meta.get("label") is None:
            continue
        names.append(name)
    return sorted(set(names))


def _to_3ch(x: torch.Tensor) -> torch.Tensor:
    # x: (B,1,H,W) -> (B,3,H,W)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def resize_bilinear(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor


def build_loaders(
    dataset_name: str,
    data_root: Path,
    batch_size: int,
    img_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Return train/val/test loaders and (n_channels, n_classes).

    We keep preprocessing in-collate (Torch ops) to avoid PIL dependency mismatches.
    """
    # Ensure data directory exists before MedMNIST tries to use it
    ensure_dir(data_root)

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    train_ds = DataClass(split="train", root=str(data_root), download=True)
    val_ds = DataClass(split="val", root=str(data_root), download=True)
    test_ds = DataClass(split="test", root=str(data_root), download=True)

    n_channels = int(info["n_channels"])

    # medmnist labels are arrays; for classification tasks, use integer class ids
    if info["task"] == "binary-class":
        n_classes = 2
    else:
        n_classes = len(info["label"])

    def collate_fn(batch):
        xs, ys = zip(*batch)
        x = torch.from_numpy(np.stack(xs)).float()  # (B,H,W,C) or (B,H,W)

        # medmnist returns (H,W) for grayscale, (H,W,3) for RGB
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        # (B,H,W,C) -> (B,C,H,W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x / 255.0

        # Resize
        if x.shape[-1] != img_size:
            x = resize_bilinear(x, img_size)

        # Normalize: ImageNet-like for 3ch; 0.5/0.5 for 1ch
        if x.shape[1] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
        else:
            x = (x - 0.5) / 0.5

        y = torch.from_numpy(np.array(ys)).long().view(-1)
        return Batch(x=x, y=y)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, n_channels, n_classes


# -------------------------
# Model / training
# -------------------------


def create_model(backbone: str, pretrained: bool, n_classes: int, in_chans: int) -> nn.Module:
    # Use timm; set in_chans to accept grayscale if needed.
    # We still convert grayscale to 3ch for CAM stability; but allow 1ch too.
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=n_classes,
        in_chans=in_chans,
    )
    return model


def train_one_seed(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, Any]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state = None

    for _epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch.x.to(device)
            y = batch.y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

        val_acc = evaluate_accuracy(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_acc": float(best_val)}


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    for batch in loader:
        x = batch.x.to(device)
        y = batch.y.to(device)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_y, dim=0)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    logits, y = predict_logits(model, loader, device)
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def compute_ece(logits: torch.Tensor, y: torch.Tensor, n_classes: int, n_bins: int) -> float:
    probs = torch.softmax(logits, dim=1)
    if n_classes == 2:
        metric = BinaryCalibrationError(n_bins=n_bins, norm="l1")
        # For binary metric: pass prob of positive class
        return float(metric(probs[:, 1], y).item())
    metric = MulticlassCalibrationError(num_classes=n_classes, n_bins=n_bins, norm="l1")
    return float(metric(probs, y).item())


def reliability_curve(logits: torch.Tensor, y: torch.Tensor, n_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == y).float()

    bins = torch.linspace(0, 1, steps=n_bins + 1)
    bin_acc = []
    bin_conf = []
    bin_count = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.any():
            bin_acc.append(float(correct[mask].mean().item()))
            bin_conf.append(float(conf[mask].mean().item()))
            bin_count.append(int(mask.sum().item()))
        else:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_count.append(0)
    return np.array(bin_conf), np.array(bin_acc), np.array(bin_count)


# -------------------------
# Corruptions (robustness)
# -------------------------


def apply_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def apply_gaussian_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    # Simple separable Gaussian blur using conv2d; kernel size must be odd.
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1

    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    # create 2D kernel
    k2d = torch.outer(g, g)
    k2d = k2d.view(1, 1, kernel_size, kernel_size)

    b, c, h, w = x.shape
    weight = k2d.to(x.device).repeat(c, 1, 1, 1)
    return F.conv2d(x, weight, padding=kernel_size // 2, groups=c)


@torch.no_grad()
def robustness_eval(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    noise_sigmas: List[float],
    blur_kernels: List[int],
) -> Dict[str, Any]:
    model.eval()

    def acc_for_transform(transform_fn) -> float:
        correct = 0
        total = 0
        for batch in test_loader:
            x = batch.x.to(device)
            y = batch.y.to(device)
            x2 = transform_fn(x)
            logits = model(x2)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        return float(correct / max(total, 1))

    out: Dict[str, Any] = {}
    out["acc_clean"] = acc_for_transform(lambda z: z)

    for s in noise_sigmas:
        out[f"acc_noise_{s}"] = acc_for_transform(lambda z, s=s: apply_gaussian_noise(z, s))
    for k in blur_kernels:
        out[f"acc_blur_{k}"] = acc_for_transform(lambda z, k=k: apply_gaussian_blur(z, k))

    return out


# -------------------------
# Attribution (Grad-CAM / IG)
# -------------------------


def _find_target_layer_for_cam(model: nn.Module) -> nn.Module:
    """Best-effort target layer selection for Grad-CAM.

    Goal: pick a late spatial feature layer (usually the last conv feature stage),
    not the classifier head.

    Priority:
    1) timm models with feature_info -> last feature module
    2) common stage containers (layer4/stages/blocks/features)
    3) fallback: last Conv2d in the whole model

    If your backbone is transformer-like (no Conv2d), Grad-CAM is not applicable
    without a reshape_transform; we raise a clear error in that case.
    """

    # 1) Prefer timm's feature_info when available.
    try:
        fi = getattr(model, "feature_info", None)
        if fi is not None and len(fi) > 0:
            last = fi[-1]
            module_name = None
            if isinstance(last, dict):
                module_name = last.get("module")
            else:
                # FeatureInfo supports dict-like indexing in many timm versions
                try:
                    module_name = last["module"]  # type: ignore[index]
                except Exception:
                    module_name = None

            if module_name:
                try:
                    feat_mod = model.get_submodule(module_name)
                except Exception:
                    feat_mod = dict(model.named_modules()).get(module_name)

                if feat_mod is not None:
                    last_conv = None
                    for _n, m in feat_mod.named_modules():
                        if isinstance(m, nn.Conv2d):
                            last_conv = m
                    return last_conv if last_conv is not None else feat_mod
    except Exception:
        pass

    # 2) Try common stage containers.
    for container_name in ("layer4", "stages", "blocks", "features"):
        container = getattr(model, container_name, None)
        if container is None:
            continue
        last_conv = None
        for _n, m in container.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            return last_conv

    # 3) Fallback: last Conv2d globally.
    last_conv = None
    for _name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError(
            "Could not find a Conv2d layer for Grad-CAM target. "
            "If you are using a transformer-like backbone (e.g., ViT), Grad-CAM requires a reshape_transform "
            "or a different CAM method that supports non-conv features."
        )
    return last_conv


def attribution_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    target_layer = _find_target_layer_for_cam(model)
    # pytorch-grad-cam expects target_layers as a list.
    target_list = [ClassifierOutputTarget(int(t.item())) for t in targets.cpu()]

    # Construct the CAM object once, and then re-use it on many images.
    # Using the context manager ensures all hooks are released cleanly.
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=x, targets=target_list)  # (B,H,W)
    sal = torch.from_numpy(grayscale_cam).float().to(x.device)
    # (B,H,W) -> (B,1,H,W)
    return sal.unsqueeze(1)


def attribution_ig(
    model: nn.Module,
    x: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    model.eval()
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x)

    # Captum expects target as int (per-example) for classification
    attr = ig.attribute(x, baselines=baseline, target=targets, n_steps=steps)
    # Aggregate channels -> (B,1,H,W)
    sal = attr.abs().mean(dim=1, keepdim=True)
    return sal


def normalize_saliency(sal: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # per-sample min-max normalization
    b = sal.shape[0]
    flat = sal.view(b, -1)
    minv = flat.min(dim=1).values.view(b, 1, 1, 1)
    maxv = flat.max(dim=1).values.view(b, 1, 1, 1)
    return (sal - minv) / (maxv - minv + eps)


@torch.no_grad()
def get_predicted_class(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return logits.argmax(dim=1)


def deletion_curve(
    model: nn.Module,
    x: torch.Tensor,
    sal: torch.Tensor,
    steps: int,
    baseline_value: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Deletion curve using top-saliency pixels.

    We track the model confidence for the original predicted class.
    Returns: fractions_removed, confidences, auc
    """

    model.eval()
    with torch.no_grad():
        pred_class = get_predicted_class(model, x)
        base_logits = model(x)
        base_probs = torch.softmax(base_logits, dim=1)
        base_conf = base_probs.gather(1, pred_class.view(-1, 1)).squeeze(1)

    sal = normalize_saliency(sal)
    b, _, h, w = sal.shape
    sal_flat = sal.view(b, -1)
    x_flat = x.view(b, x.shape[1], -1)

    # indices sorted desc by saliency
    order = torch.argsort(sal_flat, dim=1, descending=True)

    fractions = np.linspace(0, 1, steps + 1)
    confidences = []

    for frac in fractions:
        k = int(round(frac * (h * w)))
        x_masked = x_flat.clone()
        if k > 0:
            idx = order[:, :k]  # (B,k)
            # apply baseline per-channel
            for c in range(x.shape[1]):
                x_masked[:, c].scatter_(1, idx, baseline_value)
        x2 = x_masked.view(b, x.shape[1], h, w)
        with torch.no_grad():
            probs = torch.softmax(model(x2), dim=1)
            conf = probs.gather(1, pred_class.view(-1, 1)).squeeze(1)
        confidences.append(conf.detach().cpu().numpy())

    conf_arr = np.stack(confidences, axis=0)  # (T,B)
    conf_mean = conf_arr.mean(axis=1)

    auc = float(np.trapz(conf_mean, fractions))

    # Also return the base conf for sanity (not used in AUC)
    _ = base_conf
    return fractions, conf_mean, auc


def stability_metrics(
    sal1: torch.Tensor,
    sal2: torch.Tensor,
) -> Tuple[float, float]:
    """Return (SSIM, Spearman rank corr) averaged over batch."""
    sal1 = normalize_saliency(sal1).detach().cpu().numpy()
    sal2 = normalize_saliency(sal2).detach().cpu().numpy()

    ssims: List[float] = []
    sps: List[float] = []
    for i in range(sal1.shape[0]):
        a = sal1[i, 0]
        b = sal2[i, 0]
        ssims.append(float(ssim(a, b, data_range=1.0)))
        ra = a.reshape(-1)
        rb = b.reshape(-1)
        sp = stats.spearmanr(ra, rb).correlation
        if np.isnan(sp):
            sp = 0.0
        sps.append(float(sp))

    return float(np.mean(ssims)), float(np.mean(sps))


def sample_attr_loader(test_loader: DataLoader, n_samples: int, seed: int) -> Iterable[Batch]:
    # Build a deterministic subset by consuming batches and slicing.
    rng = np.random.default_rng(seed)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for batch in test_loader:
        xs.append(batch.x)
        ys.append(batch.y)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    n = x.shape[0]
    idx = rng.permutation(n)[: min(n_samples, n)]
    x2 = x[idx]
    y2 = y[idx]

    # yield in mini-batches
    bs = min(32, n_samples)
    for i in range(0, x2.shape[0], bs):
        yield Batch(x=x2[i : i + bs], y=y2[i : i + bs])


# -------------------------
# Plotting
# -------------------------


def plot_reliability(bin_conf: np.ndarray, bin_acc: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(5, 5))
    mask = ~np.isnan(bin_conf) & ~np.isnan(bin_acc)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.plot(bin_conf[mask], bin_acc[mask], marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_deletion_curve(fractions: np.ndarray, conf: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(fractions, conf, marker="o")
    plt.xlabel("Fraction of pixels deleted")
    plt.ylabel("Confidence (original predicted class)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_robustness_curve(df: pd.DataFrame, out_path: Path, title: str) -> None:
    # df has columns: severity_type, severity, acc
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=df, x="severity", y="acc", hue="severity_type", marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_stability_boxplot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    # df columns: method, perturb, metric, value
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="perturb", y="value", hue="method")
    plt.ylim(-1, 1)
    plt.xlabel("Perturbation")
    plt.ylabel("Similarity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_qualitative_examples(
    x: torch.Tensor,
    heatmaps: Dict[str, torch.Tensor],
    out_path: Path,
    title: str,
    max_items: int = 5,
) -> None:
    # x: (B,C,H,W) normalized; visualize via min-max
    b = min(x.shape[0], max_items)
    methods = list(heatmaps.keys())
    ncols = 1 + len(methods)
    plt.figure(figsize=(3.5 * ncols, 3.0 * b))

    for i in range(b):
        img = x[i].detach().cpu()
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        img = img.permute(1, 2, 0).numpy()

        ax = plt.subplot(b, ncols, i * ncols + 1)
        ax.imshow(img.squeeze() if img.shape[2] == 1 else img)
        ax.set_title("Input")
        ax.axis("off")

        for j, m in enumerate(methods):
            hm = heatmaps[m][i, 0].detach().cpu().numpy()
            ax2 = plt.subplot(b, ncols, i * ncols + 2 + j)
            ax2.imshow(img.squeeze() if img.shape[2] == 1 else img)
            ax2.imshow(hm, cmap="jet", alpha=0.45)
            ax2.set_title(m)
            ax2.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------
# Experiment driver
# -------------------------


def paired_ttest(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    # returns t-stat and p-value
    t, p = stats.ttest_rel(a, b, nan_policy="omit")
    return {"t": float(t), "p": float(p)}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str, default="./outputs", help="output root")
    parser.add_argument("--data-root", type=str, default="./data", help="dataset cache root")

    parser.add_argument("--datasets", type=str, default="all", help="comma-separated dataset names, or 'all'")

    parser.add_argument("--backbone", type=str, default="resnet18", help="timm backbone")
    parser.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained weights")

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")

    parser.add_argument("--attr-samples", type=int, default=256, help="# test samples for attribution metrics")
    parser.add_argument("--attr-batch", type=int, default=16)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--deletion-steps", type=int, default=20)
    parser.add_argument("--baseline-value", type=float, default=0.0)

    parser.add_argument("--stability-noise", type=float, default=0.01)
    parser.add_argument("--stability-brightness", type=float, default=0.05)

    parser.add_argument("--ece-bins", type=int, default=15)

    parser.add_argument("--noise-sigmas", type=str, default="0.0,0.05,0.10,0.15")
    parser.add_argument("--blur-kernels", type=str, default="1,3,5,7")

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")

    args = parser.parse_args()

    out_root = Path(args.out)
    data_root = Path(args.data_root)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"sanity_{run_id}"
    plots_dir = run_dir / "plots"
    tables_dir = run_dir / "tables"
    repro_dir = run_dir / "repro"

    ensure_dir(plots_dir)
    ensure_dir(tables_dir)
    ensure_dir(repro_dir)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.datasets == "all":
        datasets = list_medmnist_2d_classification_datasets()
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    device = get_device(args.device)

    noise_sigmas = [float(x) for x in args.noise_sigmas.split(",")]
    blur_kernels = [int(x) for x in args.blur_kernels.split(",")]

    config = {
        "run_id": run_id,
        **vars(args),
        "resolved": {
            "datasets": datasets,
            "seeds": seeds,
            "device": str(device),
        },
    }
    save_reproducibility_bundle(repro_dir, config)

    per_seed_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    deletion_rows: List[Dict[str, Any]] = []

    for dataset_name in datasets:
        info = INFO[dataset_name]

        for seed in seeds:
            seed_everything(seed)

            train_loader, val_loader, test_loader, n_channels, n_classes = build_loaders(
                dataset_name=dataset_name,
                data_root=data_root,
                batch_size=args.batch_size,
                img_size=args.img_size,
                num_workers=args.num_workers,
                pin_memory=(device.type in ("cuda",)),
                seed=seed,
            )

            # Use 3ch models by default; loaders normalize accordingly.
            in_chans = 3 if n_channels == 3 else 1

            model = create_model(args.backbone, args.pretrained, n_classes, in_chans=in_chans)

            train_meta = train_one_seed(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            logits_test, y_test = predict_logits(model, test_loader, device)
            acc_test = float((logits_test.argmax(dim=1) == y_test).float().mean().item())
            ece = compute_ece(logits_test, y_test, n_classes=n_classes, n_bins=args.ece_bins)

            rob = robustness_eval(
                model=model,
                test_loader=test_loader,
                device=device,
                noise_sigmas=noise_sigmas,
                blur_kernels=blur_kernels,
            )

            # Attribution subset
            # Build subset batches (CPU tensors), then compute on device
            attr_batches = list(sample_attr_loader(test_loader, n_samples=args.attr_samples, seed=seed + 12345))

            # Accumulate method-level metrics
            method_metrics: Dict[str, Dict[str, Any]] = {
                "gradcam": {"deletion_auc": [], "stability_ssim_noise": [], "stability_spearman_noise": [], "stability_ssim_brightness": [], "stability_spearman_brightness": []},
                "ig": {"deletion_auc": [], "stability_ssim_noise": [], "stability_spearman_noise": [], "stability_ssim_brightness": [], "stability_spearman_brightness": []},
            }

            # Qualitative examples (first batch)
            qual_x = None
            qual_heatmaps: Dict[str, torch.Tensor] = {}

            for batch in attr_batches:
                x = batch.x.to(device)

                # Ensure channels match model
                if in_chans == 3 and x.shape[1] == 1:
                    x = _to_3ch(x)
                if in_chans == 1 and x.shape[1] == 3:
                    x = x[:, :1]

                # Use original predicted class as target for saliency comparisons
                with torch.no_grad():
                    targets = get_predicted_class(model, x)

                # Grad-CAM
                sal_cam = attribution_gradcam(model, x, targets)
                # IG
                sal_ig = attribution_ig(model, x, targets, steps=args.ig_steps)

                if qual_x is None:
                    qual_x = x.detach().cpu()
                    qual_heatmaps["Grad-CAM"] = normalize_saliency(sal_cam).detach().cpu()
                    qual_heatmaps["Integrated Gradients"] = normalize_saliency(sal_ig).detach().cpu()

                for method_name, sal in [("gradcam", sal_cam), ("ig", sal_ig)]:
                    # Faithfulness deletion
                    fr, conf, auc = deletion_curve(
                        model=model,
                        x=x,
                        sal=sal,
                        steps=args.deletion_steps,
                        baseline_value=args.baseline_value,
                    )
                    method_metrics[method_name]["deletion_auc"].append(auc)

                    # Store a representative deletion curve for plotting (first attr batch only)
                    if batch is attr_batches[0]:
                        deletion_rows.append(
                            {
                                "dataset": dataset_name,
                                "seed": seed,
                                "method": method_name,
                                "fractions": fr.tolist(),
                                "conf": conf.tolist(),
                            }
                        )

                    # Stability: small gaussian noise
                    x_noise = apply_gaussian_noise(x, args.stability_noise)
                    if method_name == "gradcam":
                        sal_noise = attribution_gradcam(model, x_noise, targets)
                    else:
                        sal_noise = attribution_ig(model, x_noise, targets, steps=args.ig_steps)
                    ssim_v, sp_v = stability_metrics(sal, sal_noise)
                    method_metrics[method_name]["stability_ssim_noise"].append(ssim_v)
                    method_metrics[method_name]["stability_spearman_noise"].append(sp_v)

                    # Stability: brightness shift (in normalized space; simple additive)
                    x_bright = x + args.stability_brightness
                    if method_name == "gradcam":
                        sal_bright = attribution_gradcam(model, x_bright, targets)
                    else:
                        sal_bright = attribution_ig(model, x_bright, targets, steps=args.ig_steps)
                    ssim_v2, sp_v2 = stability_metrics(sal, sal_bright)
                    method_metrics[method_name]["stability_ssim_brightness"].append(ssim_v2)
                    method_metrics[method_name]["stability_spearman_brightness"].append(sp_v2)

            # Per-seed row (model-level)
            row = {
                "dataset": dataset_name,
                "seed": seed,
                "task": info.get("task"),
                "n_channels": n_channels,
                "n_classes": n_classes,
                "backbone": args.backbone,
                "pretrained": bool(args.pretrained),
                "img_size": args.img_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "acc_test": acc_test,
                "ece": ece,
                **train_meta,
                **rob,
            }

            # Method-level summaries (per seed)
            for method_name, mm in method_metrics.items():
                row[f"{method_name}_deletion_auc"] = float(np.mean(mm["deletion_auc"]))
                row[f"{method_name}_stability_ssim_noise"] = float(np.mean(mm["stability_ssim_noise"]))
                row[f"{method_name}_stability_spearman_noise"] = float(np.mean(mm["stability_spearman_noise"]))
                row[f"{method_name}_stability_ssim_brightness"] = float(np.mean(mm["stability_ssim_brightness"]))
                row[f"{method_name}_stability_spearman_brightness"] = float(np.mean(mm["stability_spearman_brightness"]))

                stability_rows += [
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": method_name,
                        "perturb": "noise",
                        "metric": "ssim",
                        "value": float(v),
                    }
                    for v in mm["stability_ssim_noise"]
                ]
                stability_rows += [
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": method_name,
                        "perturb": "brightness",
                        "metric": "ssim",
                        "value": float(v),
                    }
                    for v in mm["stability_ssim_brightness"]
                ]

            per_seed_rows.append(row)

            # Per-seed plots
            bin_conf, bin_acc, bin_cnt = reliability_curve(logits_test, y_test, n_bins=args.ece_bins)
            plot_reliability(
                bin_conf,
                bin_acc,
                plots_dir / f"{dataset_name}_seed{seed}_reliability.png",
                title=f"Reliability Diagram ({dataset_name}, seed={seed}, ECE={ece:.3f})",
            )

            # Qualitative saliency
            if qual_x is not None and qual_heatmaps:
                plot_qualitative_examples(
                    x=qual_x,
                    heatmaps=qual_heatmaps,
                    out_path=plots_dir / f"{dataset_name}_seed{seed}_qualitative.png",
                    title=f"Qualitative attributions ({dataset_name}, seed={seed})",
                )

        # Dataset-level robustness plot using per-seed averages
        df_seed = pd.DataFrame([r for r in per_seed_rows if r["dataset"] == dataset_name])
        rob_rows = []
        for s in noise_sigmas:
            rob_rows.append({"severity_type": "noise_sigma", "severity": s, "acc": float(df_seed[f"acc_noise_{s}"].mean())})
        for k in blur_kernels:
            rob_rows.append({"severity_type": "blur_kernel", "severity": k, "acc": float(df_seed[f"acc_blur_{k}"].mean())})
        plot_robustness_curve(
            pd.DataFrame(rob_rows),
            plots_dir / f"{dataset_name}_robustness.png",
            title=f"Robustness under simple corruptions ({dataset_name})",
        )

    # Save tables
    df = pd.DataFrame(per_seed_rows)
    df.to_csv(tables_dir / "per_seed_metrics.csv", index=False)

    # Aggregation (mean/std across seeds)
    metric_cols = [c for c in df.columns if c not in ("seed",)]
    # Keep dataset grouping
    group_cols = ["dataset", "task", "n_channels", "n_classes", "backbone", "pretrained", "img_size", "epochs", "batch_size", "lr", "weight_decay"]

    agg = df.groupby(group_cols, dropna=False)
    df_mean = agg.mean(numeric_only=True).reset_index()
    df_std = agg.std(numeric_only=True).reset_index()

    # Flatten mean/std into one table
    df_out = df_mean.copy()
    for col in df_mean.columns:
        if col in group_cols:
            continue
        df_out[f"{col}_std"] = df_std[col]

    df_out.to_csv(tables_dir / "agg_mean_std.csv", index=False)

    # Paired tests across seeds (Grad-CAM vs IG) per dataset
    tests = []
    for dataset_name in sorted(df["dataset"].unique()):
        ddf = df[df["dataset"] == dataset_name].sort_values("seed")
        if len(ddf) < 2:
            continue
        for metric in [
            "deletion_auc",
            "stability_ssim_noise",
            "stability_spearman_noise",
            "stability_ssim_brightness",
            "stability_spearman_brightness",
        ]:
            a = ddf[f"gradcam_{metric}"].to_numpy()
            b = ddf[f"ig_{metric}"].to_numpy()
            res = paired_ttest(a, b)
            tests.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "n_seeds": int(len(ddf)),
                    "gradcam_mean": float(np.mean(a)),
                    "ig_mean": float(np.mean(b)),
                    "delta_ig_minus_gradcam": float(np.mean(b - a)),
                    **res,
                }
            )

    pd.DataFrame(tests).to_csv(tables_dir / "paired_ttests_gradcam_vs_ig.csv", index=False)

    # Stability boxplots (SSIM only, pooled)
    stab_df = pd.DataFrame(stability_rows)
    if not stab_df.empty:
        for dataset_name in stab_df["dataset"].unique():
            sub = stab_df[(stab_df["dataset"] == dataset_name) & (stab_df["metric"] == "ssim")]
            if sub.empty:
                continue
            plot_stability_boxplot(
                sub,
                plots_dir / f"{dataset_name}_stability_ssim_boxplot.png",
                title=f"Attribution stability (SSIM) under small perturbations ({dataset_name})",
            )

    # Deletion curve plots (first attr batch per seed) aggregated visually
    del_df = pd.DataFrame(deletion_rows)
    if not del_df.empty:
        # Store json to allow paper fig regeneration
        del_df.to_json(tables_dir / "deletion_curves_raw.json", orient="records")

        for dataset_name in del_df["dataset"].unique():
            for method in del_df[del_df["dataset"] == dataset_name]["method"].unique():
                curves = del_df[(del_df["dataset"] == dataset_name) & (del_df["method"] == method)]
                if curves.empty:
                    continue
                # average across seeds
                fractions = np.array(curves.iloc[0]["fractions"], dtype=float)
                confs = np.stack([np.array(c, dtype=float) for c in curves["conf"].tolist()], axis=0)
                conf_mean = confs.mean(axis=0)
                plot_deletion_curve(
                    fractions,
                    conf_mean,
                    plots_dir / f"{dataset_name}_{method}_deletion_curve.png",
                    title=f"Deletion curve (mean over seeds) - {dataset_name} - {method}",
                )

    # A minimal run summary
    summary = {
        "run_dir": str(run_dir),
        "datasets": datasets,
        "n_rows": int(len(df)),
        "tables": {
            "per_seed": str(tables_dir / "per_seed_metrics.csv"),
            "agg": str(tables_dir / "agg_mean_std.csv"),
            "ttests": str(tables_dir / "paired_ttests_gradcam_vs_ig.csv"),
        },
        "plots_dir": str(plots_dir),
    }
    write_text(run_dir / "run_summary.json", json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
