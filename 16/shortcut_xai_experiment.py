import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class SplitConfig:
    name: str
    spurious_mode: str  # "correlated" | "random" | "removed"
    spurious_strength: float  # correlation in [0,1], only used if correlated


def _draw_disk_mask(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def make_synthetic_shortcut_dataset(
    *,
    n: int,
    image_size: int,
    lesion_radius: int,
    logo_size: int,
    noise_std: float,
    split: SplitConfig,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Generate a binary classification dataset.

    Label definition (causal): y=1 means lesion present; y=0 means no lesion.
    Spurious feature: bright logo square at bottom-right.

    Splits:
      - correlated: P(logo=1|y=1)=p, P(logo=1|y=0)=1-p
      - random: logo independent of label (P=0.5)
      - removed: logo always absent

    Returns a dict with:
      - images: float32, shape (N,1,H,W), approx in [0,1]
      - labels: int64, shape (N,)
      - lesion_masks: uint8, shape (N,1,H,W)
      - logo_masks: uint8, shape (N,1,H,W)
    """

    rng = np.random.default_rng(seed)
    h = w = image_size

    labels = rng.integers(0, 2, size=(n,), dtype=np.int64)
    images = rng.normal(loc=0.0, scale=noise_std, size=(n, h, w)).astype(np.float32)

    lesion_masks = np.zeros((n, h, w), dtype=np.uint8)
    logo_masks = np.zeros((n, h, w), dtype=np.uint8)

    # Precompute logo region.
    y0 = h - logo_size
    x0 = w - logo_size

    for i in range(n):
        y = int(labels[i])

        # Lesion (causal)
        if y == 1:
            # Avoid placing lesion inside the logo region for clarity.
            # (Still allows bottom-right-ish placements, just not overlapping the square.)
            while True:
                cy = int(rng.integers(lesion_radius, h - lesion_radius))
                cx = int(rng.integers(lesion_radius, w - lesion_radius))
                if not (cy >= y0 - lesion_radius and cx >= x0 - lesion_radius):
                    break
            lesion = _draw_disk_mask(h, w, cy, cx, lesion_radius)
            lesion_masks[i, lesion] = 1
            images[i, lesion] += 1.0

        # Logo (spurious)
        if split.spurious_mode == "removed":
            logo_present = False
        elif split.spurious_mode == "random":
            logo_present = bool(rng.random() < 0.5)
        elif split.spurious_mode == "correlated":
            p = float(split.spurious_strength)
            # Correlate with label.
            if y == 1:
                logo_present = bool(rng.random() < p)
            else:
                logo_present = bool(rng.random() < (1.0 - p))
        else:
            raise ValueError(f"Unknown spurious_mode: {split.spurious_mode}")

        if logo_present:
            logo_masks[i, y0:h, x0:w] = 1
            images[i, y0:h, x0:w] += 1.0

    # Normalize to a stable range for training.
    images = (images - images.min()) / (images.max() - images.min() + 1e-8)
    images = images[:, None, :, :]
    lesion_masks = lesion_masks[:, None, :, :]
    logo_masks = logo_masks[:, None, :, :]

    return {
        "images": images.astype(np.float32),
        "labels": labels.astype(np.int64),
        "lesion_masks": lesion_masks,
        "logo_masks": logo_masks,
    }


class NumpyShortcutDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        *,
        mitigation: Optional[str] = None,  # None | "cutout" | "randomize"
        logo_size: int = 8,
        noise_std: float = 0.2,
        seed: int = 0,
    ):
        self.images = data["images"]
        self.labels = data["labels"]
        self.lesion_masks = data["lesion_masks"]
        self.logo_masks = data["logo_masks"]
        self.mitigation = mitigation
        self.logo_size = int(logo_size)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

        # Cache region indices
        _, _, h, w = self.images.shape
        self._logo_y0 = h - self.logo_size
        self._logo_x0 = w - self.logo_size

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        x = self.images[idx].copy()
        y = int(self.labels[idx])
        lesion_mask = self.lesion_masks[idx]
        logo_mask = self.logo_masks[idx]

        if self.mitigation is not None:
            y0 = self._logo_y0
            x0 = self._logo_x0
            if self.mitigation == "cutout":
                x[:, y0:, x0:] = 0.0
            elif self.mitigation == "randomize":
                noise = self.rng.normal(0.0, self.noise_std, size=x[:, y0:, x0:].shape).astype(
                    np.float32
                )
                # Keep in [0,1] range to avoid distribution shift.
                x[:, y0:, x0:] = np.clip(noise * 0.25 + 0.5, 0.0, 1.0)
            else:
                raise ValueError(f"Unknown mitigation: {self.mitigation}")

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.from_numpy(lesion_mask),
            torch.from_numpy(logo_mask),
        )


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 32x32
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 16x16
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.head(x)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._grads = None
        self._handles = []

        def forward_hook(_module, _inp, out):
            self._activations = out

        def backward_hook(_module, grad_in, grad_out):
            self._grads = grad_out[0]

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
        cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        return cam

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        """Return CAM in input resolution, shape (B,1,H,W), in [0,1]."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        # Gather target logits and backprop.
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward()

        acts = self._activations  # (B,C,h,w)
        grads = self._grads  # (B,C,h,w)
        if acts is None or grads is None:
            raise RuntimeError("GradCAM hooks did not capture activations/grads")

        weights = grads.mean(dim=(-2, -1), keepdim=True)  # (B,C,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = self._normalize_cam(cam.detach())
        return cam


def input_gradient_saliency(model: nn.Module, x: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
    """Vanilla input gradients saliency, aggregated to (B,1,H,W), normalized to [0,1]."""
    x = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    score = logits.gather(1, class_idx.view(-1, 1)).sum()
    score.backward()
    grad = x.grad.detach().abs()  # (B,1,H,W)
    grad = grad - grad.amin(dim=(-2, -1), keepdim=True)
    grad = grad / (grad.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return grad


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y, *_ in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct) / float(total + 1e-8)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(epochs):
        for x, y, *_ in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()


@torch.no_grad()
def _model_confidence_for_class(model: nn.Module, x: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return probs.gather(1, class_idx.view(-1, 1)).squeeze(1)


def deletion_curve(
    *,
    model: nn.Module,
    x: torch.Tensor,
    saliency: torch.Tensor,
    target_class: torch.Tensor,
    steps: int,
    mode: str,  # "saliency" | "random"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fractions, confidences) for deletion.

    We progressively replace the top-fraction pixels (by saliency or random) with the per-image mean.
    """
    assert x.ndim == 4 and saliency.ndim == 4
    b, _, h, w = x.shape

    x0 = x.detach()
    sal = saliency.detach()

    # Flatten indices.
    flat_sal = sal.view(b, -1)
    flat_x0 = x0.view(b, -1)

    # Replacement baseline: per-image mean
    baseline = flat_x0.mean(dim=1, keepdim=True)

    fractions = np.linspace(0.0, 1.0, steps + 1)
    confidences: List[float] = []

    if mode == "saliency":
        order = torch.argsort(flat_sal, dim=1, descending=True)
    elif mode == "random":
        order = torch.stack([torch.randperm(h * w, device=x.device) for _ in range(b)], dim=0)
    else:
        raise ValueError(f"Unknown deletion mode: {mode}")

    for frac in fractions:
        k = int(round(frac * h * w))
        x_del = flat_x0.clone()
        if k > 0:
            idx = order[:, :k]
            x_del.scatter_(1, idx, baseline.expand_as(idx).to(x_del.dtype))
        x_del = x_del.view(b, 1, h, w)
        conf = _model_confidence_for_class(model, x_del, target_class)
        confidences.append(float(conf.mean().detach().cpu().item()))

    return fractions, np.array(confidences, dtype=np.float32)


def auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def saliency_iou(
    *,
    saliency: torch.Tensor,
    lesion_mask: torch.Tensor,
    top_percent: float,
) -> torch.Tensor:
    """Compute IoU between lesion mask and top-percent saliency mask.

    Returns IoU per sample, shape (B,).
    """
    b = saliency.shape[0]
    sal = saliency.detach().view(b, -1)
    lesion = lesion_mask.detach().view(b, -1).float()

    n = sal.shape[1]
    k = max(1, int(round(top_percent * n)))
    topk_vals = torch.topk(sal, k=k, dim=1).values
    thresh = topk_vals.min(dim=1).values
    sal_bin = (sal >= thresh.unsqueeze(1)).float()

    inter = (sal_bin * lesion).sum(dim=1)
    union = ((sal_bin + lesion) > 0).float().sum(dim=1)
    return inter / (union + 1e-8)


def overlay_cam_on_image(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Return RGB image with CAM overlay, both in [0,1]."""
    assert img.ndim == 2
    assert cam.ndim == 2
    heat = plt.get_cmap("jet")(cam)[:, :, :3]
    rgb = np.stack([img, img, img], axis=-1)
    out = (0.55 * rgb + 0.45 * heat)
    return np.clip(out, 0.0, 1.0)


def save_qualitative_examples(
    *,
    out_path: str,
    images: torch.Tensor,
    cams: torch.Tensor,
    lesion_masks: torch.Tensor,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    b = images.shape[0]
    fig, axes = plt.subplots(b, 3, figsize=(9, 3 * b))
    if b == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(b):
        img = images[i, 0].detach().cpu().numpy()
        cam = cams[i, 0].detach().cpu().numpy()
        mask = lesion_masks[i, 0].detach().cpu().numpy()

        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay_cam_on_image(img, cam))
        axes[i, 1].set_title("Saliency overlay")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask, cmap="gray")
        axes[i, 2].set_title("Lesion mask")
        axes[i, 2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using device: {device}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Dataset cache
    cache_path = os.path.join(
        out_dir,
        f"synthetic_cache_s{args.seed}_N{args.n_train}_{args.n_test}_img{args.image_size}.npz",
    )

    if os.path.exists(cache_path) and not args.regen_data:
        cached = np.load(cache_path)
        train_data = {
            "images": cached["train_images"],
            "labels": cached["train_labels"],
            "lesion_masks": cached["train_lesion_masks"],
            "logo_masks": cached["train_logo_masks"],
        }
        test_id_data = {
            "images": cached["test_id_images"],
            "labels": cached["test_id_labels"],
            "lesion_masks": cached["test_id_lesion_masks"],
            "logo_masks": cached["test_id_logo_masks"],
        }
        test_ood_data = {
            "images": cached["test_ood_images"],
            "labels": cached["test_ood_labels"],
            "lesion_masks": cached["test_ood_lesion_masks"],
            "logo_masks": cached["test_ood_logo_masks"],
        }
    else:
        seed_everything(args.seed)
        train_data = make_synthetic_shortcut_dataset(
            n=args.n_train,
            image_size=args.image_size,
            lesion_radius=args.lesion_radius,
            logo_size=args.logo_size,
            noise_std=args.noise_std,
            split=SplitConfig(name="train", spurious_mode="correlated", spurious_strength=args.spurious_corr),
            seed=args.seed,
        )
        test_id_data = make_synthetic_shortcut_dataset(
            n=args.n_test,
            image_size=args.image_size,
            lesion_radius=args.lesion_radius,
            logo_size=args.logo_size,
            noise_std=args.noise_std,
            split=SplitConfig(name="test_id", spurious_mode="correlated", spurious_strength=args.spurious_corr),
            seed=args.seed + 1,
        )
        test_ood_data = make_synthetic_shortcut_dataset(
            n=args.n_test,
            image_size=args.image_size,
            lesion_radius=args.lesion_radius,
            logo_size=args.logo_size,
            noise_std=args.noise_std,
            split=SplitConfig(name="test_ood", spurious_mode=args.ood_mode, spurious_strength=0.5),
            seed=args.seed + 2,
        )

        np.savez_compressed(
            cache_path,
            train_images=train_data["images"],
            train_labels=train_data["labels"],
            train_lesion_masks=train_data["lesion_masks"],
            train_logo_masks=train_data["logo_masks"],
            test_id_images=test_id_data["images"],
            test_id_labels=test_id_data["labels"],
            test_id_lesion_masks=test_id_data["lesion_masks"],
            test_id_logo_masks=test_id_data["logo_masks"],
            test_ood_images=test_ood_data["images"],
            test_ood_labels=test_ood_data["labels"],
            test_ood_lesion_masks=test_ood_data["lesion_masks"],
            test_ood_logo_masks=test_ood_data["logo_masks"],
        )

    # Loaders
    train_ds_base = NumpyShortcutDataset(
        train_data, mitigation=None, logo_size=args.logo_size, noise_std=args.noise_std, seed=args.seed
    )
    train_ds_mitig = NumpyShortcutDataset(
        train_data,
        mitigation=args.mitigation,
        logo_size=args.logo_size,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    test_id_ds = NumpyShortcutDataset(test_id_data, mitigation=None, logo_size=args.logo_size)
    test_ood_ds = NumpyShortcutDataset(test_ood_data, mitigation=None, logo_size=args.logo_size)

    train_loader_base = DataLoader(train_ds_base, batch_size=args.batch_size, shuffle=True)
    train_loader_mitig = DataLoader(train_ds_mitig, batch_size=args.batch_size, shuffle=True)
    test_id_loader = DataLoader(test_id_ds, batch_size=args.batch_size, shuffle=False)
    test_ood_loader = DataLoader(test_ood_ds, batch_size=args.batch_size, shuffle=False)

    # Models
    seed_everything(args.seed)
    model_base = SmallCNN(num_classes=2)
    train_model(model_base, train_loader_base, device=device, epochs=args.epochs, lr=args.lr)

    seed_everything(args.seed)
    model_mitig = SmallCNN(num_classes=2)
    train_model(model_mitig, train_loader_mitig, device=device, epochs=args.epochs, lr=args.lr)

    # Accuracy
    acc_base_id = accuracy(model_base.to(device), test_id_loader, device)
    acc_base_ood = accuracy(model_base.to(device), test_ood_loader, device)
    acc_mitig_id = accuracy(model_mitig.to(device), test_id_loader, device)
    acc_mitig_ood = accuracy(model_mitig.to(device), test_ood_loader, device)

    print(f"Baseline Acc ID:  {acc_base_id:.3f} | OOD: {acc_base_ood:.3f}")
    print(f"Mitigated Acc ID: {acc_mitig_id:.3f} | OOD: {acc_mitig_ood:.3f}")

    # Explanation metrics on a subset (to keep it fast)
    n_eval = min(args.eval_samples, args.n_test)

    def eval_xai_metrics(model: nn.Module, dataset: Dataset, tag: str) -> Dict[str, float]:
        model = model.to(device)
        model.eval()

        # Pick first n_eval samples deterministically
        xs = []
        ys = []
        lesion_masks = []
        for i in range(n_eval):
            x, y, lesion, _logo = dataset[i]
            xs.append(x)
            ys.append(y)
            lesion_masks.append(lesion)
        x = torch.stack(xs, dim=0).to(device)
        y = torch.stack(ys, dim=0).to(device)
        lesion = torch.stack(lesion_masks, dim=0).to(device)

        with torch.no_grad():
            pred = model(x).argmax(dim=1)

        target_class = pred if args.explain_target == "pred" else y

        # Compute saliency
        if args.explainer == "gradcam":
            explainer = GradCAM(model, target_layer=model.conv3)
            cam = explainer(x, target_class)
            explainer.close()
        elif args.explainer == "input_grad":
            cam = input_gradient_saliency(model, x, target_class)
        else:
            raise ValueError(f"Unknown explainer: {args.explainer}")

        # IoU vs lesion
        ious = saliency_iou(saliency=cam, lesion_mask=lesion, top_percent=args.iou_top_percent)
        mean_iou = float(ious.mean().detach().cpu().item())

        # Deletion (saliency vs random)
        frac, conf_sal = deletion_curve(
            model=model, x=x, saliency=cam, target_class=target_class, steps=args.deletion_steps, mode="saliency"
        )
        _, conf_rand = deletion_curve(
            model=model, x=x, saliency=cam, target_class=target_class, steps=args.deletion_steps, mode="random"
        )
        auc_sal = auc(frac, conf_sal)
        auc_rand = auc(frac, conf_rand)

        # Save curve plot
        plt.figure(figsize=(6, 4))
        plt.plot(frac, conf_sal, label=f"{tag}: delete-by-saliency")
        plt.plot(frac, conf_rand, label=f"{tag}: random delete", linestyle="--")
        plt.xlabel("Fraction deleted")
        plt.ylabel("Mean confidence (target class)")
        plt.title(f"Deletion curve ({tag})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"deletion_curve_{tag}.png"), dpi=160)
        plt.close()

        # Save some qualitative examples
        k = min(args.qual_examples, n_eval)
        save_qualitative_examples(
            out_path=os.path.join(out_dir, f"qual_{tag}.png"),
            images=x[:k].detach().cpu(),
            cams=cam[:k].detach().cpu(),
            lesion_masks=lesion[:k].detach().cpu(),
            title=f"{tag} ({args.explainer}, target={args.explain_target})",
        )

        return {
            "mean_iou": mean_iou,
            "deletion_auc_saliency": float(auc_sal),
            "deletion_auc_random": float(auc_rand),
        }

    print("Computing XAI metrics on ID...")
    base_id = eval_xai_metrics(model_base, test_id_ds, tag="baseline_id")
    mitig_id = eval_xai_metrics(model_mitig, test_id_ds, tag="mitigated_id")

    print("Computing XAI metrics on OOD...")
    base_ood = eval_xai_metrics(model_base, test_ood_ds, tag="baseline_ood")
    mitig_ood = eval_xai_metrics(model_mitig, test_ood_ds, tag="mitigated_ood")

    rows = [
        {
            "model": "baseline",
            "mitigation": "none",
            "explainer": args.explainer,
            "target": args.explain_target,
            "acc_id": acc_base_id,
            "acc_ood": acc_base_ood,
            **{f"id_{k}": v for k, v in base_id.items()},
            **{f"ood_{k}": v for k, v in base_ood.items()},
        },
        {
            "model": "mitigated",
            "mitigation": args.mitigation,
            "explainer": args.explainer,
            "target": args.explain_target,
            "acc_id": acc_mitig_id,
            "acc_ood": acc_mitig_ood,
            **{f"id_{k}": v for k, v in mitig_id.items()},
            **{f"ood_{k}": v for k, v in mitig_ood.items()},
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    print(f"Saved: {os.path.join(out_dir, 'metrics_summary.csv')}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shortcut learning + XAI faithfulness toy experiment")

    p.add_argument("--out-dir", type=str, default="results_shortcut_xai")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-test", type=int, default=1000)

    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--lesion-radius", type=int, default=5)
    p.add_argument("--logo-size", type=int, default=8)
    p.add_argument("--noise-std", type=float, default=0.25)

    p.add_argument("--spurious-corr", type=float, default=0.95, help="P(logo=1|y=1) in correlated splits")
    p.add_argument(
        "--ood-mode",
        type=str,
        default="random",
        choices=["random", "removed"],
        help="OOD split logo behavior",
    )

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument(
        "--mitigation",
        type=str,
        default="cutout",
        choices=["cutout", "randomize"],
        help="Training-time mitigation to break shortcut",
    )

    p.add_argument(
        "--explainer",
        type=str,
        default="gradcam",
        choices=["gradcam", "input_grad"],
        help="Explanation method for faithfulness evaluation",
    )
    p.add_argument(
        "--explain-target",
        type=str,
        default="pred",
        choices=["pred", "true"],
        help="Explain predicted or true label",
    )

    p.add_argument("--eval-samples", type=int, default=256, help="How many test samples to run XAI metrics on")
    p.add_argument("--qual-examples", type=int, default=6)

    p.add_argument("--deletion-steps", type=int, default=20)
    p.add_argument(
        "--iou-top-percent",
        type=float,
        default=0.10,
        help="Top fraction of pixels to threshold saliency for IoU",
    )

    p.add_argument("--regen-data", action="store_true", help="Regenerate synthetic data even if cache exists")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_experiment(args)
