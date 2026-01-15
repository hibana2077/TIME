from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from time23.models import build_model, forward_features_and_logits, get_classifier_linear


def _select_checkpoints(checkpoints_dir: Path, which: str) -> List[Path]:
    ckpts = sorted(checkpoints_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    which = which.strip().lower()
    if which == "all":
        return ckpts
    if which == "last5":
        return ckpts[-5:]

    # comma-separated epoch numbers
    parts = [p.strip() for p in which.split(",") if p.strip()]
    epochs = set(int(p) for p in parts)
    out: List[Path] = []
    for p in ckpts:
        name = p.stem  # epoch_001
        epoch = int(name.split("_")[-1])
        if epoch in epochs:
            out.append(p)
    if not out:
        raise ValueError(f"No checkpoints matched '{which}' in {checkpoints_dir}")
    return out


@torch.no_grad()
def _collect_features_and_a(
    model: nn.Module,
    loader,
    device: str,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    feats_list: List[np.ndarray] = []
    a_list: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        feats, logits = forward_features_and_logits(model, x)
        feats = feats.view(feats.shape[0], -1)

        probs = torch.softmax(logits, dim=1)
        a = probs
        a[torch.arange(a.shape[0], device=device), y] -= 1.0

        feats_list.append(feats.detach().cpu().numpy().astype(np.float32))
        a_list.append(a.detach().cpu().numpy().astype(np.float32))

    H = np.concatenate(feats_list, axis=0)
    A = np.concatenate(a_list, axis=0)

    if A.shape[1] != num_classes:
        raise ValueError(f"Unexpected num_classes: A has {A.shape[1]} columns, expected {num_classes}")

    return H, A


def compute_tracin_attributions_last_layer(
    checkpoints_dir: Path,
    train_subset_loader,
    query_loader,
    device: str,
    model_name: str,
    num_classes: int,
    which_checkpoints: str = "last5",
    query_chunk_size: int = 32,
) -> np.ndarray:
    """Compute TracIn scores using only last-layer (Linear head) gradients.

    Returns a numpy array of shape [num_queries, num_train_subset].

    Score per checkpoint k is:
      lr_k * <grad_theta L(z_train), grad_theta L(z_query)>

    Here theta is restricted to the classifier head (W,b), and we compute the
    dot product analytically using (features, probs-onehot) to avoid per-sample autograd.
    """

    ckpt_paths = _select_checkpoints(checkpoints_dir, which_checkpoints)

    # Determine sizes from loaders
    num_train = len(train_subset_loader.dataset)
    num_queries = len(query_loader.dataset)

    total_scores = np.zeros((num_queries, num_train), dtype=np.float32)

    for ckpt_path in ckpt_paths:
        # PyTorch 2.6 changed torch.load default weights_only=True, which can fail
        # for our self-generated checkpoints that include non-tensor metadata.
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        lr = float(ckpt.get("lr", 1.0))

        model = build_model(model_name, num_classes)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device)
        model.eval()

        # Validate classifier is Linear (we only need it conceptually; features are from pre-logits)
        _ = get_classifier_linear(model)

        H_train, A_train = _collect_features_and_a(model, train_subset_loader, device, num_classes)
        H_q, A_q = _collect_features_and_a(model, query_loader, device, num_classes)

        # Convert to torch for efficient matmul on device if possible
        # But keep memory modest: do query chunks.
        Ht = torch.from_numpy(H_train).to(device)
        At = torch.from_numpy(A_train).to(device)

        for q0 in range(0, num_queries, query_chunk_size):
            q1 = min(num_queries, q0 + query_chunk_size)
            Hq = torch.from_numpy(H_q[q0:q1]).to(device)
            Aq = torch.from_numpy(A_q[q0:q1]).to(device)

            # dot_feat: [train, q]
            dot_feat = Ht @ Hq.T
            # dot_a: [train, q]
            dot_a = At @ Aq.T

            scores_tq = (dot_feat + 1.0) * dot_a  # includes bias-gradient dot

            # write into [q, train]
            total_scores[q0:q1, :] += (lr * scores_tq.T).detach().cpu().numpy().astype(np.float32)

        del model
        torch.cuda.empty_cache()

    return total_scores
