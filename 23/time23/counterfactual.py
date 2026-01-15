from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from time23.models import build_model, extract_pre_logits, get_classifier_linear


@dataclass
class CounterfactualResult:
    delta_mean: float
    effect_top_mean: float
    effect_rand_mean: float


def _topk_indices(v: np.ndarray, k: int) -> np.ndarray:
    if k >= v.size:
        return np.argsort(-v)
    idx = np.argpartition(-v, kth=k - 1)[:k]
    idx = idx[np.argsort(-v[idx])]
    return idx


@torch.no_grad()
def _true_class_prob(model: nn.Module, head: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pre = extract_pre_logits(model, x)
    logits = head(pre)
    probs = torch.softmax(logits, dim=1)
    return probs[torch.arange(y.shape[0], device=y.device), y]


def _finetune_head_steps(
    model: nn.Module,
    head: nn.Linear,
    train_loader,
    device: str,
    steps: int,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> None:
    head.train()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    done = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            pre = extract_pre_logits(model, x)

        logits = head(pre)
        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        done += 1
        if done >= steps:
            break


def compute_counterfactual_finetune_delta(
    checkpoint_path: Path,
    model_name: str,
    num_classes: int,
    attribution_scores: np.ndarray,
    train_subset_ds,
    query_subset_ds,
    batch_size: int,
    num_workers: int,
    device: str,
    topk: int,
    repeats: int,
    steps: int,
    finetune_lr: float,
    weight_decay: float,
    momentum: float,
    rng_seed: int,
    max_queries: Optional[int] = None,
) -> CounterfactualResult:
    """Counterfactual removal test (checkpoint + head-only finetune).

    For each query q:
      - Remove Top-k training points (within the TracIn train subset)
      - Fine-tune ONLY the classifier head for `steps` minibatches
      - Measure effect as |p_true(after) - p_true(before)|
      - Compare vs Random-k (averaged over `repeats`)

    Returns mean delta across queries: effect_top - effect_rand.

    Notes:
    - This is much closer to inst.md than the proxy, but still an approximation:
      we fine-tune only the last layer (fast) instead of full retraining.
    """

    rng = np.random.default_rng(int(rng_seed))

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    base_state = ckpt["model_state"]

    model = build_model(model_name, num_classes)
    model.load_state_dict(base_state, strict=True)
    model.to(device)
    model.eval()

    # Freeze backbone; we will only update head weights
    for p in model.parameters():
        p.requires_grad = False

    base_head = get_classifier_linear(model)
    base_head.requires_grad_(True)

    # Prepare query tensors (we need per-query probabilities)
    q_loader = torch.utils.data.DataLoader(
        query_subset_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Cache query probs before for all queries
    p_before = []
    y_all = []
    x_all = []
    for x, y in q_loader:
        x_all.append(x)
        y_all.append(y)
    x_all_t = torch.cat(x_all, dim=0)
    y_all_t = torch.cat(y_all, dim=0)

    with torch.no_grad():
        pb = _true_class_prob(model, base_head, x_all_t.to(device), y_all_t.to(device))
    p_before = pb.detach().cpu().numpy()

    Q, M = attribution_scores.shape
    k = min(int(topk), int(M))
    r = max(1, int(repeats))
    qN = Q
    if max_queries is not None and max_queries > 0:
        qN = min(qN, int(max_queries))

    deltas = []
    effects_top = []
    effects_rand = []

    for q in range(qN):
        v = attribution_scores[q]
        top_local = _topk_indices(v, k)

        # Top-k removal
        keep = [i for i in range(M) if i not in set(top_local.tolist())]
        train_ds_top = torch.utils.data.Subset(train_subset_ds, keep)
        train_loader_top = torch.utils.data.DataLoader(
            train_ds_top,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # Reload fresh model/head for this counterfactual
        model_top = build_model(model_name, num_classes)
        model_top.load_state_dict(base_state, strict=True)
        model_top.to(device)
        model_top.eval()
        for p in model_top.parameters():
            p.requires_grad = False
        head_top = get_classifier_linear(model_top)
        head_top.requires_grad_(True)

        _finetune_head_steps(
            model=model_top,
            head=head_top,
            train_loader=train_loader_top,
            device=device,
            steps=steps,
            lr=finetune_lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        with torch.no_grad():
            pa = _true_class_prob(
                model_top,
                head_top,
                x_all_t[q : q + 1].to(device),
                y_all_t[q : q + 1].to(device),
            )
        effect_top = float(np.abs(float(pa.item()) - float(p_before[q])))

        # Random-k removal (average)
        rand_effects = []
        for _ in range(r):
            rand_local = rng.choice(M, size=k, replace=False)
            keep_r = [i for i in range(M) if i not in set(rand_local.tolist())]
            train_ds_r = torch.utils.data.Subset(train_subset_ds, keep_r)
            train_loader_r = torch.utils.data.DataLoader(
                train_ds_r,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

            model_r = build_model(model_name, num_classes)
            model_r.load_state_dict(base_state, strict=True)
            model_r.to(device)
            model_r.eval()
            for p in model_r.parameters():
                p.requires_grad = False
            head_r = get_classifier_linear(model_r)
            head_r.requires_grad_(True)

            _finetune_head_steps(
                model=model_r,
                head=head_r,
                train_loader=train_loader_r,
                device=device,
                steps=steps,
                lr=finetune_lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )

            with torch.no_grad():
                pr = _true_class_prob(
                    model_r,
                    head_r,
                    x_all_t[q : q + 1].to(device),
                    y_all_t[q : q + 1].to(device),
                )
            rand_effects.append(float(np.abs(float(pr.item()) - float(p_before[q]))))

        effect_rand = float(np.mean(rand_effects))

        deltas.append(effect_top - effect_rand)
        effects_top.append(effect_top)
        effects_rand.append(effect_rand)

    return CounterfactualResult(
        delta_mean=float(np.mean(deltas)) if deltas else float("nan"),
        effect_top_mean=float(np.mean(effects_top)) if effects_top else float("nan"),
        effect_rand_mean=float(np.mean(effects_rand)) if effects_rand else float("nan"),
    )
