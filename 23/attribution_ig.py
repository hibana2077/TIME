from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class IGResult:
    attribution_vector: np.ndarray  # shape (D,)
    n_samples: int


@torch.no_grad()
def _pick_targets(logits: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "true":
        return y
    if mode == "pred":
        return torch.argmax(logits, dim=1)
    raise ValueError("mode must be 'true' or 'pred'")


def integrated_gradients_average(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    n_batches: int = 10,
    steps: int = 32,
    baseline: float = 0.0,
    target_mode: str = "true",
    use_abs: bool = True,
) -> IGResult:
    """Computes an average feature attribution vector using Integrated Gradients.

    - For each batch, uses a zero baseline by default.
    - Target is the true label (default) or predicted label.
    - Returns mean attribution across sampled batches.

    Output vector is flattened (D,) where D=784 for MNIST.
    """
    model.eval()

    def _looks_like_opacus_wrapped(m: nn.Module) -> bool:
        # Opacus wraps models in GradSampleModule and adds forward/backward hooks
        # for per-sample gradient accounting. Those hooks can break attribution
        # methods that call torch.autograd.grad on inputs.
        mod = type(m).__module__
        name = type(m).__name__
        return ("opacus" in mod) or (name == "GradSampleModule")

    @contextmanager
    def _temporarily_disable_all_hooks(m: nn.Module):
        # Snapshot & clear hooks on all submodules, then restore.
        saved = []
        for sm in m.modules():
            saved.append(
                (
                    sm,
                    dict(getattr(sm, "_forward_hooks", {})),
                    dict(getattr(sm, "_forward_pre_hooks", {})),
                    dict(getattr(sm, "_backward_hooks", {})),
                    dict(getattr(sm, "_backward_pre_hooks", {})),
                    dict(getattr(sm, "_full_backward_hooks", {})),
                )
            )
            if hasattr(sm, "_forward_hooks"):
                sm._forward_hooks.clear()
            if hasattr(sm, "_forward_pre_hooks"):
                sm._forward_pre_hooks.clear()
            if hasattr(sm, "_backward_hooks"):
                sm._backward_hooks.clear()
            if hasattr(sm, "_backward_pre_hooks"):
                sm._backward_pre_hooks.clear()
            if hasattr(sm, "_full_backward_hooks"):
                sm._full_backward_hooks.clear()
        try:
            yield
        finally:
            for (
                sm,
                fwd,
                fwd_pre,
                bwd,
                bwd_pre,
                full_bwd,
            ) in saved:
                if hasattr(sm, "_forward_hooks"):
                    sm._forward_hooks.clear()
                    sm._forward_hooks.update(fwd)
                if hasattr(sm, "_forward_pre_hooks"):
                    sm._forward_pre_hooks.clear()
                    sm._forward_pre_hooks.update(fwd_pre)
                if hasattr(sm, "_backward_hooks"):
                    sm._backward_hooks.clear()
                    sm._backward_hooks.update(bwd)
                if hasattr(sm, "_backward_pre_hooks"):
                    sm._backward_pre_hooks.clear()
                    sm._backward_pre_hooks.update(bwd_pre)
                if hasattr(sm, "_full_backward_hooks"):
                    sm._full_backward_hooks.clear()
                    sm._full_backward_hooks.update(full_bwd)

    hook_ctx = _temporarily_disable_all_hooks(model) if _looks_like_opacus_wrapped(model) else nullcontext()

    with hook_ctx:
        total_attr: Optional[torch.Tensor] = None
        total_n = 0

        batch_count = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Ensure gradient tracking on input
            x = x.detach()
            baseline_x = torch.full_like(x, fill_value=baseline)

            # Determine targets from current model logits
            with torch.no_grad():
                logits0 = model(x)
                targets = _pick_targets(logits0, y, target_mode)

            # Integrated gradients accumulation
            attr_batch = torch.zeros_like(x)
            for s in range(1, steps + 1):
                alpha = float(s) / float(steps)
                xi = baseline_x + alpha * (x - baseline_x)
                xi.requires_grad_(True)

                logits = model(xi)
                sel = logits.gather(1, targets.view(-1, 1)).squeeze(1).sum()

                grads = torch.autograd.grad(sel, xi, retain_graph=False, create_graph=False)[0]
                attr_batch = attr_batch + grads

            attr_batch = (x - baseline_x) * (attr_batch / float(steps))
            if use_abs:
                attr_batch = attr_batch.abs()

            # Average across batch dimension to get a single vector
            attr_vec = attr_batch.view(attr_batch.size(0), -1).mean(dim=0)

            if total_attr is None:
                total_attr = attr_vec.detach().cpu()
            else:
                total_attr = total_attr + attr_vec.detach().cpu()

            total_n += 1
            batch_count += 1
            if batch_count >= n_batches:
                break

    if total_attr is None:
        raise RuntimeError("No batches were processed for IG attribution")

    mean_attr = (total_attr / float(total_n)).numpy().astype(np.float64)
    return IGResult(attribution_vector=mean_attr, n_samples=total_n)
