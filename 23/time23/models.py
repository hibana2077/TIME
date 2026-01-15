from typing import Tuple

import types

import timm
import torch
import torch.nn as nn
from opacus.validators import ModuleValidator


def _basicblock_forward_no_inplace_add(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[no-untyped-def]
    shortcut = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.drop_block(x)
    x = self.act1(x)
    x = self.aa(x)

    x = self.conv2(x)
    x = self.bn2(x)

    if self.se is not None:
        x = self.se(x)

    if self.drop_path is not None:
        x = self.drop_path(x)

    if self.downsample is not None:
        shortcut = self.downsample(shortcut)
    x = x + shortcut
    x = self.act2(x)

    return x


def _bottleneck_forward_no_inplace_add(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[no-untyped-def]
    shortcut = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.drop_block(x)
    x = self.act2(x)
    x = self.aa(x)

    x = self.conv3(x)
    x = self.bn3(x)

    if self.se is not None:
        x = self.se(x)

    if self.drop_path is not None:
        x = self.drop_path(x)

    if self.downsample is not None:
        shortcut = self.downsample(shortcut)
    x = x + shortcut
    x = self.act3(x)

    return x


def _patch_timm_resnet_inplace_add(model: nn.Module) -> int:
    """Patch timm ResNet blocks to avoid in-place residual adds.

    Opacus wraps modules with custom autograd Functions for per-sample gradients.
    timm ResNet blocks use `x += shortcut`, which can trigger:
    "... is a view and is being modified inplace" when used with Opacus.
    """

    # Import lazily so non-ResNet models don't require this symbol.
    from timm.models.resnet import BasicBlock, Bottleneck  # type: ignore

    patched = 0
    for m in model.modules():
        if isinstance(m, BasicBlock):
            m.forward = types.MethodType(_basicblock_forward_no_inplace_add, m)  # type: ignore[method-assign]
            patched += 1
        elif isinstance(m, Bottleneck):
            m.forward = types.MethodType(_bottleneck_forward_no_inplace_add, m)  # type: ignore[method-assign]
            patched += 1

    return patched


def _disable_inplace_ops(model: nn.Module) -> int:
    """Disables in-place ops (e.g., ReLU(inplace=True)).

    Opacus wraps modules with custom autograd Functions for per-sample gradients.
    In-place ops on views can trigger: "... is a view and is being modified inplace".
    """

    changed = 0
    for m in model.modules():
        inplace = getattr(m, "inplace", None)
        if isinstance(inplace, bool) and inplace:
            m.inplace = False
            changed += 1
    return changed


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    # Opacus DP-SGD cannot train with BatchNorm. Replace unsupported modules (e.g., BN -> GN)
    # to keep per-sample gradients privacy-safe.
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
        errors_after = ModuleValidator.validate(model, strict=False)
        if errors_after:
            raise RuntimeError(
                "Model still has DP-incompatible modules after fix: "
                + ", ".join(type(e).__name__ for e in errors_after)
            )

    # Important for Opacus: avoid in-place activations (common in timm ResNets).
    _disable_inplace_ops(model)

    # timm ResNet blocks also use in-place residual adds (x += shortcut), which Opacus can't handle.
    _patch_timm_resnet_inplace_add(model)

    return model


@torch.no_grad()
def forward_features_and_logits(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (features, logits).

    Features are the pre-logits representation used by the classifier head.
    """

    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        feats = model.forward_features(x)
        pre_logits = model.forward_head(feats, pre_logits=True)
        logits = model.forward_head(feats, pre_logits=False)
        return pre_logits, logits

    # Fallback: run full forward; no access to pre-logits
    logits = model(x)
    return logits, logits


@torch.no_grad()
def extract_pre_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extracts the pre-logits representation used by the classifier head."""

    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        feats = model.forward_features(x)
        pre_logits = model.forward_head(feats, pre_logits=True)
        return pre_logits.view(pre_logits.shape[0], -1)

    # Fallback: no access; use logits as features
    logits = model(x)
    return logits.view(logits.shape[0], -1)


def get_classifier_linear(model: nn.Module) -> nn.Linear:
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, nn.Linear):
            return head

    # Common ResNet naming
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc

    raise ValueError("Could not find a Linear classifier head for last-layer TracIn")
