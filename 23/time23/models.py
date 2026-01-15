from typing import Tuple

import timm
import torch
import torch.nn as nn


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
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
