from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import is_inf_epsilon


@dataclass
class TrainResult:
    model: nn.Module
    test_accuracy: float
    epsilon: float
    delta: float
    noise_multiplier: Optional[float]


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    epsilon: float,
    delta: float,
    max_grad_norm: float = 1.0,
) -> TrainResult:
    """Train model with either non-DP (epsilon=inf) or DP-SGD via Opacus.

    For DP training, uses make_private_with_epsilon to compute an appropriate noise multiplier.
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    noise_multiplier: Optional[float] = None

    if not is_inf_epsilon(epsilon):
        from opacus import PrivacyEngine

        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=epsilon,
            target_delta=delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )
        noise_multiplier = float(optimizer.noise_multiplier)

    for _epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc="train", leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    acc = evaluate_accuracy(model, test_loader, device)
    return TrainResult(
        model=model,
        test_accuracy=acc,
        epsilon=epsilon,
        delta=delta,
        noise_multiplier=noise_multiplier,
    )
