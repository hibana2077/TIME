import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from tqdm import tqdm

from time23.config import ExperimentConfig
from time23.models import build_model
from time23.utils import ensure_dir


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> Dict[str, float]:
    model.eval()
    total_acc = 0.0
    total_n = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().sum().item()
        total_loss += loss
        total_acc += acc
        total_n += y.shape[0]
    return {
        "test_loss": total_loss / max(total_n, 1),
        "test_acc": total_acc / max(total_n, 1),
    }


def train_one_run(
    run_dir: Path,
    trainloader,
    testloader,
    config: ExperimentConfig,
    epsilon: Optional[float],
    seed: int,
) -> Dict[str, float]:
    ensure_dir(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    device = config.device
    model = build_model(config.model_name, config.num_classes).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss()

    privacy_engine = None
    noise_multiplier = None
    effective_epsilon = float("inf")

    if epsilon is not None:
        sample_rate = config.batch_size / len(trainloader.dataset)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=config.delta,
            sample_rate=sample_rate,
            epochs=config.epochs,
            accountant="rdp",
        )

        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, trainloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=config.max_grad_norm,
        )

    # Simple LR schedule: cosine
    def lr_at_epoch(epoch: int) -> float:
        t = epoch / max(config.epochs, 1)
        return config.lr * 0.5 * (1.0 + np.cos(np.pi * t))

    for epoch in range(1, config.epochs + 1):
        model.train()
        lr = lr_at_epoch(epoch - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        pbar = tqdm(trainloader, desc=f"train eps={epsilon} seed={seed} epoch={epoch}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()), acc=float(_accuracy(logits, y)))

        if privacy_engine is not None:
            effective_epsilon = privacy_engine.accountant.get_epsilon(delta=config.delta)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "lr": lr,
            "seed": seed,
            "target_epsilon": epsilon,
            "effective_epsilon": effective_epsilon,
            "delta": config.delta,
            "noise_multiplier": noise_multiplier,
        }
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")

    metrics = evaluate(model, testloader, device)

    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "target_epsilon": epsilon,
                "effective_epsilon": effective_epsilon,
                "delta": config.delta,
                "noise_multiplier": noise_multiplier,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "acc": metrics["test_acc"],
        "loss": metrics["test_loss"],
        "effective_epsilon": float(effective_epsilon),
        "noise_multiplier": float(noise_multiplier) if noise_multiplier is not None else np.nan,
    }
