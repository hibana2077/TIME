from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_cifar10_dataloaders(
    batch_size: int,
    num_workers: int,
    download: bool,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tfms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = datasets.CIFAR10(root=data_dir, train=True, transform=train_tfms, download=download)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, transform=test_tfms, download=download)

    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, testloader, train_ds, test_ds
