from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


@dataclass(frozen=True)
class TabularDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def load_mnist_torch(batch_size: int = 256, num_workers: int = 2):
    """MNIST dataloaders (torchvision). Returns (train_loader, test_loader)."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, test_loader


SklearnTabularName = Literal["breast_cancer", "iris", "wine", "digits", "adult_openml"]


def load_tabular_sklearn(name: SklearnTabularName, test_size: float = 0.2, seed: int = 0) -> TabularDataset:
    """Loads common tabular datasets via sklearn.

    Notes:
    - "adult_openml" downloads from OpenML; first run requires internet.
    """
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if name == "breast_cancer":
        ds = load_breast_cancer()
        x = ds.data.astype(np.float32)
        y = ds.target.astype(np.int64)
        feature_names = list(ds.feature_names)
        target_names = [str(t) for t in ds.target_names]
    elif name == "iris":
        ds = load_iris()
        x = ds.data.astype(np.float32)
        y = ds.target.astype(np.int64)
        feature_names = list(ds.feature_names)
        target_names = [str(t) for t in ds.target_names]
    elif name == "wine":
        ds = load_wine()
        x = ds.data.astype(np.float32)
        y = ds.target.astype(np.int64)
        feature_names = list(ds.feature_names)
        target_names = [str(t) for t in ds.target_names]
    elif name == "digits":
        ds = load_digits()
        x = ds.data.astype(np.float32)
        y = ds.target.astype(np.int64)
        feature_names = [f"pixel_{i}" for i in range(x.shape[1])]
        target_names = [str(i) for i in range(10)]
    elif name == "adult_openml":
        # OpenML Adult: mixed types. We'll do basic one-hot encoding.
        adult = fetch_openml(name="adult", version=2, as_frame=True)
        frame = adult.frame
        y_raw = frame["class"]
        x_frame = frame.drop(columns=["class"])
        x_frame = x_frame.replace({"?": np.nan})
        x_frame = x_frame.fillna("missing")
        x = np.asarray(
            __one_hot_frame(x_frame),
            dtype=np.float32,
        )
        y = (y_raw == ">50K").astype(np.int64).to_numpy()
        feature_names = [f"f{i}" for i in range(x.shape[1])]
        target_names = ["<=50K", ">50K"]
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    return TabularDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
    )


def __one_hot_frame(df):
    import pandas as pd

    # One-hot encode categoricals; keep numerics.
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    num_cols = [c for c in df.columns if c not in cat_cols]

    out = []
    if num_cols:
        out.append(df[num_cols].astype("float32", errors="ignore"))
    if cat_cols:
        out.append(pd.get_dummies(df[cat_cols], dummy_na=False))

    return pd.concat(out, axis=1)
