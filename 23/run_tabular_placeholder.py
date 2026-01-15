"""Tabular dataset entrypoint (placeholder).

This script is here to show how to extend the same pipeline to sklearn tabular datasets.
MNIST is implemented end-to-end in `run_mnist_pipeline.py`.

You can use `datasets.load_tabular_sklearn()` to load common datasets and then either:
- train a torch model with DP-SGD (similar to MNIST)
- or train sklearn baselines (non-DP) for utility references

If you tell me which tabular dataset you want first (e.g. breast_cancer vs adult),
I can wire the full tabular pipeline the same way as MNIST.
"""

from __future__ import annotations

import argparse

from datasets import load_tabular_sklearn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="breast_cancer")
    args = p.parse_args()

    ds = load_tabular_sklearn(args.dataset)  # type: ignore[arg-type]
    print("Loaded", args.dataset)
    print("train", ds.x_train.shape, "test", ds.x_test.shape)


if __name__ == "__main__":
    main()
