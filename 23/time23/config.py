from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    model_name: str
    num_classes: int

    epochs: int
    batch_size: int

    lr: float
    momentum: float
    weight_decay: float

    max_grad_norm: float
    delta: float

    device: str
    num_workers: int

    train_subset: int
    query_subset: int
    topk: int
    tracin_checkpoints: str

    run_counterfactual: bool
    counterfactual_mode: str
    counterfactual_steps: int
    counterfactual_repeats: int
    counterfactual_lr: float
    counterfactual_max_queries: int

    download: bool
