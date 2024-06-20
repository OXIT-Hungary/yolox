from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    name: str
    out_dir: Path


@dataclass
class ModelConfig:
    num_classes: int
    depth: float
    width: float
    act: str


@dataclass
class DataConfig:
    num_workers: int
    input_size: List[int]
    multiscale_range: int
    data_dir: Optional[Path]
    train_ann: str
    val_ann: str
    test_ann: str


@dataclass
class TransformConfig:
    mosaic_prob: float
    mixup_prob: float
    hsv_prob: float
    flip_prob: float
    degrees: float
    translate: float
    mosaic_scale: List[float]
    enable_mixup: bool
    mixup_scale: List[float]
    shear: float


@dataclass
class TrainingConfig:
    warmup_epochs: int
    max_epoch: int
    warmup_lr: float
    min_lr_ratio: float
    basic_lr_per_img: float
    scheduler: str
    no_aug_epochs: int
    ema: bool
    weight_decay: float
    momentum: float
    print_interval: int
    eval_interval: int
    save_history_ckpt: bool


@dataclass
class TestConfig:
    output_size: List[int]
    confidence: float
    nms_threshold: float


@dataclass
class YoloXConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    dataloader: DataConfig
    transform: TransformConfig
    training: TrainingConfig
    test: TestConfig
