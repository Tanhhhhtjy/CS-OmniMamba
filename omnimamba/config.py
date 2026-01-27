from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrainingConfig:
    img_size: int = 66
    img_size_w: int = 70
    patch_size: int = 2
    stride: int = 2
    d_state: int = 32
    dim: int = 128
    depth: int = 4
    batch_size: int = 8
    epochs: int = 1000
    lr: float = 1e-4

    train_start: datetime = datetime(2023, 4, 30, 23, 0, 0)
    train_end: datetime = datetime(2023, 7, 30, 23, 59, 59)
    val_start: datetime = datetime(2023, 7, 31, 0, 0, 0)
    val_end: datetime = datetime(2023, 8, 6, 23, 59, 59)
    test_start: datetime = datetime(2023, 8, 7, 0, 0, 0)
    test_end: datetime = datetime(2023, 8, 31, 23, 59, 59)
