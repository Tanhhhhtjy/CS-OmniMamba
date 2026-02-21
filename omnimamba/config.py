from dataclasses import dataclass
from datetime import datetime
import os


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
    # L2 regularisation – reduces overfitting on small validation sets
    weight_decay: float = 1e-3
    # Cosine-restart period (epochs). patience must be >= lr_scheduler_T0
    lr_scheduler_T0: int = 30
    num_workers: int = 0 if os.name == "nt" else 4
    # Radar temporal sequence: 12 frames x 6 min = 66 min history
    # Increase to 20 (114 min) when GPU memory allows
    radar_seq_len: int = 12

    train_start: datetime = datetime(2023, 4, 30, 23, 0, 0)
    train_end: datetime = datetime(2023, 7, 30, 23, 59, 59)
    # Validation window extended to ~3 weeks to reduce loss-estimate variance
    val_start: datetime = datetime(2023, 7, 31, 0, 0, 0)
    val_end: datetime = datetime(2023, 8, 20, 23, 59, 59)
    test_start: datetime = datetime(2023, 8, 21, 0, 0, 0)
    test_end: datetime = datetime(2023, 8, 31, 23, 59, 59)
