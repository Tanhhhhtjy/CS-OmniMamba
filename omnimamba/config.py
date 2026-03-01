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
    weight_decay: float = 5e-4
    # Cosine-restart period (epochs). patience must be >= lr_scheduler_T0
    lr_scheduler_T0: int = 50
    # Early-stopping robustness: monitor EMA(val_loss) with a minimum meaningful gain
    val_ema_alpha: float = 0.2
    early_stop_min_delta: float = 1e-4
    early_stop_use_ema: bool = True
    num_workers: int = 0 if os.name == "nt" else 4
    # Radar temporal sequence: 12 frames x 6 min = 66 min history
    # Increase to 20 (114 min) when GPU memory allows
    radar_seq_len: int = 12

    # Date Splitting Configuration
    split_purge_gap_minutes: int = 360  # 6 hours purge gap to prevent data leakage

    train_start: datetime = datetime(2023, 4, 30, 23, 0, 0)
    train_end: datetime = datetime(
        2023, 8, 20, 23, 59, 59
    )  # Base train period covering up to test

    # Selected 4x5-day windows to balance the heavy event ratio exactly with train (16.0%)
    val_windows: tuple = (
        (datetime(2023, 5, 1, 0, 0, 0), datetime(2023, 5, 5, 23, 59, 59)),
        (datetime(2023, 5, 6, 0, 0, 0), datetime(2023, 5, 10, 23, 59, 59)),
        (datetime(2023, 6, 25, 0, 0, 0), datetime(2023, 6, 29, 23, 59, 59)),
        (datetime(2023, 8, 4, 0, 0, 0), datetime(2023, 8, 8, 23, 59, 59)),
    )

    # Legacy val settings kept for compatibility if needed, but splits.py will use val_windows
    val_start: datetime = datetime(2023, 7, 31, 0, 0, 0)
    val_end: datetime = datetime(2023, 8, 20, 23, 59, 59)

    test_start: datetime = datetime(2023, 8, 21, 0, 0, 0)
    test_end: datetime = datetime(2023, 8, 31, 23, 59, 59)
