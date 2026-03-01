from datetime import datetime, timedelta
from typing import Iterable, List, Tuple

from .data_match import SampleRecord

from .config import TrainingConfig


def _in_any_window(t: datetime, windows: Tuple[Tuple[datetime, datetime], ...]) -> bool:
    for start, end in windows:
        if start <= t <= end:
            return True
    return False


def _is_purged(t: datetime, cfg: TrainingConfig) -> bool:
    gap = timedelta(minutes=cfg.split_purge_gap_minutes)

    if hasattr(cfg, "val_windows") and cfg.val_windows:
        for w_start, w_end in cfg.val_windows:
            if (w_start - gap) <= t <= (w_end + gap):
                return True
    else:
        if (cfg.val_start - gap) <= t <= (cfg.val_end + gap):
            return True

    if (cfg.test_start - gap) <= t <= (cfg.test_end + gap):
        return True

    return False


def split_by_time(
    times: Iterable[datetime],
    cfg: TrainingConfig = TrainingConfig(),
) -> Tuple[List[datetime], List[datetime], List[datetime]]:
    train, val, test = [], [], []
    for t in times:
        is_test = cfg.test_start <= t <= cfg.test_end

        is_val = False
        if hasattr(cfg, "val_windows") and cfg.val_windows:
            is_val = _in_any_window(t, cfg.val_windows)
        else:
            is_val = cfg.val_start <= t <= cfg.val_end

        is_train = False
        if not is_test and not is_val:
            if cfg.train_start <= t <= cfg.train_end:
                if not _is_purged(t, cfg):
                    is_train = True

        if is_train:
            train.append(t)
        elif is_val:
            val.append(t)
        elif is_test:
            test.append(t)

    return train, val, test


def split_records(
    records: Iterable[SampleRecord],
    cfg: TrainingConfig = TrainingConfig(),
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    train, val, test = [], [], []
    for record in records:
        t = record.timestamp
        is_test = cfg.test_start <= t <= cfg.test_end

        is_val = False
        if hasattr(cfg, "val_windows") and cfg.val_windows:
            is_val = _in_any_window(t, cfg.val_windows)
        else:
            is_val = cfg.val_start <= t <= cfg.val_end

        is_train = False
        if not is_test and not is_val:
            if cfg.train_start <= t <= cfg.train_end:
                if not _is_purged(t, cfg):
                    is_train = True

        if is_train:
            train.append(record)
        elif is_val:
            val.append(record)
        elif is_test:
            test.append(record)

    return train, val, test
