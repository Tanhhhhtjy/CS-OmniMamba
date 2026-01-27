from datetime import datetime
from typing import Iterable, List, Tuple

from .data_match import SampleRecord

from .config import TrainingConfig


def split_by_time(
    times: Iterable[datetime],
    cfg: TrainingConfig = TrainingConfig(),
) -> Tuple[List[datetime], List[datetime], List[datetime]]:
    train, val, test = [], [], []
    for t in times:
        if cfg.train_start <= t <= cfg.train_end:
            train.append(t)
        elif cfg.val_start <= t <= cfg.val_end:
            val.append(t)
        elif cfg.test_start <= t <= cfg.test_end:
            test.append(t)
    return train, val, test


def split_records(
    records: Iterable[SampleRecord],
    cfg: TrainingConfig = TrainingConfig(),
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    train, val, test = [], [], []
    for record in records:
        t = record.timestamp
        if cfg.train_start <= t <= cfg.train_end:
            train.append(record)
        elif cfg.val_start <= t <= cfg.val_end:
            val.append(record)
        elif cfg.test_start <= t <= cfg.test_end:
            test.append(record)
    return train, val, test
