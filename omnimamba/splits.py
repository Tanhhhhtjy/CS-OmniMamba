from datetime import datetime
from typing import Iterable, List, Tuple

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
