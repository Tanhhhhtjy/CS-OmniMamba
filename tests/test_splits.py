from datetime import datetime

from omnimamba.splits import split_by_time


def test_split_by_time_ranges():
    samples = [
        datetime(2023, 7, 30, 12, 0),
        datetime(2023, 7, 31, 12, 0),
        datetime(2023, 8, 20, 12, 0),
    ]
    train, val, test = split_by_time(samples)
    assert datetime(2023, 7, 30, 12, 0) in train
    assert datetime(2023, 7, 31, 12, 0) in val
    assert datetime(2023, 8, 20, 12, 0) in test
