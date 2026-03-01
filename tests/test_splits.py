from datetime import datetime

from omnimamba.splits import split_by_time


def test_split_by_time_ranges():
    # Update to align with the new 4x5-day window setup
    samples = [
        datetime(2023, 7, 30, 12, 0),  # Train
        datetime(2023, 6, 26, 12, 0),  # Val (inside 3rd window 6/25-6/29)
        datetime(2023, 8, 25, 12, 0),  # Test
    ]
    train, val, test = split_by_time(samples)
    assert datetime(2023, 7, 30, 12, 0) in train
    assert datetime(2023, 6, 26, 12, 0) in val
    assert datetime(2023, 8, 25, 12, 0) in test
