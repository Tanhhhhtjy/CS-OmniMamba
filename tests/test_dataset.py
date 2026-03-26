# tests/test_dataset.py
from datetime import datetime, timedelta
from src.dataset import build_sample_index

def _make_timestamps(start: datetime, n: int, step_s: int = 360):
    """Helper: generate n consecutive timestamps at step_s-second intervals."""
    return [start + timedelta(seconds=i * step_s) for i in range(n)]

def test_continuous_sequence_yields_samples():
    ts = _make_timestamps(datetime(2025, 5, 1, 0, 0), 15)
    samples = build_sample_index(ts, T=10)
    # With 15 timestamps and T=10, we need windows of 11: indices 0..4 are valid
    assert len(samples) == 5

def test_gap_breaks_window():
    """A missing timestamp in the middle should invalidate overlapping windows."""
    ts = _make_timestamps(datetime(2025, 5, 1, 0, 0), 20)
    # Remove timestamp at index 10, creating a 12-minute gap
    ts.pop(10)
    samples = build_sample_index(ts, T=10)
    # Windows that span index 10 must be excluded
    for idx in samples:
        window = ts[idx: idx + 11]
        diffs = [(window[j+1] - window[j]).total_seconds() for j in range(10)]
        assert all(d == 360.0 for d in diffs), f"Window {idx} spans a gap"

def test_empty_when_too_few_timestamps():
    ts = _make_timestamps(datetime(2025, 5, 1), 10)  # exactly T, need T+1
    samples = build_sample_index(ts, T=10)
    assert samples == []

def test_sample_index_returns_correct_target_position():
    ts = _make_timestamps(datetime(2025, 5, 1), 12)
    samples = build_sample_index(ts, T=10)
    assert len(samples) == 2
    idx = samples[0]
    # inputs: ts[idx:idx+10], target: ts[idx+10]
    assert ts[idx + 10] - ts[idx + 9] == timedelta(seconds=360)


import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile, os
from PIL import Image
from src.dataset import RainDataset

def _make_fake_dataset(tmp_path: Path, n_days: int = 1, n_frames: int = 15):
    """
    Create minimal fake PNG files in the expected directory structure:
      tmp_path/{modality}/202505/20250501/2025-05-01-HH-MM-SS.png
    Returns (radar_prep_dir, pwv_dir, rain_dir).
    """
    from datetime import datetime, timedelta
    radar_dir = tmp_path / "radar"
    pwv_dir   = tmp_path / "pwv"
    rain_dir  = tmp_path / "rain"

    start = datetime(2025, 5, 1, 0, 0, 0)
    for i in range(n_frames):
        dt = start + timedelta(seconds=i * 360)
        month = dt.strftime("%Y%m")
        day   = dt.strftime("%Y%m%d")
        fname = dt.strftime("%Y-%m-%d-%H-%M-%S.png")
        for d in [radar_dir, pwv_dir, rain_dir]:
            p = d / month / day / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            # 66×70 single-channel grayscale PNG
            arr = np.full((66, 70), i * 10, dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(p)

    return radar_dir, pwv_dir, rain_dir


def test_dataset_length(tmp_path):
    radar_dir, pwv_dir, rain_dir = _make_fake_dataset(tmp_path, n_frames=15)
    ds = RainDataset(radar_dir, pwv_dir, rain_dir, T=10)
    # 15 timestamps, T=10 → build_sample_index uses range(n-T)=range(5) → indices 0..4
    # All 5 windows are valid continuous 11-step sequences
    assert len(ds) == 5


def test_getitem_shapes(tmp_path):
    radar_dir, pwv_dir, rain_dir = _make_fake_dataset(tmp_path, n_frames=15)
    ds = RainDataset(radar_dir, pwv_dir, rain_dir, T=10)
    sample = ds[0]
    assert "radar" in sample and "pwv" in sample and "rain" in sample and "rain_current" in sample
    assert sample["radar"].shape        == (10, 1, 66, 70)
    assert sample["pwv"].shape          == (1,  1, 66, 70)
    assert sample["rain"].shape         == (1,  1, 66, 70)
    assert sample["rain_current"].shape == (1,  1, 66, 70)


def test_getitem_dtype_and_range(tmp_path):
    radar_dir, pwv_dir, rain_dir = _make_fake_dataset(tmp_path, n_frames=15)
    ds = RainDataset(radar_dir, pwv_dir, rain_dir, T=10)
    sample = ds[0]
    for key in ["radar", "pwv", "rain", "rain_current"]:
        t = sample[key]
        assert t.dtype == torch.float32, f"{key} dtype wrong"
        assert t.min() >= 0.0 and t.max() <= 1.0, f"{key} out of [0,1]"


def test_getitem_normalisation_direction(tmp_path):
    """Pixel 0 (max signal) → value 1.0; pixel 255 → 0.0."""
    radar_dir, pwv_dir, rain_dir = _make_fake_dataset(tmp_path, n_frames=15)
    # Overwrite first rain file with all-zeros (max signal)
    from datetime import datetime
    dt = datetime(2025, 5, 1, 1, 0, 0)  # t+1 for window starting at 0
    p = rain_dir / dt.strftime("%Y%m") / dt.strftime("%Y%m%d") / dt.strftime("%Y-%m-%d-%H-%M-%S.png")
    arr = np.zeros((66, 70), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(p)
    ds = RainDataset(radar_dir, pwv_dir, rain_dir, T=10)
    sample = ds[0]
    np.testing.assert_allclose(sample["rain"].numpy(), 1.0)
