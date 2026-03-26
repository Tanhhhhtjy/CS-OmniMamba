"""
RainDataset and sample index construction.
"""
from __future__ import annotations
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.config import (
    RADAR_PREP_DIR, RAIN_DIR, PWV_DIR,
    H, W, T, STEP_SECONDS,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)
from src.transforms import normalize


# ── Helpers ───────────────────────────────────────────────────────────────────

def _date_to_dir_parts(dt: datetime) -> Tuple[str, str]:
    """Return (month_dir, day_dir) strings matching folder naming convention."""
    return dt.strftime("%Y%m"), dt.strftime("%Y%m%d")


def _timestamp_to_path(base_dir: Path, dt: datetime) -> Path:
    month, day = _date_to_dir_parts(dt)
    fname = dt.strftime("%Y-%m-%d-%H-%M-%S.png")
    return base_dir / month / day / fname


def _collect_timestamps(dirs: List[Path]) -> List[datetime]:
    """
    Collect the intersection of timestamps present in all given directories.
    Returns a sorted list of datetime objects.
    """
    sets = []
    for d in dirs:
        ts_set = set()
        for p in d.rglob("*.png"):
            stem = p.stem  # e.g. "2025-05-01-00-06-00"
            try:
                ts_set.add(datetime.strptime(stem, "%Y-%m-%d-%H-%M-%S"))
            except ValueError:
                pass
        sets.append(ts_set)
    common = sorted(sets[0].intersection(*sets[1:]))
    return common


def build_sample_index(timestamps: List[datetime], T: int = T) -> List[int]:
    """
    Given a sorted list of datetime objects, return the list of start indices i
    such that timestamps[i : i+T+1] forms an unbroken sequence of T+1 steps
    (each consecutive pair separated by exactly STEP_SECONDS seconds).
    inputs  = timestamps[i : i+T]     # T frames
    target  = timestamps[i+T]          # t+1
    """
    samples = []
    n = len(timestamps)
    for i in range(n - T):
        window = timestamps[i: i + T + 1]  # T+1 = 11 timestamps
        diffs = [
            (window[j + 1] - window[j]).total_seconds()
            for j in range(len(window) - 1)
        ]
        if all(d == float(STEP_SECONDS) for d in diffs):
            samples.append(i)
    return samples


class RainDataset(Dataset):
    """
    PyTorch Dataset for precipitation nowcasting.

    Each sample:
        radar         : Tensor [T, 1, H, W]  — normalised RADAR history (t-T+1 … t)
        pwv           : Tensor [1, 1, H, W]  — normalised PWV at t
        rain          : Tensor [1, 1, H, W]  — normalised RAIN at t+1 (target)
        rain_current  : Tensor [1, 1, H, W]  — normalised RAIN at t (for Persistence baseline)
    """

    def __init__(
        self,
        radar_dir: Path,
        pwv_dir: Path,
        rain_dir: Path,
        T: int = T,
    ) -> None:
        self.radar_dir = Path(radar_dir)
        self.pwv_dir   = Path(pwv_dir)
        self.rain_dir  = Path(rain_dir)
        self.T = T

        self.timestamps = _collect_timestamps([radar_dir, pwv_dir, rain_dir])
        self.indices = build_sample_index(self.timestamps, T=T)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        start = self.indices[idx]
        input_ts = self.timestamps[start: start + self.T]  # T frames
        target_t = self.timestamps[start + self.T]          # t+1

        # Load and normalise RADAR sequence
        radar_frames = []
        for ts in input_ts:
            arr = self._load_gray(self.radar_dir, ts)
            radar_frames.append(normalize(arr))
        radar = np.stack(radar_frames, axis=0)[:, np.newaxis, :, :]  # [T,1,H,W]

        # Load and normalise PWV (current frame = last of input sequence)
        pwv_arr = normalize(self._load_gray(self.pwv_dir, input_ts[-1]))
        pwv = pwv_arr[np.newaxis, np.newaxis, :, :]                   # [1,1,H,W]

        # Load and normalise RAIN target (t+1) and current frame (t)
        rain_arr = normalize(self._load_gray(self.rain_dir, target_t))
        rain = rain_arr[np.newaxis, np.newaxis, :, :]                  # [1,1,H,W]

        rain_curr_arr = normalize(self._load_gray(self.rain_dir, input_ts[-1]))
        rain_current  = rain_curr_arr[np.newaxis, np.newaxis, :, :]    # [1,1,H,W]

        return {
            "radar":        torch.from_numpy(radar),
            "pwv":          torch.from_numpy(pwv),
            "rain":         torch.from_numpy(rain),
            "rain_current": torch.from_numpy(rain_current),
        }

    @staticmethod
    def _load_gray(base_dir: Path, dt: datetime) -> np.ndarray:
        """Load a PNG as uint8 grayscale array of shape (H, W)."""
        path = _timestamp_to_path(base_dir, dt)
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.uint8)
