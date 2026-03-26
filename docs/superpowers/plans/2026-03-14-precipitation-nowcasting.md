# Precipitation Nowcasting — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch data pipeline and training harness for Seq→1 precipitation nowcasting using RADAR (10-frame history) + PWV (single background frame) → RAIN(t+1).

**Architecture:** Pure-Python data pipeline (no model architecture decisions yet) with a `RainDataset` PyTorch Dataset, offline RADAR preprocessing script, weighted-MSE loss, and an evaluation suite (CSI/POD/FAR/MSE/MAE — Phase 1). SSIM/FSS are deferred to Phase 2 (post-baseline model comparison). A stub model (Conv2D over flattened time-channel dim) is included only to make the training loop runnable end-to-end; the model is replaceable by swapping one class.

**Tech Stack:** Python 3.10+, PyTorch 2.x, Pillow, NumPy, pytest, tqdm

---

## File Structure

```
e:\CS-OmniMamba-2025\
├── src/
│   ├── __init__.py
│   ├── config.py              # All paths, constants, hyperparameters in one place
│   ├── preprocess.py          # Offline RADAR downsample script (run once)
│   ├── dataset.py             # RainDataset (sample index + __getitem__)
│   ├── transforms.py          # normalize() and helpers
│   ├── loss.py                # weighted_mse_loss
│   ├── metrics.py             # CSI, POD, FAR, SSIM, FSS
│   ├── baselines.py           # Persistence and Zero baselines
│   └── train.py               # Training loop + eval loop
├── tests/
│   ├── __init__.py
│   ├── test_transforms.py
│   ├── test_dataset.py
│   ├── test_loss.py
│   ├── test_metrics.py
│   └── test_baselines.py
├── scripts/
│   └── run_preprocess.py      # Entry point: python scripts/run_preprocess.py
├── requirements.txt
└── docs/  (existing)
```

---

## Chunk 1: Project Setup & Transforms

### Task 1: Project skeleton and requirements

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `scripts/__init__.py` (empty)

- [ ] **Step 1: Create `requirements.txt`**

```text
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
numpy>=1.24.0
pytest>=7.4.0
tqdm>=4.65.0
pytorch-msssim>=0.2.1
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 3: Create empty `__init__.py` files**

Create `src/__init__.py`, `tests/__init__.py`, `scripts/__init__.py` — all empty files.

- [ ] **Step 4: Verify import works**

```bash
cd e:\CS-OmniMamba-2025
python -c "import src; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py scripts/__init__.py
git commit -m "chore: project skeleton and requirements"
```

---

### Task 2: Central config

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Write `src/config.py`**

```python
"""Central configuration: paths, constants, split dates."""
from pathlib import Path

# ── Root ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent   # e:\CS-OmniMamba-2025

# ── Raw data directories ───────────────────────────────────────────────────────
RADAR_RAW_DIR  = ROOT / "2025河北雷达灰度图"
RAIN_DIR       = ROOT / "70×66_rain_pic_hebei_2025"
PWV_DIR        = ROOT / "PWV_2025_S"

# ── Preprocessed RADAR (written by preprocess.py) ─────────────────────────────
RADAR_PREP_DIR = ROOT / "radar_preprocessed"

# ── Image dimensions (H × W = rows × cols) ────────────────────────────────────
H, W = 66, 70          # RAIN / PWV / preprocessed RADAR: array shape (66, 70)

# ── Sequence parameters ────────────────────────────────────────────────────────
T = 10                  # Number of RADAR history frames (t-9 … t)
STEP_SECONDS = 360      # 6 minutes between frames

# ── Normalisation ─────────────────────────────────────────────────────────────
PIXEL_MAX = 255.0

# ── Loss ──────────────────────────────────────────────────────────────────────
RAIN_EPS    = 2.0 / 255.0   # rain/no-rain boundary after normalisation
RAIN_WEIGHT = 10.0           # weight multiplier for rain pixels

# ── Date splits ───────────────────────────────────────────────────────────────
from datetime import date
TRAIN_START = date(2025, 5, 1)
TRAIN_END   = date(2025, 7, 31)
VAL_START   = date(2025, 8, 1)
VAL_END     = date(2025, 8, 15)
TEST_START  = date(2025, 8, 16)
TEST_END    = date(2025, 8, 31)
```

- [ ] **Step 2: Verify config imports cleanly**

```bash
python -c "from src.config import ROOT, H, W, T; print(ROOT, H, W, T)"
```

Expected: prints project root path and `66 70 10`.

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: central config with paths and constants"
```

---

### Task 3: Normalisation transform

**Files:**
- Create: `src/transforms.py`
- Create: `tests/test_transforms.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transforms.py
import numpy as np
import pytest
from src.transforms import normalize, denormalize

def test_normalize_no_rain():
    """Pixel 255 (no signal) should map to 0.0."""
    arr = np.array([[255, 255]], dtype=np.uint8)
    result = normalize(arr)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.0)

def test_normalize_max_signal():
    """Pixel 0 (max signal) should map to 1.0."""
    arr = np.array([[0, 0]], dtype=np.uint8)
    result = normalize(arr)
    np.testing.assert_allclose(result, 1.0)

def test_normalize_midpoint():
    arr = np.array([[127]], dtype=np.uint8)
    result = normalize(arr)
    np.testing.assert_allclose(result, (255 - 127) / 255.0, rtol=1e-5)

def test_normalize_preserves_shape():
    arr = np.zeros((66, 70), dtype=np.uint8)
    assert normalize(arr).shape == (66, 70)

def test_denormalize_roundtrip():
    """denormalize(normalize(x)) should recover original uint8 within ±1."""
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    recovered = denormalize(normalize(arr))
    assert recovered.dtype == np.uint8
    np.testing.assert_array_equal(recovered, arr)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_transforms.py -v
```

Expected: `ImportError` — `src.transforms` does not exist yet.

- [ ] **Step 3: Write `src/transforms.py`**

```python
"""Pixel normalisation utilities shared by all modalities."""
import numpy as np

def normalize(pixel: np.ndarray) -> np.ndarray:
    """
    Convert uint8 array to float32 in [0, 1].
    Inverted encoding: 255 (no signal) → 0.0, 0 (max signal) → 1.0.
    """
    return (255.0 - pixel.astype(np.float32)) / 255.0

def denormalize(arr: np.ndarray) -> np.ndarray:
    """
    Convert normalised float32 back to uint8.
    Inverse of normalize(); clips to [0, 255] before casting.
    """
    pixel = 255.0 - arr * 255.0
    return np.clip(np.round(pixel), 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_transforms.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/transforms.py tests/test_transforms.py
git commit -m "feat: normalize/denormalize with inverted encoding"
```

---

## Chunk 2: RADAR Preprocessing & Dataset

### Task 4: Offline RADAR preprocessing script

**Files:**
- Create: `src/preprocess.py`
- Create: `scripts/run_preprocess.py`

- [ ] **Step 1: Write `src/preprocess.py`**

```python
"""
Offline RADAR preprocessing: RGBA 700×660 → uint8 grayscale 66×70.
Run once via: python scripts/run_preprocess.py
"""
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from src.config import RADAR_RAW_DIR, RADAR_PREP_DIR, H, W


def preprocess_one(src_path: Path, dst_path: Path) -> None:
    """Downsample a single RADAR PNG and save as grayscale uint8."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return  # idempotent — skip already-processed files

    img = Image.open(src_path)
    arr = np.array(img)[:, :, 0]          # R channel only (R=G=B, Alpha=255)
    # PIL.resize takes (width, height); target shape (H, W) = (66, 70)
    downsampled = Image.fromarray(arr).resize((W, H), resample=Image.LANCZOS)
    downsampled.save(dst_path)


def preprocess_all(
    raw_dir: Path = RADAR_RAW_DIR,
    out_dir: Path = RADAR_PREP_DIR,
) -> None:
    """
    Walk raw_dir recursively, preprocess every .png, mirror the
    directory structure under out_dir.
    """
    png_files = sorted(raw_dir.rglob("*.png"))
    print(f"Found {len(png_files)} RADAR files. Output → {out_dir}")
    for src in tqdm(png_files, desc="Preprocessing RADAR"):
        rel = src.relative_to(raw_dir)
        dst = out_dir / rel
        preprocess_one(src, dst)
    print("Done.")
```

- [ ] **Step 2: Write `scripts/run_preprocess.py`**

```python
"""Entry point for offline RADAR preprocessing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import preprocess_all
preprocess_all()
```

- [ ] **Step 3: Run preprocessing on a small subset to verify correctness (dry-run check)**

```python
# Quick sanity check — run interactively or as a one-off script
from pathlib import Path
from src.preprocess import preprocess_one
import numpy as np
from PIL import Image

# Pick any existing RADAR file
sample = next(Path("2025河北雷达灰度图").rglob("*.png"))
out = Path("radar_preprocessed/_test_sample.png")
out.parent.mkdir(exist_ok=True)
preprocess_one(sample, out)
arr = np.array(Image.open(out))
assert arr.shape == (66, 70), f"Got {arr.shape}"
assert arr.dtype == np.uint8
print("Preprocessing sanity check PASSED:", arr.shape)
```

Expected: prints `Preprocessing sanity check PASSED: (66, 70)`.

- [ ] **Step 4: Run full preprocessing (this will take several minutes)**

```bash
python scripts/run_preprocess.py
```

Expected: progress bar completing, `Done.` printed. Output in `radar_preprocessed/`.

- [ ] **Step 5: Commit**

```bash
git add src/preprocess.py scripts/run_preprocess.py
git commit -m "feat: offline RADAR downsample script (LANCZOS, 700x660 -> 66x70)"
```

---

### Task 5: Sample index builder

**Files:**
- Create: `src/dataset.py` (index-building portion only, `__getitem__` in next task)
- Create: `tests/test_dataset.py` (index tests only)

- [ ] **Step 1: Write the failing tests for index building**

```python
# tests/test_dataset.py
from datetime import datetime, timedelta
from src.dataset import build_sample_index

def _make_timestamps(start: datetime, n: int, step_s: int = 360):
    """Helper: generate n consecutive timestamps at step_s-second intervals."""
    return [start + timedelta(seconds=i * step_s) for i in range(n)]

def test_continuous_sequence_yields_samples():
    ts = _make_timestamps(datetime(2025, 5, 1, 0, 0), 15)
    samples = build_sample_index(ts, T=10)
    # With 15 timestamps and T=10, we need windows of 11: indices 0..3 are valid
    assert len(samples) == 4

def test_gap_breaks_window():
    """A missing timestamp in the middle should invalidate overlapping windows."""
    ts = _make_timestamps(datetime(2025, 5, 1, 0, 0), 20)
    # Remove timestamp at index 10, creating a 12-minute gap
    ts.pop(10)
    samples = build_sample_index(ts, T=10)
    # Windows that span index 10 must be excluded
    # Valid windows: indices 0-9 only (window ending before gap) → indices 0..(-1 depends on gap)
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
    assert len(samples) == 1
    idx = samples[0]
    # inputs: ts[idx:idx+10], target: ts[idx+10]
    assert ts[idx + 10] - ts[idx + 9] == timedelta(seconds=360)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_dataset.py -v
```

Expected: `ImportError` — `src.dataset` not found.

- [ ] **Step 3: Implement index builder in `src/dataset.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dataset.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: sample index builder with gap detection (total_seconds fix)"
```

---

### Task 6: RainDataset `__getitem__`

**Files:**
- Modify: `src/dataset.py` (add `RainDataset` class)
- Modify: `tests/test_dataset.py` (add `__getitem__` tests)

- [ ] **Step 1: Write failing tests for `RainDataset.__getitem__`**

Add the following to `tests/test_dataset.py`:

```python
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
    # 15 timestamps, T=10 → windows 0..3 are valid (need 11 each)
    assert len(ds) == 4


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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_dataset.py::test_dataset_length -v
pytest tests/test_dataset.py::test_getitem_shapes -v
```

Expected: `AttributeError` — `RainDataset` not defined.

- [ ] **Step 3: Add `RainDataset` class to `src/dataset.py`**

Add after the existing `build_sample_index` function:

```python
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
```

- [ ] **Step 4: Run all dataset tests**

```bash
pytest tests/test_dataset.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: RainDataset with __getitem__, normalisation, shape [T,1,H,W]"
```

---

## Chunk 3: Loss, Metrics & Baselines

### Task 7: Weighted MSE loss

**Files:**
- Create: `src/loss.py`
- Create: `tests/test_loss.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_loss.py
import torch
import pytest
from src.loss import weighted_mse_loss
from src.config import RAIN_EPS, RAIN_WEIGHT

def test_zero_loss_for_perfect_prediction():
    t = torch.rand(2, 1, 66, 70)
    assert weighted_mse_loss(t, t).item() == pytest.approx(0.0, abs=1e-6)

def test_rain_pixels_weighted_higher():
    """A unit error on a rain pixel should produce more loss than on a no-rain pixel."""
    # no-rain target: 0.0; rain target: 1.0 (above EPS)
    target_rain   = torch.ones(1, 1, 1, 1)
    target_norain = torch.zeros(1, 1, 1, 1)
    pred_off      = torch.zeros(1, 1, 1, 1)

    loss_rain   = weighted_mse_loss(pred_off, target_rain)
    loss_norain = weighted_mse_loss(pred_off, target_norain)
    # rain: weight=RAIN_WEIGHT, no-rain: weight=1
    assert loss_rain.item() > loss_norain.item()

def test_no_rain_frame_has_finite_loss():
    target = torch.zeros(2, 1, 66, 70)  # all no-rain
    pred   = torch.full_like(target, 0.5)
    loss   = weighted_mse_loss(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

def test_loss_weight_ratio():
    """With one rain pixel vs one no-rain pixel, ratio should equal RAIN_WEIGHT."""
    pred_rain   = torch.zeros(1, 1, 1, 1)
    target_rain = torch.ones(1, 1, 1, 1)   # rain: target > EPS, error = 1
    pred_no     = torch.ones(1, 1, 1, 1)
    target_no   = torch.zeros(1, 1, 1, 1)  # no-rain, error = 1

    loss_rain = weighted_mse_loss(pred_rain, target_rain).item()
    loss_no   = weighted_mse_loss(pred_no,   target_no).item()
    assert loss_rain == pytest.approx(loss_no * RAIN_WEIGHT, rel=1e-4)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_loss.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write `src/loss.py`**

```python
"""Weighted MSE loss for precipitation nowcasting."""
import torch
import torch.nn as nn
from src.config import RAIN_EPS, RAIN_WEIGHT


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pixel-wise MSE with elevated weight for rain pixels.

    Args:
        pred:   float32 tensor, normalised [0, 1], any shape.
        target: same shape as pred.

    Returns:
        Scalar loss tensor.
    """
    rain_mask = (target > RAIN_EPS).float()
    weight = 1.0 + (RAIN_WEIGHT - 1.0) * rain_mask   # 1 for no-rain, RAIN_WEIGHT for rain
    return (weight * (pred - target) ** 2).mean()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_loss.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/loss.py tests/test_loss.py
git commit -m "feat: weighted MSE loss (rain weight=10)"
```

---

### Task 8: Evaluation metrics

**Files:**
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import torch
import pytest
from src.metrics import compute_csi_pod_far, compute_mse_mae, MetricsAccumulator

# ── CSI / POD / FAR ───────────────────────────────────────────────────────────

def test_perfect_prediction_csi_one():
    t = torch.rand(2, 1, 66, 70)
    m = compute_csi_pod_far(t, t, threshold=0.02)
    assert m["csi"] == pytest.approx(1.0, abs=1e-5)
    assert m["pod"] == pytest.approx(1.0, abs=1e-5)
    assert m["far"] == pytest.approx(0.0, abs=1e-5)

def test_all_zero_prediction_csi_zero():
    target = torch.rand(2, 1, 66, 70).clamp(0.05, 1.0)  # all rain
    pred   = torch.zeros_like(target)
    m = compute_csi_pod_far(pred, target, threshold=0.02)
    assert m["csi"] == pytest.approx(0.0, abs=1e-5)
    assert m["pod"] == pytest.approx(0.0, abs=1e-5)

def test_all_rain_false_alarm():
    target = torch.zeros(2, 1, 66, 70)       # no rain
    pred   = torch.ones(2, 1, 66, 70)        # predict all rain
    m = compute_csi_pod_far(pred, target, threshold=0.02)
    assert m["far"] == pytest.approx(1.0, abs=1e-5)

def test_no_rain_in_target_returns_nan_csi():
    """When there is genuinely no rain, CSI is undefined; return NaN."""
    target = torch.zeros(1, 1, 66, 70)
    pred   = torch.zeros(1, 1, 66, 70)
    m = compute_csi_pod_far(pred, target, threshold=0.02)
    assert torch.isnan(torch.tensor(m["csi"]))

# ── MSE / MAE ─────────────────────────────────────────────────────────────────

def test_mse_mae_perfect():
    t = torch.rand(2, 1, 66, 70)
    m = compute_mse_mae(t, t)
    assert m["mse"]      == pytest.approx(0.0, abs=1e-6)
    assert m["mae_rain"] == pytest.approx(0.0, abs=1e-6)

def test_mse_mae_all_norain_gives_nan_mae_rain():
    """MAE rain is undefined (no rain pixels) → NaN."""
    target = torch.zeros(2, 1, 66, 70)
    pred   = torch.rand(2, 1, 66, 70)
    m = compute_mse_mae(pred, target)
    assert torch.isnan(torch.tensor(m["mae_rain"]))

# ── MetricsAccumulator ────────────────────────────────────────────────────────

def test_accumulator_aggregates_tp_fp_fn():
    """
    MetricsAccumulator must aggregate raw TP/FP/FN across batches,
    NOT average per-batch CSI values.

    Two batches that together form a perfect prediction should yield CSI=1.
    If averaging per-batch values, a batch with NaN CSI (no rain) would
    need to be skipped, but the overall TP/FN/FP counts are still correct.
    """
    acc = MetricsAccumulator()
    # Batch 1: all rain, perfect prediction
    t1 = torch.ones(1, 1, 66, 70) * 0.5
    acc.update(t1, t1)
    # Batch 2: also all rain, perfect prediction
    t2 = torch.ones(1, 1, 66, 70) * 0.5
    acc.update(t2, t2)
    result = acc.compute()
    assert result["csi_weak"] == pytest.approx(1.0, abs=1e-4)
    assert result["mse"]      == pytest.approx(0.0, abs=1e-5)

def test_accumulator_cross_batch_csi():
    """
    CSI computed from aggregated counts must differ from average of per-batch CSIs.
    Batch 1 has only FP (model predicts rain, no target rain) → per-batch CSI = NaN.
    Batch 2 has only TP.
    Aggregated: tp=N, fp=N, fn=0 → CSI = tp/(tp+fp+fn) = 0.5
    If we had averaged per-batch CSIs (skipping NaN), we'd get CSI=1.0 — WRONG.
    """
    acc = MetricsAccumulator()
    # Batch 1: predict all rain, target all no-rain → all FP, CSI=NaN per-batch
    pred_fp  = torch.ones(1, 1, 4, 4) * 0.5
    tgt_none = torch.zeros(1, 1, 4, 4)
    acc.update(pred_fp, tgt_none)
    # Batch 2: perfect prediction on all-rain target → all TP, CSI=1 per-batch
    t_rain = torch.ones(1, 1, 4, 4) * 0.5
    acc.update(t_rain, t_rain)
    result = acc.compute()
    # Aggregated: tp=16, fp=16, fn=0 → CSI = 16/(16+16+0) = 0.5
    assert result["csi_weak"] == pytest.approx(0.5, abs=1e-4)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Step 3: Write `src/metrics.py`**

```python
"""
Evaluation metrics for precipitation nowcasting.
All inputs are normalised float32 tensors in [0, 1].
"""
from __future__ import annotations
import math
from typing import Dict

import torch
from src.config import RAIN_EPS

# ── Threshold definitions ─────────────────────────────────────────────────────
THRESH_WEAK   = 2.0  / 255.0   # original pixel < 253
THRESH_STRONG = 55.0 / 255.0   # original pixel < 200


def compute_csi_pod_far(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute CSI, POD, FAR for a given threshold.

    Returns dict with keys: 'csi', 'pod', 'far'.
    Returns NaN for CSI/POD when there are no rain pixels in target.
    """
    pred_bin   = (pred   > threshold).float()
    target_bin = (target > threshold).float()

    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()

    denom_csi = tp + fp + fn
    denom_pod = tp + fn
    denom_far = tp + fp

    # CSI and POD are undefined when there is no rain in target (denom_pod == 0)
    csi = tp / denom_csi if denom_pod > 0 else float("nan")
    pod = tp / denom_pod if denom_pod > 0 else float("nan")
    # FAR is undefined when the model predicts no rain at all (denom_far == 0)
    far = fp / denom_far if denom_far > 0 else float("nan")

    return {"csi": csi, "pod": pod, "far": far}


def compute_mse_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Compute full-image MSE and rain-only MAE."""
    mse = ((pred - target) ** 2).mean().item()

    rain_mask = (target > RAIN_EPS)
    if rain_mask.any():
        mae_rain = (pred[rain_mask] - target[rain_mask]).abs().mean().item()
    else:
        mae_rain = float("nan")

    return {"mse": mse, "mae_rain": mae_rain}


class MetricsAccumulator:
    """
    Accumulate CSI/POD/FAR/MSE/MAE over a full evaluation epoch by collecting
    raw TP/FP/FN counts and pixel-level squared errors, then computing final
    metrics once at the end.

    This is the correct way to aggregate threshold-based metrics: computing
    CSI per batch and then averaging batches would give biased results when
    batch sizes differ or when some batches have no rain (NaN contributions).
    """

    def __init__(self) -> None:
        # TP/FP/FN counts for two thresholds
        self._tp_w = self._fp_w = self._fn_w = 0.0
        self._tp_s = self._fp_s = self._fn_s = 0.0
        # MSE accumulation (sum of squared errors + pixel count)
        self._sq_sum: float = 0.0
        self._n_pixels: int = 0
        # MAE-rain accumulation (sum of abs errors on rain pixels + count)
        self._abs_rain_sum: float = 0.0
        self._n_rain_pixels: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate raw counts from one batch (or one sample)."""
        for thresh, prefix in [(THRESH_WEAK, "w"), (THRESH_STRONG, "s")]:
            pred_bin   = (pred   > thresh).float()
            target_bin = (target > thresh).float()
            tp = (pred_bin * target_bin).sum().item()
            fp = (pred_bin * (1 - target_bin)).sum().item()
            fn = ((1 - pred_bin) * target_bin).sum().item()
            if prefix == "w":
                self._tp_w += tp; self._fp_w += fp; self._fn_w += fn
            else:
                self._tp_s += tp; self._fp_s += fp; self._fn_s += fn

        # MSE: sum of pixel-wise squared errors
        self._sq_sum   += ((pred - target) ** 2).sum().item()
        self._n_pixels += pred.numel()

        # MAE-rain: errors only on rain pixels
        rain_mask = target > RAIN_EPS
        n_rain = rain_mask.sum().item()
        if n_rain > 0:
            self._abs_rain_sum   += (pred[rain_mask] - target[rain_mask]).abs().sum().item()
            self._n_rain_pixels  += int(n_rain)

    @staticmethod
    def _csi_pod_far(tp: float, fp: float, fn: float) -> Dict[str, float]:
        denom_pod = tp + fn
        denom_csi = tp + fp + fn
        denom_far = tp + fp
        return {
            "csi": tp / denom_csi if denom_pod > 0 else float("nan"),
            "pod": tp / denom_pod if denom_pod > 0 else float("nan"),
            "far": fp / denom_far if denom_far > 0 else float("nan"),
        }

    def compute(self) -> Dict[str, float]:
        w = self._csi_pod_far(self._tp_w, self._fp_w, self._fn_w)
        s = self._csi_pod_far(self._tp_s, self._fp_s, self._fn_s)
        mse      = self._sq_sum / self._n_pixels if self._n_pixels > 0 else float("nan")
        mae_rain = (self._abs_rain_sum / self._n_rain_pixels
                    if self._n_rain_pixels > 0 else float("nan"))
        return {
            "csi_weak":   w["csi"], "pod_weak":   w["pod"], "far_weak":   w["far"],
            "csi_strong": s["csi"], "pod_strong": s["pod"], "far_strong": s["far"],
            "mse": mse,
            "mae_rain": mae_rain,
        }

    def reset(self) -> None:
        self.__init__()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: CSI/POD/FAR/MSE/MAE metrics with NaN-safe accumulator"
```

---

### Task 9: Persistence and Zero baselines

**Files:**
- Create: `src/baselines.py`
- Create: `tests/test_baselines.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_baselines.py
import torch
from src.baselines import PersistenceBaseline, ZeroBaseline

def test_persistence_copies_last_rain_frame():
    # batch: last RAIN frame at t is known; predict RAIN(t+1) = RAIN(t)
    rain_t = torch.rand(4, 1, 66, 70)
    pred   = PersistenceBaseline.predict(rain_t)
    assert pred.shape == rain_t.shape
    assert torch.allclose(pred, rain_t)

def test_zero_baseline_all_zeros():
    shape = (4, 1, 66, 70)
    pred = ZeroBaseline.predict(shape)
    assert pred.shape == shape
    assert pred.sum().item() == 0.0
    assert pred.dtype == torch.float32
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_baselines.py -v
```

- [ ] **Step 3: Write `src/baselines.py`**

```python
"""Persistence and Zero baselines for evaluation."""
from __future__ import annotations
from typing import Tuple
import torch


class PersistenceBaseline:
    """Predict RAIN(t+1) = RAIN(t). Requires RAIN at current timestep."""

    @staticmethod
    def predict(rain_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rain_t: normalised RAIN at time t, shape [B, 1, H, W].
        Returns:
            Prediction for t+1 (identical copy).
        """
        return rain_t.clone()


class ZeroBaseline:
    """Predict all-no-rain everywhere."""

    @staticmethod
    def predict(shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Args:
            shape: desired output shape, e.g. (B, 1, H, W).
        Returns:
            Zero tensor (no rain predicted).
        """
        return torch.zeros(shape, dtype=torch.float32)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_baselines.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/baselines.py tests/test_baselines.py
git commit -m "feat: persistence and zero baselines"
```

---

## Chunk 4: Training Loop (End-to-End)

### Task 10: Stub model

**Files:**
- Create: `src/model_stub.py`

> This stub exists solely to make the training loop runnable. Replace with your actual model by implementing the same interface.

- [ ] **Step 1: Write `src/model_stub.py`**

```python
"""
Minimal stub model: RADAR time steps + PWV flattened into channel dimension,
then processed by a 2D convolutional network.
NOT intended as the final model. Replace by implementing the same forward() signature.

Interface contract:
    Input:
        radar : Tensor [B, T, 1, H, W]
        pwv   : Tensor [B, 1, 1, H, W]
    Output:
        Tensor [B, 1, H, W]  — predicted RAIN(t+1), normalised [0, 1]
"""
import torch
import torch.nn as nn
from src.config import T, H, W


class StubModel(nn.Module):
    def __init__(self, t: int = T) -> None:
        super().__init__()
        in_ch = t + 1   # T RADAR frames + 1 PWV frame (all flattened to channel dim)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, radar: torch.Tensor, pwv: torch.Tensor) -> torch.Tensor:
        # radar: [B, T, 1, H, W] → squeeze channel → [B, T, H, W]
        # pwv:   [B, 1, 1, H, W] → squeeze channel → [B, 1, H, W]
        radar_2d = radar.squeeze(2)
        pwv_2d   = pwv.squeeze(2)
        x = torch.cat([radar_2d, pwv_2d], dim=1)  # [B, T+1, H, W]
        return self.net(x)   # [B, 1, H, W]
```

- [ ] **Step 2: Quick smoke test (no pytest)**

```python
python -c "
import torch
from src.model_stub import StubModel
m = StubModel()
radar = torch.rand(2, 10, 1, 66, 70)
pwv   = torch.rand(2, 1,  1, 66, 70)
out = m(radar, pwv)
print('output shape:', out.shape)
assert out.shape == (2, 1, 66, 70)
print('StubModel smoke test PASSED')
"
```

Expected: `output shape: torch.Size([2, 1, 66, 70])` and `PASSED`.

- [ ] **Step 3: Commit**

```bash
git add src/model_stub.py
git commit -m "feat: stub Conv2D model for training loop smoke test"
```

---

### Task 11: Training loop

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Write `src/train.py`**

```python
"""
Training and evaluation loop.

Usage:
    python src/train.py --epochs 10 --batch-size 8

The model is imported from src.model_stub; replace StubModel with your
actual model class without modifying this file.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    RADAR_PREP_DIR, RAIN_DIR, PWV_DIR,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    T,
)
from src.dataset import RainDataset
from src.loss import weighted_mse_loss
from src.metrics import MetricsAccumulator
from src.model_stub import StubModel


# ── Split filtering ───────────────────────────────────────────────────────────

def _filter_dataset_by_split(ds: RainDataset, split: str) -> RainDataset:
    """
    Return a filtered view of the dataset containing only samples whose
    TARGET frame falls within the requested date range.

    NOTE: we filter on ds.timestamps[i + ds.T]  (the target frame, t+1),
    NOT on ds.timestamps[i] (the window start, t-T+1).  Filtering by the
    window start would allow the target frame to fall into the next split,
    leaking labels across the train/val/test boundary.
    """
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val":   (VAL_START,   VAL_END),
        "test":  (TEST_START,  TEST_END),
    }
    start, end = splits[split]
    ds.indices = [
        i for i in ds.indices
        if start <= ds.timestamps[i + ds.T].date() <= end
    ]
    return ds


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        radar = batch["radar"].to(device)
        pwv   = batch["pwv"].to(device)
        rain  = batch["rain"].squeeze(1).to(device)   # [B, 1, H, W]

        optimiser.zero_grad()
        pred = model(radar, pwv)
        loss = weighted_mse_loss(pred, rain)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device) -> dict:
    model.eval()
    acc = MetricsAccumulator()
    for batch in tqdm(loader, desc="eval ", leave=False):
        radar = batch["radar"].to(device)
        pwv   = batch["pwv"].to(device)
        rain  = batch["rain"].squeeze(1).to(device)

        pred = model(radar, pwv)
        acc.update(pred.cpu(), rain.cpu())
    return acc.compute()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = _filter_dataset_by_split(
        RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR), "train"
    )
    val_ds = _filter_dataset_by_split(
        RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR), "val"
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model = StubModel(t=T).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, device)
        val_metrics = eval_epoch(model, val_loader, device)

        csi  = val_metrics.get("csi_weak",  float("nan"))
        far  = val_metrics.get("far_weak",  float("nan"))
        mse  = val_metrics.get("mse",       float("nan"))
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"| val_csi_weak={csi:.4f} | val_far_weak={far:.4f} | val_mse={mse:.5f}"
        )

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path("checkpoints") / "stub_last.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    # ── Baseline evaluation (run once, for comparison) ────────────────────────
    # PersistenceBaseline uses RAIN(t) from batch["rain_current"].
    # ZeroBaseline uses all-zero prediction.
    from src.baselines import PersistenceBaseline, ZeroBaseline

    @torch.no_grad()
    def eval_baselines(loader, device):
        acc_p = MetricsAccumulator()
        acc_z = MetricsAccumulator()
        for batch in tqdm(loader, desc="baselines", leave=False):
            rain_target  = batch["rain"].squeeze(1).to(device)        # [B,1,H,W] t+1
            rain_current = batch["rain_current"].squeeze(1).to(device) # [B,1,H,W] t
            pred_p = PersistenceBaseline.predict(rain_current)
            pred_z = ZeroBaseline.predict(rain_target.shape)
            acc_p.update(pred_p.cpu(), rain_target.cpu())
            acc_z.update(pred_z.cpu(), rain_target.cpu())
        return acc_p.compute(), acc_z.compute()

    p_metrics, z_metrics = eval_baselines(val_loader, device)
    print(f"[Persistence] csi_weak={p_metrics.get('csi_weak', float('nan')):.4f} "
          f"| mse={p_metrics.get('mse', float('nan')):.5f}")
    print(f"[Zero      ] csi_weak={z_metrics.get('csi_weak', float('nan')):.4f} "
          f"| mse={z_metrics.get('mse', float('nan')):.5f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: End-to-end smoke test with fake data**

```bash
# This uses the tiny fake dataset from tests/ to verify the loop runs without errors
python -c "
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import tempfile, sys

# Build a tiny fake dataset
sys.path.insert(0, '.')
from tests.test_dataset import _make_fake_dataset
from src.dataset import RainDataset
from src.train import train_epoch, eval_epoch
from src.model_stub import StubModel

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    radar_dir, pwv_dir, rain_dir = _make_fake_dataset(tmp, n_frames=15)
    ds = RainDataset(radar_dir, pwv_dir, rain_dir, T=10)
    loader = DataLoader(ds, batch_size=2)

    model = StubModel()
    opt   = torch.optim.Adam(model.parameters())
    loss  = train_epoch(model, loader, opt, torch.device('cpu'))
    metrics = eval_epoch(model, loader, torch.device('cpu'))
    print('train_loss:', loss)
    print('val_metrics:', metrics)
    print('End-to-end smoke test PASSED')
"
```

Expected: prints loss and metrics, then `PASSED`.

- [ ] **Step 3: Commit**

```bash
git add src/train.py src/model_stub.py
git commit -m "feat: training loop with weighted MSE + MetricsAccumulator"
```

---

### Task 12: Run full test suite

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS, zero failures.

- [ ] **Step 2: Final commit**

```bash
git add .
git commit -m "chore: verified full test suite green"
```

---

## Notes for Model Replacement

To swap in your actual model, replace `StubModel` in `src/train.py`:

```python
# src/train.py — change this one import:
from src.your_model import YourModel as StubModel
```

Your model must satisfy:
- `forward(radar: Tensor[B,T,1,H,W], pwv: Tensor[B,1,1,H,W]) → Tensor[B,1,H,W]`
- Output in [0, 1] (use Sigmoid or clamp)

No other changes required.
