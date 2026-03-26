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
