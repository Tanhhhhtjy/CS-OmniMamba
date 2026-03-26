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
