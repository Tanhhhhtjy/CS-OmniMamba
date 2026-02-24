import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from omnimamba.data_match import parse_time


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _safe_listdir(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return os.listdir(path)


def _file_info(folder: str, filename: str) -> Tuple[str, str, Optional[datetime], bool]:
    full_path = os.path.join(folder, filename)
    stem, ext = os.path.splitext(filename)
    ext_ok = ext.lower() in IMAGE_EXTS
    ts = parse_time(stem) if ext_ok else None
    return full_path, ext.lower(), ts, ext_ok


def scan_folder(folder: str, sample_images: int = 2000) -> Dict:
    names = _safe_listdir(folder)
    total_files = 0
    ext_counter: Dict[str, int] = {}
    image_files = 0
    valid_ts_files = 0
    invalid_ts_files = 0
    duplicate_timestamps = 0

    ts_counter: Dict[str, int] = {}
    valid_image_paths: List[str] = []

    for name in names:
        full_path = os.path.join(folder, name)
        if not os.path.isfile(full_path):
            continue
        total_files += 1
        _, ext, ts, is_image = _file_info(folder, name)
        ext_counter[ext] = ext_counter.get(ext, 0) + 1

        if not is_image:
            continue

        image_files += 1
        if ts is None:
            invalid_ts_files += 1
            continue

        valid_ts_files += 1
        ts_key = ts.strftime("%Y-%m-%d-%H-%M-%S")
        ts_counter[ts_key] = ts_counter.get(ts_key, 0) + 1
        if ts_counter[ts_key] > 1:
            duplicate_timestamps += 1
        valid_image_paths.append(full_path)

    timestamp_values = sorted(ts_counter.keys())
    ts_start = timestamp_values[0] if timestamp_values else None
    ts_end = timestamp_values[-1] if timestamp_values else None

    image_health = image_health_check(valid_image_paths, max_samples=sample_images)

    return {
        "folder": folder,
        "total_files": total_files,
        "image_files": image_files,
        "valid_timestamp_images": valid_ts_files,
        "invalid_timestamp_images": invalid_ts_files,
        "duplicate_timestamp_files": duplicate_timestamps,
        "timestamp_start": ts_start,
        "timestamp_end": ts_end,
        "extensions": ext_counter,
        "image_health": image_health,
    }


def image_health_check(image_paths: List[str], max_samples: int = 2000) -> Dict:
    if not image_paths:
        return {
            "checked_images": 0,
            "read_failures": 0,
            "constant_image_count": 0,
            "shape_counts": {},
            "pixel_min": None,
            "pixel_max": None,
            "pixel_mean": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "sat0_ratio": None,
            "sat255_ratio": None,
        }

    rng = np.random.default_rng(42)
    if len(image_paths) > max_samples:
        chosen = rng.choice(image_paths, size=max_samples, replace=False).tolist()
    else:
        chosen = list(image_paths)

    read_failures = 0
    constant_images = 0
    shape_counts: Dict[str, int] = {}

    min_vals = []
    max_vals = []
    mean_vals = []
    sat0 = 0
    sat255 = 0
    pixel_total = 0

    for path in chosen:
        try:
            arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        except Exception:
            read_failures += 1
            continue

        shape_key = f"{arr.shape[0]}x{arr.shape[1]}"
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1

        a_min = int(arr.min())
        a_max = int(arr.max())
        if a_min == a_max:
            constant_images += 1

        min_vals.append(a_min)
        max_vals.append(a_max)
        mean_vals.append(float(arr.mean()))

        sat0 += int((arr == 0).sum())
        sat255 += int((arr == 255).sum())
        pixel_total += int(arr.size)

    if not min_vals:
        return {
            "checked_images": len(chosen),
            "read_failures": read_failures,
            "constant_image_count": constant_images,
            "shape_counts": shape_counts,
            "pixel_min": None,
            "pixel_max": None,
            "pixel_mean": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "sat0_ratio": None,
            "sat255_ratio": None,
        }

    all_means = np.array(mean_vals, dtype=float)

    return {
        "checked_images": len(chosen),
        "read_failures": read_failures,
        "constant_image_count": constant_images,
        "shape_counts": shape_counts,
        "pixel_min": int(min(min_vals)),
        "pixel_max": int(max(max_vals)),
        "pixel_mean": float(all_means.mean()),
        "p01": float(np.percentile(all_means, 1)),
        "p50": float(np.percentile(all_means, 50)),
        "p99": float(np.percentile(all_means, 99)),
        "sat0_ratio": float(sat0 / max(pixel_total, 1)),
        "sat255_ratio": float(sat255 / max(pixel_total, 1)),
    }


def build_timeline_stats(folder_summary: Dict) -> Dict:
    start = folder_summary.get("timestamp_start")
    end = folder_summary.get("timestamp_end")
    valid_count = folder_summary.get("valid_timestamp_images", 0)
    if not start or not end or valid_count <= 1:
        return {
            "days_covered": 0,
            "expected_6min_steps": 0,
            "estimated_coverage_ratio": 0.0,
        }

    start_dt = datetime.strptime(start, "%Y-%m-%d-%H-%M-%S")
    end_dt = datetime.strptime(end, "%Y-%m-%d-%H-%M-%S")
    total_minutes = max((end_dt - start_dt).total_seconds() / 60.0, 0.0)
    expected = int(total_minutes // 6) + 1
    coverage = valid_count / max(expected, 1)

    return {
        "days_covered": float(total_minutes / 60.0 / 24.0),
        "expected_6min_steps": expected,
        "estimated_coverage_ratio": float(coverage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and health-check PWV/RADAR/RAIN image datasets.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--sample-images", type=int, default=2000)
    parser.add_argument("--output-dir", default="./results/data_audit")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    modal_folders = {
        "PWV": os.path.join(args.data_root, "PWV"),
        "RADAR": os.path.join(args.data_root, "RADAR"),
        "RAIN": os.path.join(args.data_root, "RAIN"),
    }

    summary = {"data_root": args.data_root, "generated_at": datetime.now().isoformat()}
    for key, path in modal_folders.items():
        summary[key] = scan_folder(path, sample_images=args.sample_images)
        summary[key]["timeline"] = build_timeline_stats(summary[key])

    out_path = os.path.join(args.output_dir, "inventory.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved inventory: {out_path}")


if __name__ == "__main__":
    main()
