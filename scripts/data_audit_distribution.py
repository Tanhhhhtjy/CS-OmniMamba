import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from omnimamba.config import TrainingConfig
from omnimamba.data_match import match_samples
from omnimamba.splits import split_records


BINS = [
    ("NoRain", None, 0.0),
    ("Drizzle", 0.0, 2.5),
    ("Light", 2.5, 8.0),
    ("Moderate", 8.0, 16.0),
    ("Heavy", 16.0, 30.0),
    ("Torrential", 30.0, 50.0),
    ("Extreme", 50.0, None),
]


def decode_rain_from_png(path: str, rain_max: float, use_log: bool) -> np.ndarray:
    gray = np.array(Image.open(path).convert("L"), dtype=np.float32)
    inv = 255.0 - np.clip(gray, 0.0, 255.0)
    if not use_log:
        return inv / 255.0 * rain_max
    max_log = np.log1p(rain_max)
    return np.expm1(inv / 255.0 * max_log)


def bin_mask(values: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    if low is None:
        return values <= high
    if high is None:
        return values > low
    return (values > low) & (values <= high)


def init_split_bucket() -> Dict:
    return {
        "record_count": 0,
        "hour_hist": [0] * 24,
        "pixel_bins": {
            "T+1h": {name: 0 for name, _, _ in BINS},
            "T+2h": {name: 0 for name, _, _ in BINS},
            "T+3h": {name: 0 for name, _, _ in BINS},
        },
        "sample_event_bins": {name: 0 for name, _, _ in BINS},
        "sample_max_rain": [],
    }


def summarize_split(
    records,
    rain_max: float,
    use_log: bool,
    sample_limit: int | None,
) -> Dict:
    out = init_split_bucket()
    selected = list(records)
    if sample_limit is not None and sample_limit > 0:
        selected = selected[:sample_limit]

    for record in selected:
        out["record_count"] += 1
        out["hour_hist"][record.timestamp.hour] += 1

        paths = [record.target_1h_path, record.target_2h_path, record.target_3h_path]
        local_max = []

        for idx, path in enumerate(paths):
            horizon = f"T+{idx + 1}h"
            arr = decode_rain_from_png(path, rain_max=rain_max, use_log=use_log)
            local_max.append(float(arr.max()))

            for name, low, high in BINS:
                out["pixel_bins"][horizon][name] += int(bin_mask(arr, low, high).sum())

        sample_max = max(local_max)
        out["sample_max_rain"].append(sample_max)
        for name, low, high in BINS:
            if low is None:
                in_bin = sample_max <= high
            elif high is None:
                in_bin = sample_max > low
            else:
                in_bin = (sample_max > low) and (sample_max <= high)
            if in_bin:
                out["sample_event_bins"][name] += 1
                break

    # convert raw counts to ratios
    ratios = {"pixel_bins": {}, "sample_event_bins": {}}
    for horizon, counts in out["pixel_bins"].items():
        total = sum(counts.values())
        ratios["pixel_bins"][horizon] = {
            k: float(v / max(total, 1)) for k, v in counts.items()
        }

    total_samples = out["record_count"]
    ratios["sample_event_bins"] = {
        k: float(v / max(total_samples, 1)) for k, v in out["sample_event_bins"].items()
    }

    max_arr = np.array(out["sample_max_rain"], dtype=float) if out["sample_max_rain"] else np.array([])
    out["sample_max_rain_stats"] = {
        "mean": float(max_arr.mean()) if max_arr.size else None,
        "p50": float(np.percentile(max_arr, 50)) if max_arr.size else None,
        "p90": float(np.percentile(max_arr, 90)) if max_arr.size else None,
        "p99": float(np.percentile(max_arr, 99)) if max_arr.size else None,
        "max": float(max_arr.max()) if max_arr.size else None,
    }
    out["ratios"] = ratios
    out.pop("sample_max_rain", None)
    return out


def plot_split_rain_bins(summary: Dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    splits = ["train", "val", "test"]
    horizons = ["T+1h", "T+2h", "T+3h"]

    for horizon in horizons:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(splits))
        bottom = np.zeros(len(splits), dtype=float)

        for name, _, _ in BINS:
            vals = [summary[s]["ratios"]["pixel_bins"][horizon][name] for s in splits]
            ax.bar(x, vals, bottom=bottom, label=name)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Pixel Ratio")
        ax.set_title(f"Rain Bin Distribution ({horizon})")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"rain_bins_{horizon}.png"), dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze rain intensity distributions by split.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./results/data_audit")
    parser.add_argument("--rain-max", type=float, default=50.0)
    parser.add_argument("--no-rain-log", action="store_true", help="Use linear decode instead of log decode.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional limit per split for fast dry-run.")
    args = parser.parse_args()

    cfg = TrainingConfig()

    pwv_dir = os.path.join(args.data_root, "PWV")
    radar_dir = os.path.join(args.data_root, "RADAR")
    rain_dir = os.path.join(args.data_root, "RAIN")

    records = match_samples(pwv_dir, radar_dir, rain_dir, cfg)
    train_records, val_records, test_records = split_records(records, cfg)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "decode_mode": "linear" if args.no_rain_log else "log",
        "rain_max": args.rain_max,
        "bins": [name for name, _, _ in BINS],
        "train": summarize_split(train_records, args.rain_max, not args.no_rain_log, args.sample_limit),
        "val": summarize_split(val_records, args.rain_max, not args.no_rain_log, args.sample_limit),
        "test": summarize_split(test_records, args.rain_max, not args.no_rain_log, args.sample_limit),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")

    out_json = os.path.join(args.output_dir, "rain_bins_by_split.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_split_rain_bins(summary, figures_dir)

    print(f"Saved distribution summary: {out_json}")
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
