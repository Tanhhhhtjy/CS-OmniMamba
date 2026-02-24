import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from omnimamba.config import TrainingConfig
from omnimamba.data_match import match_samples
from omnimamba.splits import split_records


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p.astype(float), eps, None)
    q = np.clip(q.astype(float), eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * np.log(a / b))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def build_hour_hist(records) -> np.ndarray:
    hist = np.zeros(24, dtype=float)
    for r in records:
        hist[r.timestamp.hour] += 1
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def estimate_heavy_event_ratio(records, threshold_mm: float = 16.0) -> float:
    import numpy as np
    from PIL import Image

    if not records:
        return 0.0

    heavy = 0
    total = 0
    max_log = np.log1p(50.0)

    for record in records:
        paths = [record.target_1h_path, record.target_2h_path, record.target_3h_path]
        event_max = 0.0
        for path in paths:
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
            rain = np.expm1((255.0 - arr) / 255.0 * max_log)
            event_max = max(event_max, float(rain.max()))
        heavy += 1 if event_max >= threshold_mm else 0
        total += 1

    return float(heavy / max(total, 1))


def interval_overlap_count(
    times_a: List[datetime],
    times_b: List[datetime],
    history_minutes: int,
    future_minutes: int,
) -> int:
    if not times_a or not times_b:
        return 0

    a_intervals = [(t - timedelta(minutes=history_minutes), t + timedelta(minutes=future_minutes)) for t in times_a]
    b_intervals = [(t - timedelta(minutes=history_minutes), t + timedelta(minutes=future_minutes)) for t in times_b]

    a_intervals.sort(key=lambda x: x[0])
    b_intervals.sort(key=lambda x: x[0])

    i = 0
    j = 0
    overlaps = 0

    while i < len(a_intervals) and j < len(b_intervals):
        a0, a1 = a_intervals[i]
        b0, b1 = b_intervals[j]

        if a1 < b0:
            i += 1
            continue
        if b1 < a0:
            j += 1
            continue

        overlaps += 1
        if a1 <= b1:
            i += 1
        else:
            j += 1

    return overlaps


def load_distribution_probs(path: str) -> Dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rain_prob_vector(dist: Dict, split_name: str, horizon: str) -> np.ndarray:
    bins = dist["bins"]
    probs = dist[split_name]["ratios"]["pixel_bins"][horizon]
    return np.array([probs[b] for b in bins], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze split drift and potential boundary leakage.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./results/data_audit")
    parser.add_argument(
        "--distribution-json",
        default="./results/data_audit/rain_bins_by_split.json",
        help="Path to output of data_audit_distribution.py",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = TrainingConfig()
    records = match_samples(
        os.path.join(args.data_root, "PWV"),
        os.path.join(args.data_root, "RADAR"),
        os.path.join(args.data_root, "RAIN"),
        cfg,
    )
    train_records, val_records, test_records = split_records(records, cfg)

    split_map = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    split_summary = {}
    for name, recs in split_map.items():
        ts = [r.timestamp for r in recs]
        split_summary[name] = {
            "count": len(recs),
            "start": ts[0].isoformat() if ts else None,
            "end": ts[-1].isoformat() if ts else None,
        }

    history_minutes = (cfg.radar_seq_len - 1) * 6
    future_minutes = 3 * 60

    leakage = {
        "history_minutes": history_minutes,
        "future_minutes": future_minutes,
        "train_val_overlaps": interval_overlap_count(
            [r.timestamp for r in train_records],
            [r.timestamp for r in val_records],
            history_minutes,
            future_minutes,
        ),
        "val_test_overlaps": interval_overlap_count(
            [r.timestamp for r in val_records],
            [r.timestamp for r in test_records],
            history_minutes,
            future_minutes,
        ),
        "train_test_overlaps": interval_overlap_count(
            [r.timestamp for r in train_records],
            [r.timestamp for r in test_records],
            history_minutes,
            future_minutes,
        ),
    }

    hour_hists = {k: build_hour_hist(v) for k, v in split_map.items()}

    dist = load_distribution_probs(args.distribution_json)
    rain_jsd = {}
    if dist is not None:
        for horizon in ("T+1h", "T+2h", "T+3h"):
            train_vec = rain_prob_vector(dist, "train", horizon)
            val_vec = rain_prob_vector(dist, "val", horizon)
            test_vec = rain_prob_vector(dist, "test", horizon)
            rain_jsd[horizon] = {
                "train_val": js_divergence(train_vec, val_vec),
                "train_test": js_divergence(train_vec, test_vec),
                "val_test": js_divergence(val_vec, test_vec),
            }

    hour_jsd = {
        "train_val": js_divergence(hour_hists["train"], hour_hists["val"]),
        "train_test": js_divergence(hour_hists["train"], hour_hists["test"]),
        "val_test": js_divergence(hour_hists["val"], hour_hists["test"]),
    }

    heavy_ratio = {
        "train": estimate_heavy_event_ratio(train_records),
        "val": estimate_heavy_event_ratio(val_records),
        "test": estimate_heavy_event_ratio(test_records),
    }

    out = {
        "generated_at": datetime.now().isoformat(),
        "split_summary": split_summary,
        "leakage_risk": leakage,
        "hour_distribution_jsd": hour_jsd,
        "rain_bin_jsd": rain_jsd,
        "heavy_event_ratio_ge_16mmh": heavy_ratio,
        "notes": {
            "rain_bin_jsd_available": dist is not None,
            "distribution_json": args.distribution_json,
            "hint": "Run scripts/data_audit_distribution.py first for rain_bin_jsd.",
        },
    }

    out_path = os.path.join(args.output_dir, "split_drift_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved split drift report: {out_path}")


if __name__ == "__main__":
    main()
