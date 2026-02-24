import argparse
import bisect
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

from omnimamba.config import TrainingConfig
from omnimamba.data_match import _build_name_map, parse_time


RADAR_MAX_ANCHOR_GAP_SECONDS = 3600
RADAR_GAP_TOLERANCE_SECONDS = 600


def _choose_best_anchor(
    t1: datetime,
    radar_times: List[datetime],
    radar_paths: List[str],
) -> Tuple[int, datetime, str]:
    idx = bisect.bisect_left(radar_times, t1)
    candidates = []
    if idx < len(radar_times):
        candidates.append((idx, radar_times[idx], radar_paths[idx]))
    if idx > 0:
        candidates.append((idx - 1, radar_times[idx - 1], radar_paths[idx - 1]))
    return min(candidates, key=lambda item: abs((item[1] - t1).total_seconds()))


def _build_radar_sequence_with_stats(
    anchor_idx: int,
    radar_times: List[datetime],
    radar_paths: List[str],
    seq_len: int,
) -> Tuple[List[str], int, List[float]]:
    seq = [radar_paths[anchor_idx]]
    prev_time = radar_times[anchor_idx]
    idx = anchor_idx - 1
    gap_seconds: List[float] = []

    while len(seq) < seq_len and idx >= 0:
        gap = (prev_time - radar_times[idx]).total_seconds()
        gap_seconds.append(gap)
        if gap <= RADAR_GAP_TOLERANCE_SECONDS:
            seq.append(radar_paths[idx])
            prev_time = radar_times[idx]
        idx -= 1

    pad_count = 0
    while len(seq) < seq_len:
        seq.append(seq[-1])
        pad_count += 1

    seq.reverse()
    return seq, pad_count, gap_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit matching quality for PWV-RADAR-RAIN records.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./results/data_audit")
    parser.add_argument("--radar-seq-len", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = TrainingConfig()
    if args.radar_seq_len is not None:
        cfg.radar_seq_len = args.radar_seq_len

    pwv_dir = os.path.join(args.data_root, "PWV")
    radar_dir = os.path.join(args.data_root, "RADAR")
    rain_dir = os.path.join(args.data_root, "RAIN")

    pwv_map = _build_name_map(pwv_dir)
    radar_map = _build_name_map(radar_dir)
    rain_map = _build_name_map(rain_dir)

    radar_pairs = []
    for name, path in radar_map.items():
        dt = parse_time(name)
        if dt is not None:
            radar_pairs.append((dt, path))
    radar_pairs.sort(key=lambda x: x[0])
    radar_times = [t for t, _ in radar_pairs]
    radar_paths = [p for _, p in radar_pairs]

    reason_counts: Dict[str, int] = {
        "pwv_bad_timestamp": 0,
        "out_of_split_range": 0,
        "no_radar_frames": 0,
        "radar_anchor_too_far": 0,
        "missing_target_1h": 0,
        "missing_target_2h": 0,
        "missing_target_3h": 0,
        "matched": 0,
    }

    anchor_offsets = []
    padding_counts = []
    seq_gap_seconds = []

    for name, pwv_path in pwv_map.items():
        t1 = parse_time(name)
        if t1 is None:
            reason_counts["pwv_bad_timestamp"] += 1
            continue

        if t1 < cfg.train_start or t1 > cfg.test_end:
            reason_counts["out_of_split_range"] += 1
            continue

        if not radar_times:
            reason_counts["no_radar_frames"] += 1
            continue

        best_idx, best_time, _ = _choose_best_anchor(t1, radar_times, radar_paths)
        offset_sec = abs((best_time - t1).total_seconds())
        if offset_sec > RADAR_MAX_ANCHOR_GAP_SECONDS:
            reason_counts["radar_anchor_too_far"] += 1
            continue

        missing_target = False
        for h, key in ((1, "missing_target_1h"), (2, "missing_target_2h"), (3, "missing_target_3h")):
            target_key = (t1 + timedelta(hours=h)).strftime("%Y-%m-%d-%H-%M-%S")
            if target_key not in rain_map:
                reason_counts[key] += 1
                missing_target = True
        if missing_target:
            continue

        _, pad_count, gaps = _build_radar_sequence_with_stats(
            best_idx, radar_times, radar_paths, cfg.radar_seq_len
        )
        reason_counts["matched"] += 1
        anchor_offsets.append(offset_sec)
        padding_counts.append(pad_count)
        seq_gap_seconds.extend(gaps)

    matched = reason_counts["matched"]
    total_pwv = len(pwv_map)

    def _stats(values: List[float]) -> Dict:
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
        arr = np.array(values, dtype=float)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        }

    result = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "train_start": cfg.train_start.isoformat(),
            "train_end": cfg.train_end.isoformat(),
            "val_start": cfg.val_start.isoformat(),
            "val_end": cfg.val_end.isoformat(),
            "test_start": cfg.test_start.isoformat(),
            "test_end": cfg.test_end.isoformat(),
            "radar_seq_len": cfg.radar_seq_len,
        },
        "counts": {
            "total_pwv_candidates": total_pwv,
            "matched_records": matched,
            "matched_ratio": float(matched / max(total_pwv, 1)),
            "reasons": reason_counts,
        },
        "anchor_offset_seconds": _stats(anchor_offsets),
        "radar_padding_per_sample": _stats(padding_counts),
        "radar_interframe_gap_seconds": _stats(seq_gap_seconds),
        "padding_trigger_ratio": float(sum(1 for x in padding_counts if x > 0) / max(matched, 1)),
    }

    out_path = os.path.join(args.output_dir, "match_quality.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved match quality report: {out_path}")


if __name__ == "__main__":
    main()
