from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import bisect
import os

from .config import TrainingConfig

# Gap tolerance between consecutive radar frames (seconds)
_RADAR_GAP_TOLERANCE = 600  # 10 minutes


@dataclass
class SampleRecord:
    timestamp: datetime
    pwv_path: str
    # Current-frame radar (kept for backward compat)
    radar_path: str
    target_1h_path: str
    target_2h_path: str
    target_3h_path: str
    # Temporal radar sequence: oldest -> newest (last element == radar_path)
    radar_seq_paths: List[str] = field(default_factory=list)


def parse_time(name: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d-%H-%M-%S", "%Y-%m-%d-%H-%M"):
        try:
            return datetime.strptime(name, fmt)
        except ValueError:
            continue
    return None


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _build_name_map(folder: str) -> Dict[str, str]:
    result = {}
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in _IMAGE_EXTS:
            continue
        full_path = os.path.join(folder, fname)
        if os.path.isfile(full_path):
            result[os.path.splitext(fname)[0]] = full_path
    return result


def _build_radar_sequence(
    anchor_idx: int,
    radar_times: List[datetime],
    radar_paths: List[str],
    seq_len: int,
) -> List[str]:
    """Return a list of `seq_len` radar paths ending at anchor_idx.

    Works backward from anchor_idx collecting frames whose gap to the
    previous selected frame is within _RADAR_GAP_TOLERANCE.  If the
    history is shorter than seq_len, the earliest available frame is
    repeated (forward-fill / repeat padding).
    """
    seq = [radar_paths[anchor_idx]]
    prev_time = radar_times[anchor_idx]
    idx = anchor_idx - 1
    while len(seq) < seq_len and idx >= 0:
        gap = (prev_time - radar_times[idx]).total_seconds()
        if gap <= _RADAR_GAP_TOLERANCE:
            seq.append(radar_paths[idx])
            prev_time = radar_times[idx]
        idx -= 1
    # Pad with earliest available frame if not enough history
    while len(seq) < seq_len:
        seq.append(seq[-1])
    # Reverse so index 0 = oldest, index -1 = current
    seq.reverse()
    return seq


def match_samples(
    pwv_dir: str,
    radar_dir: str,
    rain_dir: str,
    cfg: TrainingConfig = TrainingConfig(),
) -> List[SampleRecord]:
    pwv_map = _build_name_map(pwv_dir)
    radar_map = _build_name_map(radar_dir)
    rain_map = _build_name_map(rain_dir)

    radar_times = []
    radar_paths = []
    for name, path in radar_map.items():
        dt = parse_time(name)
        if dt:
            radar_times.append(dt)
            radar_paths.append(path)

    radar_pairs = sorted(zip(radar_times, radar_paths))
    radar_times = [t for t, _ in radar_pairs]
    radar_paths = [p for _, p in radar_pairs]

    records: List[SampleRecord] = []
    for name, pwv_path in pwv_map.items():
        t1 = parse_time(name)
        if not t1:
            continue

        if t1 < cfg.train_start or t1 > cfg.test_end:
            continue

        if not radar_times:
            continue

        idx = bisect.bisect_left(radar_times, t1)
        candidates = []
        if idx < len(radar_times):
            candidates.append((idx, radar_times[idx], radar_paths[idx]))
        if idx > 0:
            candidates.append((idx - 1, radar_times[idx - 1], radar_paths[idx - 1]))

        best_idx, best_time, best_path = min(
            candidates, key=lambda item: abs((item[1] - t1).total_seconds())
        )
        if abs((best_time - t1).total_seconds()) > 3600:
            continue

        target_paths = []
        for h in (1, 2, 3):
            target_key = (t1 + timedelta(hours=h)).strftime("%Y-%m-%d-%H-%M-%S")
            target_path = rain_map.get(target_key)
            if not target_path:
                target_paths = []
                break
            target_paths.append(target_path)

        if not target_paths:
            continue

        radar_seq = _build_radar_sequence(
            best_idx, radar_times, radar_paths, cfg.radar_seq_len
        )

        records.append(
            SampleRecord(
                timestamp=t1,
                pwv_path=pwv_path,
                radar_path=best_path,
                target_1h_path=target_paths[0],
                target_2h_path=target_paths[1],
                target_3h_path=target_paths[2],
                radar_seq_paths=radar_seq,
            )
        )

    records.sort(key=lambda r: r.timestamp)
    return records
