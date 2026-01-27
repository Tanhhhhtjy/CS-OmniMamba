from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import bisect
import os

from .config import TrainingConfig


@dataclass
class SampleRecord:
    timestamp: datetime
    pwv_path: str
    radar_path: str
    target_1h_path: str
    target_2h_path: str
    target_3h_path: str


def parse_time(name: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d-%H-%M-%S", "%Y-%m-%d-%H-%M"):
        try:
            return datetime.strptime(name, fmt)
        except ValueError:
            continue
    return None


def _build_name_map(folder: str) -> Dict[str, str]:
    return {
        os.path.splitext(fname)[0]: os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, fname))
    }


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
            candidates.append((radar_times[idx], radar_paths[idx]))
        if idx > 0:
            candidates.append((radar_times[idx - 1], radar_paths[idx - 1]))

        best = min(candidates, key=lambda item: abs((item[0] - t1).total_seconds()))
        if abs((best[0] - t1).total_seconds()) > 3600:
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

        records.append(
            SampleRecord(
                timestamp=t1,
                pwv_path=pwv_path,
                radar_path=best[1],
                target_1h_path=target_paths[0],
                target_2h_path=target_paths[1],
                target_3h_path=target_paths[2],
            )
        )

    return records
