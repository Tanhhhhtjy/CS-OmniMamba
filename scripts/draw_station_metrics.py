import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scripts.pipeline_utils import parse_pwv_line


def load_gnss_records(gnss_root: str) -> List[Dict]:
    records: List[Dict] = []
    for root, _, files in os.walk(gnss_root):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    record = parse_pwv_line(line)
                    if record is not None:
                        records.append(record)
    return records


def nearest_index(sorted_arr: np.ndarray, value: np.datetime64) -> int:
    idx = np.searchsorted(sorted_arr, value)
    if idx <= 0:
        return 0
    if idx >= len(sorted_arr):
        return len(sorted_arr) - 1
    before = sorted_arr[idx - 1]
    after = sorted_arr[idx]
    if abs(value - before) <= abs(after - value):
        return idx - 1
    return idx


def match_station_series(
    records: List[Dict],
    ds: xr.Dataset,
    max_time_diff_min: int,
) -> Dict[str, List[Tuple[float, float]]]:
    times = ds["valid_time"].values.astype("datetime64[ns]")
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    tcwv = ds["tcwv"].values

    max_diff = np.timedelta64(max_time_diff_min, "m")
    pairs: Dict[str, List[Tuple[float, float]]] = {}

    for record in records:
        ts = np.datetime64(record["timestamp"])
        if ts < times[0] or ts > times[-1]:
            continue
        t_idx = nearest_index(times, ts)
        if abs(times[t_idx] - ts) > max_diff:
            continue

        lat_idx = int(np.abs(lats - record["lat"]).argmin())
        lon_idx = int(np.abs(lons - record["lon"]).argmin())
        era5_val = tcwv[t_idx, lat_idx, lon_idx]
        if np.isnan(era5_val) or np.isnan(record["pwv"]):
            continue

        site = record["site"]
        pairs.setdefault(site, []).append((record["pwv"], float(era5_val)))

    return pairs


def compute_station_metrics(
    pairs: Dict[str, List[Tuple[float, float]]]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    stations = []
    mean_bias = []
    rms = []

    for site, values in pairs.items():
        if not values:
            continue
        arr = np.array(values, dtype=float)
        gnss = arr[:, 0]
        era5 = arr[:, 1]
        diff = era5 - gnss
        stations.append(site)
        mean_bias.append(np.mean(diff))
        rms.append(np.sqrt(np.mean(diff ** 2)))

    return stations, np.array(mean_bias), np.array(rms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot station-wise mean bias and RMS.")
    parser.add_argument(
        "--era5",
        default="source/2023_05-08_ERA5 PWV.nc",
        help="ERA5 PWV NetCDF path",
    )
    parser.add_argument(
        "--gnss-root",
        default="source/GNSS/PWV",
        help="GNSS PWV folder",
    )
    parser.add_argument(
        "--max-time-diff-min",
        type=int,
        default=30,
        help="Max time difference (minutes) for pairing",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum paired samples per station",
    )
    parser.add_argument(
        "--output",
        default="ERA5_GNSS_station_bias_rms.png",
        help="Output image filename",
    )
    parser.add_argument("--show", action="store_true", help="Show figure window")
    args = parser.parse_args()

    if not os.path.exists(args.era5):
        raise FileNotFoundError(args.era5)
    if not os.path.exists(args.gnss_root):
        raise FileNotFoundError(args.gnss_root)

    ds = xr.open_dataset(args.era5)
    if "tcwv" not in ds.variables:
        raise ValueError("tcwv not found in ERA5 dataset")

    records = load_gnss_records(args.gnss_root)
    if not records:
        raise ValueError("No GNSS PWV records found")

    pairs = match_station_series(records, ds, args.max_time_diff_min)
    pairs = {k: v for k, v in pairs.items() if len(v) >= args.min_samples}
    if not pairs:
        raise ValueError("No stations met the minimum sample requirement")

    stations, mean_bias, rms = compute_station_metrics(pairs)

    order = np.argsort(stations)
    mean_bias = mean_bias[order]
    rms = rms[order]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mean_bias, label="Mean Bias", color="#4C72B0", linewidth=1.5)
    ax.plot(rms, label="RMS", color="#DD8452", linewidth=1.5)

    ax.set_xlabel("GNSS stations")
    ax.set_ylabel("Mean Bias (mm)")
    ax.set_xlim(0, len(mean_bias) - 1)
    ax.set_ylim(-8, 8)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to: {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
