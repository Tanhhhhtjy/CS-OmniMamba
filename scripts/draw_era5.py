import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scripts.pipeline_utils import parse_pwv_line


def calculate_metrics(x, y):
    bias = np.mean(y - x)
    mae = mean_absolute_error(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    r = np.corrcoef(x, y)[0, 1]
    return bias, mae, rmse, r


def load_gnss_pwv(gnss_root):
    records = []
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


def nearest_index(sorted_arr, value):
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


def match_gnss_era5(records, ds, max_time_diff_min=30):
    times = ds["valid_time"].values.astype("datetime64[ns]")
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    tcwv = ds["tcwv"].values

    max_diff = np.timedelta64(max_time_diff_min, "m")

    gnss_vals = []
    era5_vals = []

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
        gnss_vals.append(record["pwv"])
        era5_vals.append(float(era5_val))

    return np.array(gnss_vals), np.array(era5_vals)


def parse_args():
    parser = argparse.ArgumentParser(description="ERA5 vs GNSS PWV density plot")
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
        "--output",
        default="ERA5_vs_GNSS_Density_Plot.png",
        help="Output image filename",
    )
    parser.add_argument("--show", action="store_true", help="Show figure window")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.era5):
        raise FileNotFoundError(args.era5)
    if not os.path.exists(args.gnss_root):
        raise FileNotFoundError(args.gnss_root)

    ds = xr.open_dataset(args.era5)
    if "tcwv" not in ds.variables:
        raise ValueError("tcwv not found in ERA5 dataset")

    gnss_records = load_gnss_pwv(args.gnss_root)
    if not gnss_records:
        raise ValueError("No GNSS PWV records found")

    gnss_data, era5_data = match_gnss_era5(
        gnss_records, ds, max_time_diff_min=args.max_time_diff_min
    )
    if gnss_data.size == 0:
        raise ValueError("No matched GNSS/ERA5 pairs found")

    valid_mask = ~np.isnan(gnss_data) & ~np.isnan(era5_data)
    x = gnss_data[valid_mask]
    y = era5_data[valid_mask]

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    bias, mae, rmse, r = calculate_metrics(x, y)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(x, y, c=z, s=5, cmap="jet", edgecolor="none")

    axis_limit = [-2, 92]
    ax.plot(axis_limit, axis_limit, "k--", alpha=0.5, linewidth=1.5)
    ax.set_xlim(axis_limit)
    ax.set_ylim(axis_limit)

    font_label = {"family": "sans-serif", "weight": "bold", "size": 12}
    ax.set_xlabel("GNSS PWV (mm)", fontdict=font_label)
    ax.set_ylabel("ERA5 PWV (mm)", fontdict=font_label)
    ax.grid(True, linestyle="-", alpha=0.6)

    text_str = (
        f"Bias = {bias:.2f} mm\n"
        f"MAE = {mae:.2f} mm\n"
        f"RMSE = {rmse:.2f} mm\n"
        f"R = {r:.3f}"
    )
    props = dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.9, edgecolor="gray")
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=props)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Density", size=11)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to: {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
