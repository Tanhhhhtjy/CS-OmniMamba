import argparse
import os

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter

from scripts.pipeline_utils import parse_pwv_line

matplotlib.use("Agg")


def load_gnss_records(gnss_root):
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


def match_station_pairs(records, ds, max_time_diff_min=30):
    times = ds["valid_time"].values.astype("datetime64[ns]")
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    tcwv = ds["tcwv"].values
    max_diff = np.timedelta64(max_time_diff_min, "m")

    pairs = {}
    meta = {}

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
        meta[site] = (record["lon"], record["lat"])

    return pairs, meta


def compute_station_metrics(pairs, meta, min_samples=30):
    lons = []
    lats = []
    mean_bias = []
    rmse = []

    for site, values in pairs.items():
        if len(values) < min_samples:
            continue
        arr = np.array(values, dtype=float)
        gnss = arr[:, 0]
        era5 = arr[:, 1]
        diff = era5 - gnss
        lon, lat = meta[site]
        lons.append(lon)
        lats.append(lat)
        mean_bias.append(np.mean(diff))
        rmse.append(np.sqrt(np.mean(diff ** 2)))

    return np.array(lons), np.array(lats), np.array(mean_bias), np.array(rmse)


def build_boundary():
    urls = [
        "https://geo.datav.aliyun.com/areas_v3/bound/110000_full.json",
        "https://geo.datav.aliyun.com/areas_v3/bound/120000_full.json",
        "https://geo.datav.aliyun.com/areas_v3/bound/130000_full.json",
    ]
    gdfs = [gpd.read_file(url) for url in urls]
    jjj_gdf = pd.concat(gdfs, ignore_index=True)
    total_boundary = jjj_gdf.dissolve()
    minx, miny, maxx, maxy = total_boundary.total_bounds
    lon_min, lon_max = minx - 0.5, maxx + 0.5
    lat_min, lat_max = miny - 0.5, maxy + 0.5
    return total_boundary, (lon_min, lon_max, lat_min, lat_max)


def make_clip_path(boundary):
    geom = boundary.geometry.values[0]
    if geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    else:
        polygons = [geom]

    vertices = []
    codes = []
    for poly in polygons:
        x, y = poly.exterior.coords.xy
        verts = list(zip(x, y))
        vertices.extend(verts)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY])
    return Path(vertices, codes)


def interpolate_field(lons, lats, values, grid_lon, grid_lat, smooth_sigma=1.0):
    rbf = Rbf(lons, lats, values, function="linear")
    grid = rbf(grid_lon, grid_lat)
    if smooth_sigma and smooth_sigma > 0:
        grid = gaussian_filter(grid, sigma=smooth_sigma)
    return grid


def plot_field(ax, grid, boundary, extent, title, cmap, vmin, vmax):
    lon_min, lon_max, lat_min, lat_max = extent
    im = ax.imshow(
        grid,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)

    clip_path = make_clip_path(boundary)
    patch = PathPatch(clip_path, transform=ax.transData)
    im.set_clip_path(patch)
    return im


def main():
    parser = argparse.ArgumentParser(description="Plot Mean Bias and RMSE maps.")
    parser.add_argument("--era5", default="source/2023_05-08_ERA5 PWV.nc")
    parser.add_argument("--gnss-root", default="source/GNSS/PWV")
    parser.add_argument("--max-time-diff-min", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--output", default="ERA5_GNSS_bias_rmse_map.png")
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

    pairs, meta = match_station_pairs(records, ds, args.max_time_diff_min)
    lons, lats, bias, rmse = compute_station_metrics(pairs, meta, args.min_samples)
    if lons.size == 0:
        raise ValueError("No stations met the minimum sample requirement")

    boundary, extent = build_boundary()
    lon_min, lon_max, lat_min, lat_max = extent
    grid_lon = np.linspace(lon_min, lon_max, args.grid_size)
    grid_lat = np.linspace(lat_min, lat_max, args.grid_size)
    lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)

    bias_grid = interpolate_field(
        lons, lats, bias, lon_mesh, lat_mesh, smooth_sigma=args.smooth_sigma
    )
    rmse_grid = interpolate_field(
        lons, lats, rmse, lon_mesh, lat_mesh, smooth_sigma=args.smooth_sigma
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bias_im = plot_field(
        axes[0],
        bias_grid,
        boundary,
        extent,
        "(a) Mean Bias",
        cmap="jet",
        vmin=-6,
        vmax=6,
    )
    rmse_im = plot_field(
        axes[1],
        rmse_grid,
        boundary,
        extent,
        "(b) RMSE",
        cmap="jet",
        vmin=0,
        vmax=10,
    )

    cbar1 = fig.colorbar(bias_im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Mean Bias (mm)")
    cbar2 = fig.colorbar(rmse_im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("RMSE (mm)")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
