from datetime import datetime, timedelta
import os

import numpy as np


def build_6min_timeline(start, end):
    timeline = []
    current = start
    while current <= end:
        timeline.append(current)
        current += timedelta(minutes=6)
    return timeline


def floor_to_30min(ts):
    minute = 0 if ts.minute < 30 else 30
    return ts.replace(minute=minute, second=0, microsecond=0)


def floor_to_hour(ts):
    return ts.replace(minute=0, second=0, microsecond=0)


def linear_to_grayscale_invert(data, vmin, vmax, gamma=1.0):
    clipped = np.clip(data, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    if gamma and gamma != 1.0:
        normalized = np.clip(normalized, 0.0, 1.0) ** gamma
    grayscale = np.round(normalized * 255.0).astype(np.uint8)
    return 255 - grayscale


def rain_log_to_grayscale_invert(data, max_rainfall):
    clipped = np.clip(data, 0, max_rainfall)
    log_vals = np.log1p(clipped)
    max_log = np.log1p(max_rainfall)
    grayscale = np.round((log_vals / max_log) * 255.0).astype(np.uint8)
    return 255 - grayscale


def parse_pwv_line(line):
    parts = line.strip().split()
    if len(parts) < 16:
        return None
    ts = datetime(
        int(parts[0]),
        int(parts[1]),
        int(parts[2]),
        int(parts[3]),
        int(parts[4]),
        int(parts[5]),
    )
    return {
        "timestamp": ts,
        "site": parts[6],
        "lon": float(parts[7]),
        "lat": float(parts[8]),
        "pwv": float(parts[15]),
    }


def parse_rain_line(line):
    parts = line.strip().split()
    if len(parts) < 14:
        return None
    ts = datetime(int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7]))
    return {
        "station": parts[0],
        "lon": float(parts[2]),
        "lat": float(parts[1]),
        "rainfall": float(parts[13]),
        "timestamp": ts,
    }


def create_highres_grid(lat_range, lon_range, density):
    lon_span = lon_range[1] - lon_range[0]
    lat_span = lat_range[1] - lat_range[0]
    n_lon = int(lon_span * density)
    n_lat = int(lat_span * density)
    lon_centers = np.linspace(
        lon_range[0] + 0.5 / density,
        lon_range[1] - 0.5 / density,
        n_lon,
    )
    lat_centers = np.linspace(
        lat_range[0] + 0.5 / density,
        lat_range[1] - 0.5 / density,
        n_lat,
    )
    grid_lons, grid_lats = np.meshgrid(lon_centers, lat_centers)
    return grid_lats, grid_lons


def downsample_grid(highres, target_shape):
    from scipy.ndimage import zoom

    zoom_factors = (
        target_shape[0] / highres.shape[0],
        target_shape[1] / highres.shape[1],
    )
    return zoom(highres, zoom_factors, order=1)


def create_target_grid(lat_range, lon_range, shape):
    lats = np.linspace(lat_range[0], lat_range[1], shape[0])
    lons = np.linspace(lon_range[0], lon_range[1], shape[1])
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    return grid_lats, grid_lons


def points_to_grid(
    records,
    grid_lons,
    grid_lats,
    value_key,
    kernel="thin_plate_spline",
    smoothing=0.1,
    epsilon=None,
):
    if not records:
        return np.zeros_like(grid_lats)
    lons = np.array([r["lon"] for r in records], dtype=float)
    lats = np.array([r["lat"] for r in records], dtype=float)
    values = np.array([r[value_key] for r in records], dtype=float)
    coords = np.column_stack((lons, lats))
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    if unique_coords.shape[0] != coords.shape[0]:
        sums = np.zeros(unique_coords.shape[0], dtype=float)
        counts = np.zeros(unique_coords.shape[0], dtype=float)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1.0)
        values = sums / counts
        lons = unique_coords[:, 0]
        lats = unique_coords[:, 1]
    if kernel == "idw":
        return idw_interpolate(lons, lats, values, grid_lons, grid_lats, power=1.0)
    return rbf_interpolate(
        lons,
        lats,
        values,
        grid_lons,
        grid_lats,
        kernel=kernel,
        smoothing=smoothing,
        epsilon=epsilon,
    )


def merge_time_ranges(ranges):
    start = None
    end = None
    for item in ranges:
        if item is None:
            continue
        item_start, item_end = item
        if item_start is None or item_end is None:
            continue
        start = item_start if start is None or item_start < start else start
        end = item_end if end is None or item_end > end else end
    return start, end


def resolve_time_range(scanned_range, start_override, end_override):
    if start_override is not None or end_override is not None:
        return start_override, end_override
    return scanned_range


def get_records_for_timestamp(records_by_ts, ts, align_fn):
    key = align_fn(ts)
    return records_by_ts.get(key, [])


def save_grayscale_image(data, path):
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray(data.astype(np.uint8), mode="L")
    img.save(path)


def flip_vertical(data):
    return np.flipud(data)


def default_pwv_kernel():
    return "linear"


def load_radar_grid(path):
    import h5py

    with h5py.File(path, "r") as f:
        data = f["data0"][0, 0, 0, 0, :, :]
    return np.array(data)


def compute_pwv_frame(
    records_by_ts,
    ts,
    grid_lons,
    grid_lats,
    target_shape,
    kernel=None,
    smoothing=0.0,
    epsilon=None,
    vmin=0.0,
    vmax=70.0,
    blur_sigma=4.0,
):
    records = get_records_for_timestamp(records_by_ts, ts, floor_to_30min)
    kernel = default_pwv_kernel() if kernel is None else kernel
    highres = points_to_grid(
        records,
        grid_lons,
        grid_lats,
        value_key="pwv",
        kernel=kernel,
        smoothing=smoothing,
        epsilon=epsilon,
    )
    if blur_sigma and blur_sigma > 0:
        from scipy.ndimage import gaussian_filter

        highres = gaussian_filter(highres, sigma=blur_sigma)
    downsampled = downsample_grid(highres, target_shape)
    flipped = flip_vertical(downsampled)
    return linear_to_grayscale_invert(flipped, vmin=vmin, vmax=vmax)


def compute_rain_frame(records_by_ts, ts, grid_lons, grid_lats, target_shape, kernel="thin_plate_spline", smoothing=0.1, max_rainfall=50.0):
    records = get_records_for_timestamp(records_by_ts, ts, floor_to_hour)
    highres = points_to_grid(records, grid_lons, grid_lats, value_key="rainfall", kernel=kernel, smoothing=smoothing)
    downsampled = downsample_grid(highres, target_shape)
    flipped = flip_vertical(downsampled)
    return rain_log_to_grayscale_invert(flipped, max_rainfall=max_rainfall)


def compute_radar_frame(radar_root, ts, target_shape, vmin=0.0, vmax=100.0, gamma=1.5):
    path = build_radar_filepath(radar_root, ts)
    if not os.path.exists(path):
        return np.full(target_shape, 255, dtype=np.uint8)
    grid = load_radar_grid(path)
    downsampled = downsample_grid(grid, target_shape)
    flipped = flip_vertical(downsampled)
    return linear_to_grayscale_invert(flipped, vmin=vmin, vmax=vmax, gamma=gamma)


def parse_radar_filename(filename):
    time_str = filename.replace(".nc", "")
    return datetime.strptime(time_str, "%Y%m%d_%H%M%S")


def scan_pwv_time_range(pwv_root):
    start = None
    end = None
    for name in os.listdir(pwv_root):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(pwv_root, name)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("YYYY"):
                    continue
                record = parse_pwv_line(line)
                if record is None:
                    continue
                ts = record["timestamp"]
                start = ts if start is None or ts < start else start
                end = ts if end is None or ts > end else end
    return start, end


def scan_rain_time_range(rain_root):
    start = None
    end = None
    for name in os.listdir(rain_root):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(rain_root, name)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Station"):
                    continue
                record = parse_rain_line(line)
                if record is None:
                    continue
                ts = record["timestamp"]
                start = ts if start is None or ts < start else start
                end = ts if end is None or ts > end else end
    return start, end


def scan_radar_time_range(radar_root):
    start = None
    end = None
    for root, _, files in os.walk(radar_root):
        for name in files:
            if not name.endswith(".nc"):
                continue
            ts = parse_radar_filename(name)
            start = ts if start is None or ts < start else start
            end = ts if end is None or ts > end else end
    return start, end


def format_timestamp(ts):
    return ts.strftime("%Y-%m-%d-%H-%M-%S")


def parse_timestamp_str(value):
    return datetime.strptime(value, "%Y-%m-%d-%H-%M-%S")


def rbf_interpolate(
    lons,
    lats,
    values,
    grid_lons,
    grid_lats,
    kernel="thin_plate_spline",
    smoothing=0.1,
    epsilon=None,
):
    from scipy.interpolate import RBFInterpolator

    obs_points = np.column_stack((lons, lats))
    grid_points = np.column_stack((grid_lons.ravel(), grid_lats.ravel()))
    rbf = RBFInterpolator(obs_points, values, kernel=kernel, smoothing=smoothing, epsilon=epsilon)
    interpolated = rbf(grid_points)
    return interpolated.reshape(grid_lats.shape)


def idw_interpolate(lons, lats, values, grid_lons, grid_lats, power=1.0):
    obs_points = np.column_stack((lons, lats))
    grid_points = np.column_stack((grid_lons.ravel(), grid_lats.ravel()))
    diff = grid_points[:, None, :] - obs_points[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    zero_mask = distances == 0

    out = np.empty(grid_points.shape[0], dtype=float)
    has_zero = zero_mask.any(axis=1)
    if np.any(has_zero):
        first_zero = np.argmax(zero_mask, axis=1)
        out[has_zero] = values[first_zero[has_zero]]

    if np.any(~has_zero):
        dist = distances[~has_zero]
        weights = 1.0 / np.power(dist, power)
        out[~has_zero] = (weights * values).sum(axis=1) / weights.sum(axis=1)

    return out.reshape(grid_lats.shape)


def build_radar_filepath(radar_root, ts):
    year_month = ts.strftime("%Y%m")
    year_month_day = ts.strftime("%Y%m%d")
    time_str = ts.strftime("%Y%m%d_%H%M%S")
    return os.path.join(radar_root, year_month, year_month_day, f"{time_str}.nc")


def load_pwv_records(pwv_root):
    records = {}
    for name in os.listdir(pwv_root):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(pwv_root, name)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("YYYY"):
                    continue
                record = parse_pwv_line(line)
                if record is None:
                    continue
                ts = record["timestamp"]
                records.setdefault(ts, []).append(record)
    return records


def load_rain_records(rain_root):
    records = {}
    for name in os.listdir(rain_root):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(rain_root, name)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Station"):
                    continue
                record = parse_rain_line(line)
                if record is None:
                    continue
                ts = record["timestamp"]
                records.setdefault(ts, []).append(record)
    return records
