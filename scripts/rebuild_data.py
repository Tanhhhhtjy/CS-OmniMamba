import argparse
import os

from scripts.pipeline_utils import (
    build_6min_timeline,
    compute_pwv_frame,
    compute_radar_frame,
    compute_rain_frame,
    create_highres_grid,
    format_timestamp,
    load_pwv_records,
    load_rain_records,
    parse_timestamp_str,
    resolve_time_range,
    save_grayscale_image,
    scan_pwv_time_range,
    scan_radar_time_range,
    scan_rain_time_range,
)

LAT_RANGE = (36, 43)
LON_RANGE = (113, 120)
TARGET_SHAPE = (66, 70)
DENSITY = 85


DEFAULT_PWV_ROOT = "/root/autodl-tmp/tanh/预备数据/PWV/2023-河北PWV数据"
DEFAULT_RAIN_ROOT = "/root/autodl-tmp/tanh/预备数据/RAIN/2023-河北"
DEFAULT_RADAR_ROOT = "/root/autodl-tmp/tanh/预备数据/RADAR"


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild PWV/RADAR/RAIN grayscale datasets.")
    parser.add_argument("--pwv-root", default=DEFAULT_PWV_ROOT)
    parser.add_argument("--rain-root", default=DEFAULT_RAIN_ROOT)
    parser.add_argument("--radar-root", default=DEFAULT_RADAR_ROOT)
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD-HH-MM-SS")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD-HH-MM-SS")
    return parser.parse_args()


def resolve_override(ts_str):
    return parse_timestamp_str(ts_str) if ts_str else None


def rebuild_pwv(pwv_root, output_root, start_override, end_override, resume=False):
    scanned = scan_pwv_time_range(pwv_root)
    start, end = resolve_time_range(scanned, start_override, end_override)
    if start is None or end is None:
        raise ValueError("PWV time range not found")

    records = load_pwv_records(pwv_root)
    grid_lats, grid_lons = create_highres_grid(LAT_RANGE, LON_RANGE, DENSITY)
    timeline = build_6min_timeline(start, end)

    out_dir = os.path.join(output_root, "PWV")
    os.makedirs(out_dir, exist_ok=True)
    for idx, ts in enumerate(timeline, start=1):
        filename = f"{format_timestamp(ts)}.png"
        out_path = os.path.join(out_dir, filename)
        if resume and os.path.exists(out_path):
            continue
        frame = compute_pwv_frame(
            records,
            ts,
            grid_lons,
            grid_lats,
            TARGET_SHAPE,
            kernel="linear",
            smoothing=0.0,
            blur_sigma=4.0,
        )
        save_grayscale_image(frame, out_path)
        if idx % 200 == 0:
            print(f"PWV progress: {idx}/{len(timeline)}")


def rebuild_rain(rain_root, output_root, start_override, end_override, resume=False):
    scanned = scan_rain_time_range(rain_root)
    start, end = resolve_time_range(scanned, start_override, end_override)
    if start is None or end is None:
        raise ValueError("RAIN time range not found")

    records = load_rain_records(rain_root)
    grid_lats, grid_lons = create_highres_grid(LAT_RANGE, LON_RANGE, DENSITY)
    timeline = build_6min_timeline(start, end)

    out_dir = os.path.join(output_root, "RAIN")
    os.makedirs(out_dir, exist_ok=True)
    for idx, ts in enumerate(timeline, start=1):
        filename = f"{format_timestamp(ts)}.png"
        out_path = os.path.join(out_dir, filename)
        if resume and os.path.exists(out_path):
            continue
        frame = compute_rain_frame(records, ts, grid_lons, grid_lats, TARGET_SHAPE)
        save_grayscale_image(frame, out_path)
        if idx % 200 == 0:
            print(f"RAIN progress: {idx}/{len(timeline)}")


def rebuild_radar(radar_root, output_root, start_override, end_override, gamma=1.5, vmax=100.0):
    scanned = scan_radar_time_range(radar_root)
    start, end = resolve_time_range(scanned, start_override, end_override)
    if start is None or end is None:
        raise ValueError("RADAR time range not found")

    timeline = build_6min_timeline(start, end)
    out_dir = os.path.join(output_root, "RADAR")
    os.makedirs(out_dir, exist_ok=True)
    for idx, ts in enumerate(timeline, start=1):
        frame = compute_radar_frame(radar_root, ts, TARGET_SHAPE, gamma=gamma, vmax=vmax)
        filename = f"{format_timestamp(ts)}.png"
        save_grayscale_image(frame, os.path.join(out_dir, filename))
        if idx % 200 == 0:
            print(f"RADAR progress: {idx}/{len(timeline)}")


def main():
    args = parse_args()
    start_override = resolve_override(args.start)
    end_override = resolve_override(args.end)

    rebuild_pwv(args.pwv_root, args.output_root, start_override, end_override)
    rebuild_rain(args.rain_root, args.output_root, start_override, end_override)
    rebuild_radar(args.radar_root, args.output_root, start_override, end_override)


if __name__ == "__main__":
    main()
