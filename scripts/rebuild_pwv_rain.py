#!/usr/bin/env python3
import argparse
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import os

from scripts.pipeline_utils import (
    build_6min_timeline,
    compute_pwv_frame,
    compute_rain_frame,
    create_highres_grid,
    format_timestamp,
    load_pwv_records,
    load_rain_records,
    parse_timestamp_str,
    resolve_time_range,
    save_grayscale_image,
    scan_pwv_time_range,
    scan_rain_time_range,
)
from scripts.rebuild_data import (
    DEFAULT_PWV_ROOT,
    DEFAULT_RAIN_ROOT,
    DENSITY,
    LAT_RANGE,
    LON_RANGE,
    TARGET_SHAPE,
    rebuild_pwv,
    rebuild_rain,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild PWV/RAIN grayscale datasets.")
    parser.add_argument("--pwv-root", default=DEFAULT_PWV_ROOT)
    parser.add_argument("--rain-root", default=DEFAULT_RAIN_ROOT)
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD-HH-MM-SS")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD-HH-MM-SS")
    parser.add_argument("--resume", action="store_true", help="Skip existing output files")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=200,
        help="Report progress every N frames (parallel mode only)",
    )
    return parser.parse_args()


def resolve_override(ts_str):
    return parse_timestamp_str(ts_str) if ts_str else None


_PWV_RECORDS = None
_PWV_GRID_LATS = None
_PWV_GRID_LONS = None
_PWV_OUT_DIR = None
_PWV_RESUME = False

_RAIN_RECORDS = None
_RAIN_GRID_LATS = None
_RAIN_GRID_LONS = None
_RAIN_OUT_DIR = None
_RAIN_RESUME = False


def _init_pwv_worker(records, grid_lats, grid_lons, out_dir, resume):
    global _PWV_RECORDS, _PWV_GRID_LATS, _PWV_GRID_LONS, _PWV_OUT_DIR, _PWV_RESUME
    _PWV_RECORDS = records
    _PWV_GRID_LATS = grid_lats
    _PWV_GRID_LONS = grid_lons
    _PWV_OUT_DIR = out_dir
    _PWV_RESUME = resume


def _init_rain_worker(records, grid_lats, grid_lons, out_dir, resume):
    global _RAIN_RECORDS, _RAIN_GRID_LATS, _RAIN_GRID_LONS, _RAIN_OUT_DIR, _RAIN_RESUME
    _RAIN_RECORDS = records
    _RAIN_GRID_LATS = grid_lats
    _RAIN_GRID_LONS = grid_lons
    _RAIN_OUT_DIR = out_dir
    _RAIN_RESUME = resume


def _pwv_worker(ts):
    filename = f"{format_timestamp(ts)}.png"
    out_path = os.path.join(_PWV_OUT_DIR, filename)
    if _PWV_RESUME and os.path.exists(out_path):
        return
    frame = compute_pwv_frame(
        _PWV_RECORDS,
        ts,
        _PWV_GRID_LONS,
        _PWV_GRID_LATS,
        TARGET_SHAPE,
        kernel="linear",
        smoothing=0.0,
        blur_sigma=4.0,
    )
    save_grayscale_image(frame, out_path)


def _rain_worker(ts):
    filename = f"{format_timestamp(ts)}.png"
    out_path = os.path.join(_RAIN_OUT_DIR, filename)
    if _RAIN_RESUME and os.path.exists(out_path):
        return
    frame = compute_rain_frame(_RAIN_RECORDS, ts, _RAIN_GRID_LONS, _RAIN_GRID_LATS, TARGET_SHAPE)
    save_grayscale_image(frame, out_path)


def _parallel_rebuild_pwv(
    pwv_root,
    output_root,
    start_override,
    end_override,
    resume,
    workers,
    progress_interval,
):
    scanned = scan_pwv_time_range(pwv_root)
    start, end = resolve_time_range(scanned, start_override, end_override)
    if start is None or end is None:
        raise ValueError("PWV time range not found")

    records = load_pwv_records(pwv_root)
    grid_lats, grid_lons = create_highres_grid(LAT_RANGE, LON_RANGE, DENSITY)
    timeline = build_6min_timeline(start, end)

    out_dir = os.path.join(output_root, "PWV")
    os.makedirs(out_dir, exist_ok=True)

    ctx = mp.get_context("fork")
    try:
        pool = ctx.Pool(
            processes=workers,
            initializer=_init_pwv_worker,
            initargs=(records, grid_lats, grid_lons, out_dir, resume),
        )
    except PermissionError:
        pool = None
    if pool is not None:
        with pool:
            for idx, _ in enumerate(pool.imap_unordered(_pwv_worker, timeline), start=1):
                if progress_interval and idx % progress_interval == 0:
                    print(f"PWV progress: {idx}/{len(timeline)}")
        return
    _init_pwv_worker(records, grid_lats, grid_lons, out_dir, resume)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, _ in enumerate(executor.map(_pwv_worker, timeline), start=1):
            if progress_interval and idx % progress_interval == 0:
                print(f"PWV progress: {idx}/{len(timeline)}")


def _parallel_rebuild_rain(
    rain_root,
    output_root,
    start_override,
    end_override,
    resume,
    workers,
    progress_interval,
):
    scanned = scan_rain_time_range(rain_root)
    start, end = resolve_time_range(scanned, start_override, end_override)
    if start is None or end is None:
        raise ValueError("RAIN time range not found")

    records = load_rain_records(rain_root)
    grid_lats, grid_lons = create_highres_grid(LAT_RANGE, LON_RANGE, DENSITY)
    timeline = build_6min_timeline(start, end)

    out_dir = os.path.join(output_root, "RAIN")
    os.makedirs(out_dir, exist_ok=True)

    ctx = mp.get_context("fork")
    try:
        pool = ctx.Pool(
            processes=workers,
            initializer=_init_rain_worker,
            initargs=(records, grid_lats, grid_lons, out_dir, resume),
        )
    except PermissionError:
        pool = None
    if pool is not None:
        with pool:
            for idx, _ in enumerate(pool.imap_unordered(_rain_worker, timeline), start=1):
                if progress_interval and idx % progress_interval == 0:
                    print(f"RAIN progress: {idx}/{len(timeline)}")
        return
    _init_rain_worker(records, grid_lats, grid_lons, out_dir, resume)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, _ in enumerate(executor.map(_rain_worker, timeline), start=1):
            if progress_interval and idx % progress_interval == 0:
                print(f"RAIN progress: {idx}/{len(timeline)}")


def run(
    pwv_root,
    rain_root,
    output_root,
    start_override,
    end_override,
    resume=False,
    workers=1,
    progress_interval=200,
):
    if workers <= 1:
        rebuild_pwv(pwv_root, output_root, start_override, end_override, resume=resume)
        rebuild_rain(rain_root, output_root, start_override, end_override, resume=resume)
        return
    _parallel_rebuild_pwv(
        pwv_root,
        output_root,
        start_override,
        end_override,
        resume,
        workers,
        progress_interval,
    )
    _parallel_rebuild_rain(
        rain_root,
        output_root,
        start_override,
        end_override,
        resume,
        workers,
        progress_interval,
    )


def main():
    args = parse_args()
    start_override = resolve_override(args.start)
    end_override = resolve_override(args.end)
    run(
        args.pwv_root,
        args.rain_root,
        args.output_root,
        start_override,
        end_override,
        resume=args.resume,
        workers=args.workers,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
