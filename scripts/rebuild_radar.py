#!/usr/bin/env python3
import argparse

from scripts.pipeline_utils import parse_timestamp_str
from scripts.rebuild_data import DEFAULT_RADAR_ROOT, rebuild_radar


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild RADAR grayscale dataset.")
    parser.add_argument("--radar-root", default=DEFAULT_RADAR_ROOT)
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD-HH-MM-SS")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD-HH-MM-SS")
    parser.add_argument("--gamma", type=float, default=1.5, help="Gamma correction factor")
    parser.add_argument("--vmax", type=float, default=100.0, help="Upper bound for grayscale mapping")
    return parser.parse_args()


def resolve_override(ts_str):
    return parse_timestamp_str(ts_str) if ts_str else None


def run(radar_root, output_root, start_override, end_override, gamma=1.5, vmax=100.0):
    rebuild_radar(radar_root, output_root, start_override, end_override, gamma=gamma, vmax=vmax)


def main():
    args = parse_args()
    start_override = resolve_override(args.start)
    end_override = resolve_override(args.end)
    run(
        args.radar_root,
        args.output_root,
        start_override,
        end_override,
        gamma=args.gamma,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
