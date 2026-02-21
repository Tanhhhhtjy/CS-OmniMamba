import argparse
import os
import numpy as np

from PIL import Image

from scripts.pipeline_utils import parse_timestamp_str


def compute_hist_stats(gray_array):
    flat = gray_array.astype(float).ravel()
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "p10": float(np.percentile(flat, 10)),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
    }


def load_grayscale_image(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def collect_images_in_range(root, start, end):
    files = []
    for name in os.listdir(root):
        if not name.endswith(".png"):
            continue
        ts = parse_timestamp_str(name.replace(".png", ""))
        if start <= ts <= end:
            files.append(os.path.join(root, name))
    return sorted(files)


def compute_stats_for_dir(root):
    arrays = []
    for name in os.listdir(root):
        if not name.endswith(".png"):
            continue
        arrays.append(load_grayscale_image(os.path.join(root, name)))
    if not arrays:
        return None
    stacked = np.concatenate([a.ravel() for a in arrays])
    return compute_hist_stats(stacked)


def compute_stats_for_range(root, start, end):
    files = collect_images_in_range(root, start, end)
    if not files:
        return None
    arrays = [load_grayscale_image(path) for path in files]
    stacked = np.concatenate([a.ravel() for a in arrays])
    return compute_hist_stats(stacked)


def format_stats(stats):
    return (
        f"min={stats['min']}, max={stats['max']}, mean={stats['mean']}, "
        f"p10={stats['p10']}, p50={stats['p50']}, p90={stats['p90']}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare new data vs sota_data stats.")
    parser.add_argument("--sota-root", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD-HH-MM-SS")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD-HH-MM-SS")
    return parser.parse_args()


def main():
    args = parse_args()
    start = parse_timestamp_str(args.start)
    end = parse_timestamp_str(args.end)

    for subdir in ("PWV", "RADAR", "RAIN"):
        sota_dir = os.path.join(args.sota_root, subdir)
        new_dir = os.path.join(args.new_root, subdir)
        if not os.path.isdir(sota_dir) or not os.path.isdir(new_dir):
            print(f"{subdir}: missing directory")
            continue
        sota_stats = compute_stats_for_range(sota_dir, start, end)
        new_stats = compute_stats_for_range(new_dir, start, end)
        print(f"{subdir} sota: {format_stats(sota_stats) if sota_stats else 'no data'}")
        print(f"{subdir} new:  {format_stats(new_stats) if new_stats else 'no data'}")


if __name__ == "__main__":
    main()
