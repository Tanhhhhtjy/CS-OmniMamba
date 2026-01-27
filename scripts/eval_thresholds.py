import argparse
import os
from dataclasses import replace
from typing import Dict, List, Tuple

import torch

from omnimamba.config import TrainingConfig
from omnimamba.data_match import match_samples
from omnimamba.dataset import build_loaders
from omnimamba.model import CrossAttentionMamba
from omnimamba.splits import split_records


THRESHOLDS = [
    ("No Rain", None, 0.0),
    ("Drizzle Rain", 0.0, 2.5),
    ("Light Rain", 2.5, 8.0),
    ("Moderate Rain", 8.0, 16.0),
    ("Heavy Rain", 16.0, 30.0),
    ("Torrential Rain", 30.0, 50.0),
    ("Extreme Rainstorm", 50.0, None),
]


def _bin_mask(values: torch.Tensor, low: float | None, high: float | None) -> torch.Tensor:
    if low is None:
        return values <= high
    if high is None:
        return values > low
    return (values > low) & (values <= high)


def _update_counts(
    counts: Dict[str, torch.Tensor],
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    for name, low, high in THRESHOLDS:
        pred_mask = _bin_mask(preds, low, high)
        tgt_mask = _bin_mask(targets, low, high)

        hits = (pred_mask & tgt_mask).sum(dim=(0, 2, 3))
        misses = ((~pred_mask) & tgt_mask).sum(dim=(0, 2, 3))
        false_alarms = (pred_mask & (~tgt_mask)).sum(dim=(0, 2, 3))

        counts[name][0] += hits
        counts[name][1] += misses
        counts[name][2] += false_alarms


def _metrics_from_counts(counts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    results: Dict[str, torch.Tensor] = {}
    for name, stats in counts.items():
        hits, misses, false_alarms = stats
        pod = hits / torch.clamp(hits + misses, min=1)
        csi = hits / torch.clamp(hits + misses + false_alarms, min=1)
        far = false_alarms / torch.clamp(hits + false_alarms, min=1)
        results[name] = torch.stack([pod, csi, far], dim=0)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate POD/CSI/FAR by rain bins.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--checkpoint", default="./results/best_model.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--rain-scale",
        type=float,
        default=1.0,
        help="Scale applied to predictions/targets before thresholding.",
    )
    parser.add_argument(
        "--rain-offset",
        type=float,
        default=0.0,
        help="Offset applied to predictions/targets before thresholding.",
    )
    args = parser.parse_args()

    cfg = TrainingConfig()
    cfg = replace(
        cfg,
        epochs=args.epochs if args.epochs is not None else cfg.epochs,
        batch_size=args.batch_size if args.batch_size is not None else cfg.batch_size,
        lr=args.lr if args.lr is not None else cfg.lr,
    )

    data_root = args.data_root
    pwv_dir = os.path.join(data_root, "PWV")
    radar_dir = os.path.join(data_root, "RADAR")
    rain_dir = os.path.join(data_root, "RAIN")

    if not all(os.path.exists(p) for p in (pwv_dir, radar_dir, rain_dir)):
        raise SystemExit("Data folders not found. Expected ./data/PWV, ./data/RADAR, ./data/RAIN")

    records = match_samples(pwv_dir, radar_dir, rain_dir, cfg)
    if not records:
        raise SystemExit("No matched samples found. Check dataset and time ranges.")

    _, _, test_records = split_records(records, cfg)
    if not test_records:
        raise SystemExit("No test samples found for current time split.")

    _, _, test_loader = build_loaders([], [], test_records, cfg)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = CrossAttentionMamba(
        img_size=cfg.img_size,
        img_size_w=cfg.img_size_w,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        d_state=cfg.d_state,
        dim=cfg.dim,
        depth=cfg.depth,
        num_classes=3,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    counts: Dict[str, torch.Tensor] = {
        name: torch.zeros(3, device=device) for name, _, _ in THRESHOLDS
    }

    with torch.no_grad():
        for img1, img2, targets in test_loader:
            img1, img2, targets = img1.to(device), img2.to(device), targets.to(device)
            preds = model(img1, img2)

            preds = preds * args.rain_scale + args.rain_offset
            targets = targets * args.rain_scale + args.rain_offset

            _update_counts(counts, preds, targets)

    metrics = _metrics_from_counts(counts)
    for name, values in metrics.items():
        pod, csi, far = values
        print(f"[{name}]")
        for idx in range(3):
            print(
                f"  T+{idx + 1}h | POD: {pod[idx]:.4f} | CSI: {csi[idx]:.4f} | FAR: {far[idx]:.4f}"
            )


if __name__ == "__main__":
    main()
