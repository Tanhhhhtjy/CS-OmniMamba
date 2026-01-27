import argparse
import random
from dataclasses import replace
import os

import numpy as np
import torch

from omnimamba.config import TrainingConfig
from omnimamba.constants import RESULTS_DIR_DEFAULT
from omnimamba.data_match import match_samples
from omnimamba.dataset import build_loaders
from omnimamba.model import CrossAttentionMamba
from omnimamba.splits import split_records
from omnimamba.train_loop import train, validate_epoch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniMamba training")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = TrainingConfig()
    cfg = replace(
        cfg,
        epochs=args.epochs if args.epochs is not None else cfg.epochs,
        batch_size=args.batch_size if args.batch_size is not None else cfg.batch_size,
        lr=args.lr if args.lr is not None else cfg.lr,
    )

    _set_seed(args.seed)

    data_root = args.data_root
    pwv_dir = os.path.join(data_root, "PWV")
    radar_dir = os.path.join(data_root, "RADAR")
    rain_dir = os.path.join(data_root, "RAIN")

    if not all(os.path.exists(p) for p in (pwv_dir, radar_dir, rain_dir)):
        raise SystemExit("Data folders not found. Expected ./data/PWV, ./data/RADAR, ./data/RAIN")

    records = match_samples(pwv_dir, radar_dir, rain_dir, cfg)
    if not records:
        raise SystemExit("No matched samples found. Check dataset and time ranges.")

    train_records, val_records, test_records = split_records(records, cfg)

    train_loader, val_loader, test_loader = build_loaders(
        train_records, val_records, test_records, cfg
    )

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

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    train(model, train_loader, val_loader, device, cfg, results_dir)

    val_loss, val_metrics = validate_epoch(model, val_loader, device)
    test_loss, test_metrics = validate_epoch(model, test_loader, device)

    print(f"Val MSE: {val_loss:.5f}")
    print(f"Test MSE: {test_loss:.5f}")
    for t in range(3):
        print(
            f"Val T+{t + 1}h | MAE: {val_metrics['mae'][t]:.4f} | MAPE: {val_metrics['mape'][t]:.2f}% | PSNR: {val_metrics['psnr'][t]:.2f} | SSIM: {val_metrics['ssim'][t]:.4f}"
        )
    for t in range(3):
        print(
            f"Test T+{t + 1}h | MAE: {test_metrics['mae'][t]:.4f} | MAPE: {test_metrics['mape'][t]:.2f}% | PSNR: {test_metrics['psnr'][t]:.2f} | SSIM: {test_metrics['ssim'][t]:.4f}"
        )


if __name__ == "__main__":
    main()
