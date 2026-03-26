"""Offline visualization for precipitation nowcasting runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for src imports
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# ── Color map ────────────────────────────────────────────────────────────────


def get_custom_rain_cmap() -> ListedColormap:
    """7-level discrete rain colormap (strong→none = dark brown→white)."""
    colors_rgb = [
        (97, 40, 31),  # 0: deep brown — heavy rain
        (250, 1, 246),  # 1: magenta
        (0, 0, 254),  # 2: deep blue
        (101, 183, 252),  # 3: light blue
        (61, 185, 63),  # 4: deep green
        (166, 242, 142),  # 5: light green
        (254, 254, 254),  # 6: white — no rain
    ]
    colors_norm = [(r / 255, g / 255, b / 255) for r, g, b in colors_rgb]
    return ListedColormap(colors_norm, name="custom_rain_discrete")


def get_pwv_cmap(discrete: bool = False) -> LinearSegmentedColormap:
    """PWV colormap aligned with predict_final.py."""
    colors_rgb = [
        (0, 0, 139),
        (0, 50, 200),
        (0, 120, 255),
        (0, 180, 255),
        (0, 220, 200),
        (50, 255, 150),
        (150, 255, 50),
        (255, 255, 0),
        (255, 180, 0),
        (255, 100, 0),
        (220, 0, 0),
    ]
    colors_norm = [(r / 255, g / 255, b / 255) for r, g, b in colors_rgb]
    n = 11 if discrete else 512
    return LinearSegmentedColormap.from_list("custom_pwv_discrete" if discrete else "custom_pwv_smooth", colors_norm, N=n)


def get_radar_cmap(discrete: bool = False) -> LinearSegmentedColormap:
    """RADAR colormap aligned with predict_final.py."""
    colors_rgb = [
        (255, 50, 10),
        (255, 100, 20),
        (255, 160, 40),
        (255, 220, 60),
        (200, 255, 100),
        (140, 255, 180),
        (100, 255, 220),
        (80, 220, 255),
        (60, 180, 240),
        (40, 120, 200),
        (25, 60, 140),
        (15, 30, 80),
        (5, 10, 30),
    ]
    colors_norm = [(r / 255, g / 255, b / 255) for r, g, b in colors_rgb]
    n = 13 if discrete else 512
    return LinearSegmentedColormap.from_list("custom_radar_discrete" if discrete else "custom_radar_smooth", colors_norm, N=n)


def get_rain_cmap(discrete: bool = False) -> LinearSegmentedColormap:
    """RAIN colormap aligned with predict_final.py."""
    colors_rgb = [
        (97, 40, 31),
        (250, 1, 246),
        (0, 0, 254),
        (101, 183, 252),
        (61, 185, 63),
        (166, 242, 142),
        (255, 255, 255),
    ]
    colors_norm = [(r / 255, g / 255, b / 255) for r, g, b in colors_rgb]
    n = 7 if discrete else 256
    return LinearSegmentedColormap.from_list("custom_rain_discrete" if discrete else "custom_rain_smooth", colors_norm, N=n)


def _to_display_space(arr: torch.Tensor) -> torch.Tensor:
    """
    Convert current project tensors (1 = stronger signal) back to the
    predict_final.py display convention (0 = stronger signal).
    """
    return (1.0 - arr).clamp(0.0, 1.0)


# ── Training curves ──────────────────────────────────────────────────────────


def plot_curves(run_dir: Path, csi_persistence: float | None = None) -> None:
    """Read metrics.json and plot 3-row training curves."""
    metrics_path = run_dir / "metrics.json"
    data = json.loads(metrics_path.read_text())

    epochs = [d["epoch"] for d in data]
    train_loss = [d["train_loss"] for d in data]
    val_mse = [d["val_mse"] for d in data]
    val_csi = [d["val_csi_weak"] for d in data]
    val_far = [d["val_far_weak"] for d in data]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(epochs, train_loss, "o-", label="train_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_mse, "s-", color="tab:orange", label="val_mse")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_csi, "^-", color="tab:green", label="val_csi_weak")
    axes[2].plot(epochs, val_far, "v-", color="tab:red", label="val_far_weak")
    if csi_persistence is not None:
        axes[2].axhline(
            csi_persistence,
            ls="--",
            color="gray",
            label=f"persistence CSI={csi_persistence:.4f}",
        )
    axes[2].set_ylabel("CSI / FAR")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves — {run_dir.name}")
    fig.tight_layout()
    vis_dir = run_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    fig.savefig(vis_dir / "loss_curve.png", dpi=150)
    plt.close(fig)


# ── Model loading helper ─────────────────────────────────────────────────────


def _load_model(run_dir: Path, epoch: int | None, device: torch.device):
    """
    Load model from checkpoint.

    NOTE: Currently hardcoded to ConvLSTMModel. Future improvement:
    - Read model type from run_dir/config.json or checkpoint metadata
    - Support dynamic model loading for SimVP/PredRNN/MambaCast/etc.
    """
    from src.config import T
    from src.model_convlstm import ConvLSTMModel as StubModel

    if epoch is not None:
        ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
    else:
        ckpt_path = run_dir / "last.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Available checkpoints in {run_dir}:\n"
            f"{list(run_dir.glob('*.pt'))}"
        )

    model = StubModel(t=T).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


# ── Sample comparison ────────────────────────────────────────────────────────


def plot_samples_from_tensors(
    run_dir: Path,
    radar_frames: torch.Tensor,  # [B, 1, H, W]
    pwv_frames: torch.Tensor,  # [B, 1, H, W]
    pred: torch.Tensor,  # [B, num_steps, H, W]
    target: torch.Tensor,  # [B, num_steps, H, W]
    epoch: int | None,
    diff_vmax: float,
) -> None:
    """Plot sample comparison grid aligned with predict_final.py."""
    num_steps = pred.shape[1]
    if num_steps not in (1, 3):
        raise ValueError(f"num_steps must be 1 or 3, got {num_steps}")

    pwv_display = _to_display_space(pwv_frames).cpu()
    radar_display = _to_display_space(radar_frames).cpu()
    pred_display = _to_display_space(pred).cpu()
    target_display = _to_display_space(target).cpu()

    pwv_cmap = get_pwv_cmap(discrete=True)
    radar_cmap = get_radar_cmap(discrete=True)
    rain_cmap = get_rain_cmap(discrete=True)

    if num_steps == 1:
        fig, axes = plt.subplots(pred.shape[0], 4, figsize=(20, 4 * pred.shape[0]), squeeze=False)
        for row in range(pred.shape[0]):
            pred_show = pred_display[row, 0].clone()
            pred_show[pred_show > 0.95] = 1.0
            panels = [
                (axes[row, 0], pwv_display[row, 0].numpy(), pwv_cmap, "PWV", "bicubic"),
                (axes[row, 1], radar_display[row, 0].numpy(), radar_cmap, "RADAR", "bicubic"),
                (axes[row, 2], pred_show.numpy(), rain_cmap, "Pred +1h", "bicubic"),
                (axes[row, 3], target_display[row, 0].numpy(), rain_cmap, "True +1h", "bicubic"),
            ]
            for ax, img, cmap, title, interp in panels:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation=interp)
                if row == 0:
                    ax.set_title(title, fontsize=12, fontweight="bold")
                ax.axis("off")
    else:
        fig, axes = plt.subplots(pred.shape[0] * 2, 4, figsize=(20, 8 * pred.shape[0]), squeeze=False)
        for row in range(pred.shape[0]):
            pred_show = pred_display[row].clone()
            pred_show[pred_show > 0.95] = 1.0
            top = row * 2
            bottom = top + 1
            panels = [
                (axes[top, 0], pwv_display[row, 0].numpy(), pwv_cmap, "PWV"),
                (axes[top, 1], radar_display[row, 0].numpy(), radar_cmap, "RADAR"),
                (axes[top, 2], pred_show[0].numpy(), rain_cmap, "Pred +1h"),
                (axes[top, 3], target_display[row, 0].numpy(), rain_cmap, "True +1h"),
                (axes[bottom, 0], pred_show[1].numpy(), rain_cmap, "Pred +2h"),
                (axes[bottom, 1], target_display[row, 1].numpy(), rain_cmap, "True +2h"),
                (axes[bottom, 2], pred_show[2].numpy(), rain_cmap, "Pred +3h"),
                (axes[bottom, 3], target_display[row, 2].numpy(), rain_cmap, "True +3h"),
            ]
            for ax, img, cmap, title in panels:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="bicubic")
                if row == 0:
                    ax.set_title(title, fontsize=12, fontweight="bold")
                ax.axis("off")

    vis_dir = run_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    fname = f"vis_epoch_{epoch:03d}.png" if epoch is not None else "vis_last.png"
    fig.savefig(vis_dir / fname, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.1)
    plt.close(fig)


def plot_samples(
    run_dir: Path,
    epoch: int | None,
    n_samples: int,
    split: str,
    diff_vmax: float,
    device: torch.device,
) -> None:
    """Load checkpoint, run inference on first batch, plot comparison."""
    from torch.utils.data import DataLoader

    from src.config import RADAR_PREP_DIR, RAIN_DIR, PWV_DIR
    from src.dataset import RainDataset
    from src.train import _filter_dataset_by_split

    model = _load_model(run_dir, epoch, device)

    ds = _filter_dataset_by_split(
        RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR), split
    )
    if len(ds) == 0:
        raise ValueError(
            f"Split '{split}' has no samples. Check date ranges in src/config.py"
        )

    loader = DataLoader(ds, batch_size=n_samples, shuffle=False, num_workers=0)
    try:
        batch = next(iter(loader))
    except StopIteration:
        raise ValueError(f"DataLoader for split '{split}' is empty")

    radar = batch["radar"].to(device)  # [B, T, 1, H, W]
    pwv = batch["pwv"].to(device)  # [B, 1, 1, H, W]
    rain = batch["rain"].to(device)  # [B, 1, 1, H, W]

    with torch.no_grad():
        pred = model(radar, pwv)  # [B, 1, H, W]

    # NOTE: Current single-step assumption. For future multi-step support:
    # - dataset should return rain: [B, num_steps, 1, H, W]
    # - model should return pred: [B, num_steps, H, W]
    # - Then: target = rain.squeeze(2) to get [B, num_steps, H, W]
    radar_last = radar[:, -1]  # [B, 1, H, W]
    pwv_squeezed = pwv[:, 0]  # [B, 1, H, W]
    target = rain[:, 0]  # [B, 1, H, W]

    plot_samples_from_tensors(
        run_dir, radar_last, pwv_squeezed, pred, target, epoch, diff_vmax
    )


# ── Threshold sensitivity ────────────────────────────────────────────────────


def plot_threshold_from_tensors(
    run_dir: Path,
    all_preds: list[torch.Tensor],  # list of [B, *, H, W]
    all_targets: list[torch.Tensor],
) -> None:
    """Sweep thresholds and plot CSI/POD/FAR curves."""
    from src.metrics import THRESH_STRONG, THRESH_WEAK

    thresholds = np.arange(0, 1.01, 0.02)
    n_thresh = len(thresholds)

    # Use parallel arrays to accumulate TP/FP/FN, avoiding float dict keys
    tp_accum = np.zeros(n_thresh)
    fp_accum = np.zeros(n_thresh)
    fn_accum = np.zeros(n_thresh)

    # Stream accumulation to avoid concatenating large tensors
    for pred, target in zip(all_preds, all_targets):
        for i, th in enumerate(thresholds):
            pred_bin = (pred > th).float()
            target_bin = (target > th).float()
            tp_accum[i] += (pred_bin * target_bin).sum().item()
            fp_accum[i] += (pred_bin * (1 - target_bin)).sum().item()
            fn_accum[i] += ((1 - pred_bin) * target_bin).sum().item()

    # Compute CSI/POD/FAR, preserving NaN semantics from src.metrics
    csi_vals, pod_vals, far_vals = [], [], []
    for i in range(n_thresh):
        tp, fp, fn = tp_accum[i], fp_accum[i], fn_accum[i]
        denom_pod = tp + fn
        denom_far = tp + fp
        denom_csi = tp + fp + fn

        # Match src.metrics.compute_csi_pod_far NaN semantics:
        # - CSI/POD: NaN when no rain in target (denom_pod == 0)
        # - FAR: NaN when model predicts no rain (denom_far == 0)
        csi = tp / denom_csi if denom_pod > 0 else float("nan")
        pod = tp / denom_pod if denom_pod > 0 else float("nan")
        far = fp / denom_far if denom_far > 0 else float("nan")

        csi_vals.append(csi)
        pod_vals.append(pod)
        far_vals.append(far)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, csi_vals, label="CSI")
    ax.plot(thresholds, pod_vals, label="POD")
    ax.plot(thresholds, far_vals, label="FAR")
    ax.axvline(
        THRESH_WEAK,
        ls="--",
        color="gray",
        alpha=0.7,
        label=f"THRESH_WEAK={THRESH_WEAK:.4f}",
    )
    ax.axvline(
        THRESH_STRONG,
        ls="--",
        color="dimgray",
        alpha=0.7,
        label=f"THRESH_STRONG={THRESH_STRONG:.4f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sensitivity — {run_dir.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    vis_dir = run_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    fig.savefig(vis_dir / "threshold_curve.png", dpi=150)
    plt.close(fig)


def plot_threshold(
    run_dir: Path,
    epoch: int | None,
    split: str,
    device: torch.device,
) -> None:
    """Load checkpoint, run inference on full split, plot threshold curve."""
    from torch.utils.data import DataLoader

    from src.config import RADAR_PREP_DIR, RAIN_DIR, PWV_DIR
    from src.dataset import RainDataset
    from src.train import _filter_dataset_by_split

    model = _load_model(run_dir, epoch, device)

    ds = _filter_dataset_by_split(
        RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR), split
    )
    if len(ds) == 0:
        raise ValueError(
            f"Split '{split}' has no samples. Check date ranges in src/config.py"
        )

    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["radar"].to(device), batch["pwv"].to(device))
            all_preds.append(pred.cpu())
            all_targets.append(batch["rain"].squeeze(1).cpu())

    plot_threshold_from_tensors(run_dir, all_preds, all_targets)


# ── CLI entry point ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visualization for precipitation nowcasting"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Run directory (absolute path or relative under runs/)",
    )
    parser.add_argument(
        "--mode",
        choices=["curves", "samples", "threshold", "all"],
        default="all",
    )
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--csi-persistence", type=float, default=None)
    parser.add_argument("--diff-vmax", type=float, default=0.5)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_name)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = Path("runs") / args.run_name
    device = torch.device(args.device)

    if args.mode in ("curves", "all"):
        print(f"Plotting training curves → {run_dir / 'vis' / 'loss_curve.png'}")
        plot_curves(run_dir, csi_persistence=args.csi_persistence)

    if args.mode in ("samples", "all"):
        print(f"Plotting sample comparison → {run_dir / 'vis'}/")
        plot_samples(
            run_dir, args.epoch, args.n_samples, args.split, args.diff_vmax, device
        )

    if args.mode in ("threshold", "all"):
        print(
            f"Plotting threshold curve → {run_dir / 'vis' / 'threshold_curve.png'}"
        )
        plot_threshold(run_dir, args.epoch, args.split, device)

    print("Done.")


if __name__ == "__main__":
    main()
