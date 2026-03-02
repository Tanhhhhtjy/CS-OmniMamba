from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch

from .config import TrainingConfig
from .losses import SpectralStructuralWeightedLoss
from .metrics import csi, ets, mae, psnr, ssim_simple
from .viz import plot_gate_history, plot_losses, show_results


def _update_ema(prev_ema: Optional[float], value: float, alpha: float) -> float:
    if prev_ema is None:
        return value
    return alpha * value + (1.0 - alpha) * prev_ema


def _is_significant_improvement(current: float, best: float, min_delta: float) -> bool:
    return current < (best - min_delta)


def train_epoch(
    model,
    dataloader,
    device,
    criterion,
    optimizer,
    scheduler=None,
    epoch: int = 0,
) -> Tuple[float, float]:
    """Run one training epoch.

    scheduler is stepped *per batch* using the fractional-epoch convention
    required by CosineAnnealingWarmRestarts so that the LR curve is smooth.
    """
    model.train()

    running_loss = 0.0
    gate_accum = 0.0
    gate_count = 0
    n_batches = max(len(dataloader), 1)

    for i, (img1, radar_seq, targets) in enumerate(dataloader):
        img1, radar_seq, targets = img1.to(device), radar_seq.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(img1, radar_seq)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Per-batch LR update for CosineAnnealingWarmRestarts
        if scheduler is not None:
            scheduler.step(epoch + i / n_batches)

        running_loss += loss.item()
        if getattr(model, "cross_attn", None) and model.cross_attn.last_gate_mean is not None:
            gate_accum += model.cross_attn.last_gate_mean
            gate_count += 1

    avg_loss = running_loss / n_batches
    avg_gate = gate_accum / gate_count if gate_count > 0 else None
    return avg_loss, avg_gate


def validate_epoch(
    model,
    dataloader,
    device,
    criterion=None,
) -> Tuple[float, Dict[str, List[float]]]:
    model.eval()
    if criterion is None:
        criterion = SpectralStructuralWeightedLoss(
            w_mae=1.0, w_fft=0.05, w_ssim=0.1, heavy_rain_boost=5.0
        ).to(device)
    total_loss = 0.0

    total_mae = [0.0, 0.0, 0.0]
    total_csi = [0.0, 0.0, 0.0]
    total_ets = [0.0, 0.0, 0.0]
    total_psnr = [0.0, 0.0, 0.0]
    total_ssim = [0.0, 0.0, 0.0]
    num_batches = 0

    with torch.no_grad():
        for img1, radar_seq, targets in dataloader:
            img1, radar_seq, targets = img1.to(device), radar_seq.to(device), targets.to(device)
            output = model(img1, radar_seq)

            loss = criterion(output, targets)
            total_loss += loss.item()

            for t in range(3):
                out_t = output[:, t]
                tgt_t = targets[:, t]

                total_mae[t] += mae(out_t, tgt_t).item()
                # threshold=0.04 ≈ 10 mm/h after [0,1] normalisation;
                # 0.1 was too high and caused trivially-perfect CSI on sparse rain fields
                total_csi[t] += csi(out_t, tgt_t, threshold=0.04).item()
                total_ets[t] += ets(out_t, tgt_t, threshold=0.04).item()
                total_psnr[t] += psnr(out_t, tgt_t).item()
                total_ssim[t] += ssim_simple(out_t, tgt_t).item()

            num_batches += 1

    denom = max(num_batches, 1)
    avg_loss = total_loss / denom
    metrics = {
        "mae":  [v / denom for v in total_mae],
        "csi":  [v / denom for v in total_csi],
        "ets":  [v / denom for v in total_ets],
        "psnr": [v / denom for v in total_psnr],
        "ssim": [v / denom for v in total_ssim],
    }
    return avg_loss, metrics


def train(
    model,
    tr_loader,
    val_loader,
    device,
    cfg: TrainingConfig,
    results_dir: str,
    test_loader=None,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    criterion = SpectralStructuralWeightedLoss(
        w_mae=1.0, w_fft=0.05, w_ssim=0.1, heavy_rain_boost=5.0
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # T_0 from config; patience must be > T_0 so early-stopping cannot fire
    # before the first cosine-restart cycle finishes.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.lr_scheduler_T0
    )
    train_losses: List[float] = []
    val_losses: List[float] = []
    gate_history: List[float] = []
    val_ema_history: List[float] = []
    val_ets_history: List[float] = []

    best_monitor = float("inf")
    val_ema: Optional[float] = None
    patience = cfg.lr_scheduler_T0 * 4  # always >= 4 × T_0
    counter = 0

    print(f"Start training for {cfg.epochs} epochs...")

    for epoch in range(cfg.epochs):
        # Scheduler is now stepped per-batch inside train_epoch
        tr_loss, avg_gate = train_epoch(
            model, tr_loader, device, criterion, optimizer, scheduler=scheduler, epoch=epoch
        )
        train_losses.append(tr_loss)
        gate_history.append(avg_gate)

        val_loss, metrics = validate_epoch(model, val_loader, device, criterion)
        val_losses.append(val_loss)

        val_ema = _update_ema(val_ema, val_loss, cfg.val_ema_alpha)
        val_ema_history.append(val_ema)
        val_ets_mean = sum(metrics["ets"]) / max(len(metrics["ets"]), 1)
        val_ets_history.append(val_ets_mean)

        monitor_value = val_ema if cfg.early_stop_use_ema else val_loss

        gate_display = avg_gate if avg_gate is not None else "N/A"
        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | TrnLoss: {tr_loss:.5f} | ValLoss: {val_loss:.5f} | ValEMA: {val_ema:.5f} | ValETS@0.04(mean): {val_ets_mean:.4f} | Gate: {gate_display}"
        )

        if _is_significant_improvement(monitor_value, best_monitor, cfg.early_stop_min_delta):
            best_monitor = monitor_value
            counter = 0
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            print(f"  --> Best model saved (monitor: {monitor_value:.5f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {counter} epochs without significant improvement.")
                break

        # Save latest checkpoint for resume
        torch.save(model.state_dict(), os.path.join(results_dir, "latest_model.pth"))

        if (epoch + 1) % 5 == 0:
            show_results(model, val_loader, device, epoch, results_dir)
            print("Validation metrics:")
            for t in range(3):
                print(
                    f"  T+{t + 1}h | MAE: {metrics['mae'][t]:.4f} | CSI: {metrics['csi'][t]:.4f} | ETS: {metrics['ets'][t]:.4f} | PSNR: {metrics['psnr'][t]:.2f} | SSIM: {metrics['ssim'][t]:.4f}"
                )

    plot_losses(train_losses, val_losses, results_dir)
    plot_gate_history(gate_history, results_dir)
    torch.save(model.state_dict(), os.path.join(results_dir, "final_model.pth"))

    # Load best checkpoint and evaluate on test set
    best_ckpt = os.path.join(results_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print("Loaded best_model.pth for final evaluation.")

    val_loss_final, val_metrics = validate_epoch(model, val_loader, device, criterion)
    report: Dict = {
        "val": {"loss": val_loss_final, "metrics": val_metrics},
        "early_stopping": {
            "monitor": "val_loss_ema" if cfg.early_stop_use_ema else "val_loss",
            "val_ema_alpha": cfg.val_ema_alpha,
            "min_delta": cfg.early_stop_min_delta,
            "patience": patience,
            "best_monitor": best_monitor,
            "last_val_ema": val_ema_history[-1] if val_ema_history else None,
            "last_val_ets_mean": val_ets_history[-1] if val_ets_history else None,
        },
    }

    if test_loader is not None:
        test_loss, test_metrics = validate_epoch(model, test_loader, device, criterion)
        report["test"] = {"loss": test_loss, "metrics": test_metrics}
        print(f"Test Loss: {test_loss:.5f}")
        for t in range(3):
            print(
                f"  Test T+{t + 1}h | MAE: {test_metrics['mae'][t]:.4f} | CSI: {test_metrics['csi'][t]:.4f} | ETS: {test_metrics['ets'][t]:.4f} | PSNR: {test_metrics['psnr'][t]:.2f} | SSIM: {test_metrics['ssim'][t]:.4f}"
            )

    with open(os.path.join(results_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
