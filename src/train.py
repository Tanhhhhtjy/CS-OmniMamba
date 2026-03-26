"""
Training and evaluation loop.

Usage:
    python src/train.py --epochs 10 --batch-size 8

The model is imported from src.model_convlstm (ConvLSTMModel aliased as StubModel).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    RADAR_PREP_DIR, RAIN_DIR, PWV_DIR,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    T,
)
from src.dataset import RainDataset
from src.loss import weighted_mse_loss, facl_loss
from src.metrics import MetricsAccumulator
from src.model_convlstm import ConvLSTMModel as StubModel


def _sanitize_for_json(obj):
    """Recursively replace float NaN/Inf with None for valid JSON output."""
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# ── Split filtering ───────────────────────────────────────────────────────────

def _filter_dataset_by_split(ds: RainDataset, split: str) -> RainDataset:
    """
    Return a filtered view of the dataset containing only samples whose
    TARGET frame falls within the requested date range.

    NOTE: we filter on ds.timestamps[i + ds.T]  (the target frame, t+1),
    NOT on ds.timestamps[i] (the window start, t-T+1).  Filtering by the
    window start would allow the target frame to fall into the next split,
    leaking labels across the train/val/test boundary.
    """
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val":   (VAL_START,   VAL_END),
        "test":  (TEST_START,  TEST_END),
    }
    start, end = splits[split]
    ds.indices = [
        i for i in ds.indices
        if start <= ds.timestamps[i + ds.T].date() <= end
    ]
    return ds


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, device, loss_fn=None) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        radar = batch["radar"].to(device)
        pwv   = batch["pwv"].to(device)
        rain  = batch["rain"].squeeze(1).to(device)   # [B, 1, H, W]

        optimiser.zero_grad()
        pred = model(radar, pwv)
        loss = loss_fn(pred, rain) if loss_fn is not None else weighted_mse_loss(pred, rain)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device) -> dict:
    model.eval()
    acc = MetricsAccumulator()
    for batch in tqdm(loader, desc="eval ", leave=False):
        radar = batch["radar"].to(device)
        pwv   = batch["pwv"].to(device)
        rain  = batch["rain"].squeeze(1).to(device)

        pred = model(radar, pwv)
        acc.update(pred.cpu(), rain.cpu())
    return acc.compute()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name",   type=str, default="convlstm",
                        help="Subdirectory name; outputs written to runs/{run_name}/")
    parser.add_argument("--ckpt-every", type=int, default=5,
                        help="Save a checkpoint every N epochs")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; evaluate a checkpoint on val or test split")
    parser.add_argument("--ckpt",   type=str, default=None,
                        help="Checkpoint path for --eval-only or --resume")
    parser.add_argument("--split",  type=str, default="val", choices=["val", "test"],
                        help="Split to evaluate when using --eval-only")
    parser.add_argument("--vis-every", type=int, default=None,
                        help="Save sample visualization every N epochs (default: same as --ckpt-every, 0 to disable)")
    parser.add_argument("--loss",      type=str, default="mse",
                        choices=["mse", "facl"],
                        help="Loss: 'mse'=weighted_mse_loss; "
                             "'facl'=weighted_mse + facl_loss")
    parser.add_argument("--lambda-facl", type=float, default=1.0,
                        help="Scale factor applied to facl_loss before adding to weighted_mse "
                             "(only used when --loss facl; default 1.0 = original behaviour)")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw"],
                        help="Optimizer: 'adam' (default) or 'adamw' (weight_decay=1e-4)")
    parser.add_argument("--scheduler", type=str, default="none",
                        choices=["none", "cosine"],
                        help="LR scheduler: 'none' (constant) or 'cosine' (CosineAnnealingLR)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a structured checkpoint (.pt)")
    args = parser.parse_args()

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        if args.ckpt is None:
            parser.error("--eval-only requires --ckpt")
        ckpt_path = Path(args.ckpt)
        run_dir = ckpt_path.parent
        device = torch.device(args.device)

        # Load model (support both structured and legacy bare state_dict)
        ckpt = torch.load(ckpt_path, map_location=device)
        model = StubModel(t=T).to(device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)

        # Build dataset for the requested split
        ds = _filter_dataset_by_split(
            RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR), args.split
        )
        loader = DataLoader(ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)
        print(f"Evaluating on {args.split} ({len(ds)} samples) ...")
        metrics = eval_epoch(model, loader, device)
        print("  " + "  ".join(f"{k}={v:.4f}" if v == v else f"{k}=null"
                                for k, v in metrics.items()))

        out_path = run_dir / f"eval_{args.split}.json"
        out_path.write_text(json.dumps(_sanitize_for_json(metrics), indent=2))
        print(f"Results saved → {out_path}")
        return

    run_dir = Path("runs") / args.run_name
    if args.resume:
        # Derive run_dir from checkpoint path so all outputs stay in the same run
        run_dir = Path(args.resume).parent
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = _filter_dataset_by_split(
        RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR), "train"
    )
    val_ds = _filter_dataset_by_split(
        RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR), "val"
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Save run config for reproducibility
    import datetime
    config = {
        "run_name": args.run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "ckpt_every": args.ckpt_every,
        "model": "ConvLSTMModel",
        "loss":      "facl+mse" if args.loss == "facl" else "weighted_mse_loss",
        "lambda_facl": args.lambda_facl if args.loss == "facl" else None,
        "optimizer": "AdamW"    if args.optimizer == "adamw" else "Adam",
        "scheduler": "CosineAnnealingLR" if args.scheduler == "cosine" else "none",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model = StubModel(t=T).to(device)
    if args.optimizer == "adamw":
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=args.epochs, eta_min=0.0
        )
    else:
        scheduler = None

    # ── Loop ──────────────────────────────────────────────────────────────────
    if args.loss == "facl":
        def loss_fn(pred, target):
            return weighted_mse_loss(pred, target) + args.lambda_facl * facl_loss(pred, target)
    else:
        loss_fn = None  # train_epoch falls back to weighted_mse_loss

    vis_every = args.vis_every if args.vis_every is not None else args.ckpt_every
    best_csi = -float("inf")
    best_epoch = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimiser.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_csi    = ckpt.get("best_csi", -float("inf"))
            best_epoch  = ckpt.get("best_epoch", 0)
            if scheduler is not None and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            # Note: checkpoints saved before this change have no "scheduler" key;
            # in that case the scheduler silently resets to epoch 0. This is safe.
            print(f"Resumed from {resume_path} | epoch={ckpt['epoch']}, best_csi={best_csi:.4f}")
        else:
            # Legacy bare state_dict: can only restore model weights
            model.load_state_dict(ckpt)
            print(f"Resumed model weights from {resume_path} (legacy format, optimizer state not restored)")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, device, loss_fn=loss_fn)
        val_metrics = eval_epoch(model, val_loader, device)

        m = val_metrics
        csi = m.get("csi_weak", float("nan"))
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"| val_csi_weak={csi:.4f} | val_far_weak={m.get('far_weak', float('nan')):.4f}"
            f" | val_mse={m.get('mse', float('nan')):.5f}"
        )

        # Append full metrics to JSON log (NaN → null)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_csi_weak":   m.get("csi_weak"),
            "val_pod_weak":   m.get("pod_weak"),
            "val_far_weak":   m.get("far_weak"),
            "val_csi_strong": m.get("csi_strong"),
            "val_pod_strong": m.get("pod_strong"),
            "val_far_strong": m.get("far_strong"),
            "val_mse":        m.get("mse"),
            "val_mae_rain":   m.get("mae_rain"),
        }
        metrics_path = run_dir / "metrics.json"
        history = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
        history.append(_sanitize_for_json(entry))
        metrics_path.write_text(json.dumps(history, indent=2))

        # Save best checkpoint
        if csi == csi and csi > best_csi:  # csi == csi guards against NaN
            best_csi = csi
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "optimizer": optimiser.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler is not None else None,
                 "epoch": epoch, "best_csi": best_csi, "best_epoch": best_epoch},
                run_dir / "best.pt",
            )
            print(f"  → New best: epoch {epoch}, val_csi_weak={csi:.4f}")

        # Save periodic checkpoint (structured)
        if epoch % args.ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimiser.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler is not None else None,
                 "epoch": epoch, "best_csi": best_csi, "best_epoch": best_epoch},
                run_dir / f"epoch_{epoch:03d}.pt",
            )

        if scheduler is not None:
            scheduler.step()

        # Auto-visualization (failure must not block training)
        if vis_every > 0 and epoch % vis_every == 0:
            try:
                from scripts.visualize import plot_samples_from_tensors
                model.eval()
                with torch.no_grad():
                    vis_batch = next(iter(val_loader))
                    vis_radar = vis_batch["radar"][:2].to(device)   # [2, T, 1, H, W]
                    vis_pwv   = vis_batch["pwv"][:2].to(device)     # [2, 1, 1, H, W]
                    vis_rain  = vis_batch["rain"][:2].squeeze(1).to(device)  # [2, 1, H, W]
                    vis_pred  = model(vis_radar, vis_pwv)            # [2, 1, H, W]
                # plot_samples_from_tensors expects:
                #   radar_frames: [B, 1, H, W]  (last radar frame)
                #   pwv_frames:   [B, 1, H, W]
                #   pred:         [B, 1, H, W]
                #   target:       [B, 1, H, W]
                plot_samples_from_tensors(
                    run_dir=run_dir,
                    radar_frames=vis_radar[:, -1],   # last frame: [2, 1, H, W]
                    pwv_frames=vis_pwv[:, 0],        # [2, 1, H, W]
                    pred=vis_pred.cpu(),
                    target=vis_rain.cpu(),
                    epoch=epoch,
                    diff_vmax=0.3,
                )
            except Exception as e:
                print(f"  [vis] Warning: visualization failed at epoch {epoch}: {e}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    torch.save(
        {"model": model.state_dict(), "optimizer": optimiser.state_dict(),
         "scheduler": scheduler.state_dict() if scheduler is not None else None,
         "epoch": args.epochs, "best_csi": best_csi, "best_epoch": best_epoch},
        run_dir / "last.pt",
    )
    print(f"Checkpoint saved → {run_dir / 'last.pt'} | best epoch={best_epoch}, best_csi={best_csi:.4f}")

    # ── Baseline evaluation (run once, for comparison) ────────────────────────
    # PersistenceBaseline uses RAIN(t) from batch["rain_current"].
    # ZeroBaseline uses all-zero prediction.
    from src.baselines import PersistenceBaseline, ZeroBaseline

    @torch.no_grad()
    def eval_baselines(loader, device):
        acc_p = MetricsAccumulator()
        acc_z = MetricsAccumulator()
        for batch in tqdm(loader, desc="baselines", leave=False):
            rain_target  = batch["rain"].squeeze(1).to(device)        # [B,1,H,W] t+1
            rain_current = batch["rain_current"].squeeze(1).to(device) # [B,1,H,W] t
            pred_p = PersistenceBaseline.predict(rain_current)
            pred_z = ZeroBaseline.predict(rain_target.shape)
            acc_p.update(pred_p.cpu(), rain_target.cpu())
            acc_z.update(pred_z.cpu(), rain_target.cpu())
        return acc_p.compute(), acc_z.compute()

    p_metrics, z_metrics = eval_baselines(val_loader, device)
    print(f"[Persistence] csi_weak={p_metrics.get('csi_weak', float('nan')):.4f} "
          f"| mse={p_metrics.get('mse', float('nan')):.5f}")
    print(f"[Zero      ] csi_weak={z_metrics.get('csi_weak', float('nan')):.4f} "
          f"| mse={z_metrics.get('mse', float('nan')):.5f}")

    baselines = {"val": {"persistence": p_metrics, "zero": z_metrics}}
    (run_dir / "baselines.json").write_text(
        json.dumps(_sanitize_for_json(baselines), indent=2)
    )
    print(f"Baselines saved → {run_dir / 'baselines.json'}")


if __name__ == "__main__":
    main()
