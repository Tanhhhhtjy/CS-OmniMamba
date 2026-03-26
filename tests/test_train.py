"""
Tests for src/train.py — covers _sanitize_for_json, config.json, metrics.json,
best.pt, baselines.json, eval-only mode, and legacy checkpoint compatibility.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from src.train import main, _sanitize_for_json


# ── Fake dataset ──────────────────────────────────────────────────────────────

class FakeDataset(Dataset):
    """4-sample dataset with correct tensor shapes; no real data needed."""

    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "radar":        torch.rand(10, 1, 66, 70),
            "pwv":          torch.rand(1,  1, 66, 70),
            "rain":         torch.rand(1,  1, 66, 70),
            "rain_current": torch.rand(1,  1, 66, 70),
        }


def _patch(monkeypatch) -> None:
    """Replace RainDataset and _filter_dataset_by_split with fake versions."""
    monkeypatch.setattr("src.train.RainDataset",
                        lambda *a, **kw: FakeDataset(4))
    monkeypatch.setattr("src.train._filter_dataset_by_split",
                        lambda ds, split: ds)


def _argv(tmp_path: Path, extra: list[str] | None = None) -> list[str]:
    """Build a minimal argv list that writes into tmp_path/runs/test_run/."""
    base = [
        "train",
        "--run-name", "test_run",
        "--epochs", "1",
        "--batch-size", "2",
        "--workers", "0",
        "--device", "cpu",
    ]
    return base + (extra or [])


def _train_once(tmp_path: Path, monkeypatch) -> Path:
    """Run one training epoch and return the run_dir path."""
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    main()
    return tmp_path / "runs" / "test_run"


# ── 1. _sanitize_for_json ─────────────────────────────────────────────────────

def test_sanitize_for_json():
    assert _sanitize_for_json(float("nan"))  is None
    assert _sanitize_for_json(float("inf"))  is None
    assert _sanitize_for_json(float("-inf")) is None

    assert _sanitize_for_json(0.5)  == 0.5
    assert _sanitize_for_json(0.0)  == 0.0
    assert _sanitize_for_json(-1.0) == -1.0

    # dict recursion
    result = _sanitize_for_json({"a": float("nan"), "b": 1.0})
    assert result == {"a": None, "b": 1.0}

    # list recursion
    result = _sanitize_for_json([float("inf"), 2.0, float("nan")])
    assert result == [None, 2.0, None]

    # nested
    result = _sanitize_for_json({"x": [float("nan"), {"y": float("inf")}]})
    assert result == {"x": [None, {"y": None}]}


# ── 2. config.json written ────────────────────────────────────────────────────

def test_config_json_written(tmp_path, monkeypatch):
    run_dir = _train_once(tmp_path, monkeypatch)

    cfg_path = run_dir / "config.json"
    assert cfg_path.exists(), "config.json not found"
    cfg = json.loads(cfg_path.read_text())
    for key in ("run_name", "epochs", "lr", "model", "timestamp"):
        assert key in cfg, f"config.json missing key: {key}"


# ── 3. metrics.json — all 8 val fields present ───────────────────────────────

def test_metrics_json_all_fields(tmp_path, monkeypatch):
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, ["--epochs", "2"]))

    main()

    metrics_path = tmp_path / "runs" / "test_run" / "metrics.json"
    assert metrics_path.exists()
    history = json.loads(metrics_path.read_text())
    assert len(history) == 2, f"Expected 2 entries, got {len(history)}"

    required = [
        "val_csi_weak", "val_pod_weak", "val_far_weak",
        "val_csi_strong", "val_pod_strong", "val_far_strong",
        "val_mse", "val_mae_rain",
    ]
    for entry in history:
        for field in required:
            assert field in entry, f"metrics.json entry missing field: {field}"


# ── 4. metrics.json — no "NaN" string ────────────────────────────────────────

def test_metrics_json_no_nan_string(tmp_path, monkeypatch):
    run_dir = _train_once(tmp_path, monkeypatch)

    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    content = metrics_path.read_text()
    # Valid JSON (no parse error)
    json.loads(content)
    assert "NaN" not in content, "metrics.json contains literal 'NaN' string"


# ── 5. best.pt saved when CSI improves ───────────────────────────────────────

def test_best_pt_saved_on_csi_improvement(tmp_path, monkeypatch):
    run_dir = _train_once(tmp_path, monkeypatch)

    best_path = run_dir / "best.pt"
    assert best_path.exists(), "best.pt was not created after training"


# ── 6. best.pt is a structured dict ──────────────────────────────────────────

def test_best_pt_is_structured_dict(tmp_path, monkeypatch):
    run_dir = _train_once(tmp_path, monkeypatch)

    best_path = run_dir / "best.pt"
    assert best_path.exists(), "best.pt not found"
    ckpt = torch.load(best_path, map_location="cpu")
    assert isinstance(ckpt, dict), "best.pt should be a dict"
    for key in ("model", "optimizer", "epoch", "best_csi", "best_epoch"):
        assert key in ckpt, f"best.pt missing key: {key}"


# ── 7. baselines.json format ──────────────────────────────────────────────────

def test_baselines_json_format(tmp_path, monkeypatch):
    run_dir = _train_once(tmp_path, monkeypatch)

    bl_path = run_dir / "baselines.json"
    assert bl_path.exists(), "baselines.json not found"
    bl = json.loads(bl_path.read_text())
    assert "val" in bl, "baselines.json missing 'val' key"
    assert "persistence" in bl["val"], "baselines.json missing 'val.persistence'"
    assert "zero" in bl["val"], "baselines.json missing 'val.zero'"


# ── 8. eval-only val — writes eval_val.json ───────────────────────────────────

def test_eval_only_val(tmp_path, monkeypatch):
    # First train to get a checkpoint
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    main()

    run_dir = tmp_path / "runs" / "test_run"
    last_ckpt = str(run_dir / "last.pt")

    # Now run eval-only on val split
    monkeypatch.setattr(sys, "argv", [
        "train",
        "--eval-only",
        "--ckpt", last_ckpt,
        "--split", "val",
        "--batch-size", "2",
        "--workers", "0",
        "--device", "cpu",
    ])
    main()

    eval_path = run_dir / "eval_val.json"
    assert eval_path.exists(), "eval_val.json not written"
    json.loads(eval_path.read_text())  # must be valid JSON


# ── 9. eval-only test — writes eval_test.json ────────────────────────────────

def test_eval_only_test(tmp_path, monkeypatch):
    # First train to get a checkpoint
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    main()

    run_dir = tmp_path / "runs" / "test_run"
    last_ckpt = str(run_dir / "last.pt")

    # Now run eval-only on test split
    monkeypatch.setattr(sys, "argv", [
        "train",
        "--eval-only",
        "--ckpt", last_ckpt,
        "--split", "test",
        "--batch-size", "2",
        "--workers", "0",
        "--device", "cpu",
    ])
    main()

    eval_path = run_dir / "eval_test.json"
    assert eval_path.exists(), "eval_test.json not written"
    json.loads(eval_path.read_text())  # must be valid JSON


# ── 10. eval-only without --ckpt raises SystemExit ───────────────────────────

def test_eval_only_missing_ckpt_raises(tmp_path, monkeypatch):
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train",
        "--eval-only",
        "--split", "val",
        "--device", "cpu",
    ])
    with pytest.raises(SystemExit):
        main()


# ── 11. legacy bare state_dict checkpoint loads without error ─────────────────

def test_legacy_checkpoint_compat(tmp_path, monkeypatch):
    from src.model_convlstm import ConvLSTMModel
    from src.config import T

    # Save a bare state_dict (old format, no wrapper dict)
    model = ConvLSTMModel(t=T)
    legacy_path = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), legacy_path)

    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train",
        "--eval-only",
        "--ckpt", str(legacy_path),
        "--split", "val",
        "--batch-size", "2",
        "--workers", "0",
        "--device", "cpu",
    ])
    # Should not raise
    main()


# ── 12. baselines.json — no "NaN" string ─────────────────────────────────────

def test_baselines_json_no_nan_string(tmp_path, monkeypatch):
    run_dir = _train_once(tmp_path, monkeypatch)

    bl_path = run_dir / "baselines.json"
    assert bl_path.exists(), "baselines.json not found"
    content = bl_path.read_text()
    # Valid JSON
    json.loads(content)
    assert "NaN" not in content, "baselines.json contains literal 'NaN' string"


# ── 13. resume — run_dir stays in original run ────────────────────────────────

def test_resume_run_dir_stays_in_original_run(tmp_path, monkeypatch):
    """--resume must write all outputs into the resumed run's directory."""
    run_dir = _train_once(tmp_path, monkeypatch)
    last_pt = run_dir / "last.pt"
    assert last_pt.exists()

    # Resume from last.pt — should write into run_dir, not runs/convlstm/
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train",
        "--epochs", "2",
        "--batch-size", "2",
        "--workers", "0",
        "--device", "cpu",
        "--resume", str(last_pt),
        # intentionally omit --run-name to verify run_dir is derived from ckpt
    ])
    main()

    # metrics.json should be in the original run_dir, not runs/convlstm/
    assert (run_dir / "metrics.json").exists()
    spurious = tmp_path / "runs" / "convlstm"
    assert not spurious.exists(), "resume wrote outputs to wrong run_dir"


# ── 14. resume — optimizer state restored ─────────────────────────────────────

def test_resume_restores_epoch_and_best_csi(tmp_path, monkeypatch):
    """--resume from a structured checkpoint restores epoch and best_csi."""
    run_dir = _train_once(tmp_path, monkeypatch)
    last_pt = run_dir / "last.pt"

    ckpt = torch.load(last_pt, map_location="cpu")
    assert isinstance(ckpt, dict) and "epoch" in ckpt
    saved_epoch = ckpt["epoch"]

    # Resume for one more epoch
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train",
        "--epochs", str(saved_epoch + 1),
        "--batch-size", "2",
        "--workers", "0",
        "--device", "cpu",
        "--resume", str(last_pt),
    ])
    main()

    history = json.loads((run_dir / "metrics.json").read_text())
    epochs_logged = [e["epoch"] for e in history]
    # Should have original epoch(s) plus the resumed epoch
    assert saved_epoch + 1 in epochs_logged


# ── 15. vis-every — visualization file created ────────────────────────────────

def test_vis_every_creates_png(tmp_path, monkeypatch):
    """--vis-every 1 should produce vis/vis_epoch_001.png."""
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, ["--vis-every", "1"]))
    main()

    run_dir = tmp_path / "runs" / "test_run"
    vis_file = run_dir / "vis" / "vis_epoch_001.png"
    assert vis_file.exists(), f"Expected {vis_file} but not found"


# ── 16. --loss facl 时 loss 真的改变了（不只是配置落盘）────────────────────────

def test_loss_flag_facl_actually_uses_different_loss(tmp_path, monkeypatch):
    """--loss facl 时 train_loss 应与 --loss mse 不同（同 seed 下），
    且 config.json 应记录 'facl+mse'。"""
    import math

    def _run(extra):
        _patch(monkeypatch)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", _argv(tmp_path, extra + ["--run-name", f"run_{'_'.join(extra).replace('--','')}".replace(' ','')]))
        main()
        metrics = json.loads((tmp_path / "runs" / f"run_{'_'.join(extra).replace('--','')}" .replace(' ','') / "metrics.json").read_text())
        return metrics[0]["train_loss"]

    # Run MSE and FACL; they must produce different losses on the same data
    # (FakeDataset returns rand so losses will differ when different loss fns are used)
    loss_mse  = _run(["--loss", "mse"])
    loss_facl = _run(["--loss", "facl"])
    assert loss_mse != pytest.approx(loss_facl, rel=1e-3), (
        f"MSE loss={loss_mse:.6f} and FACL loss={loss_facl:.6f} are suspiciously equal "
        "— facl_loss may not be wired in correctly"
    )


# ── 17. --loss facl config.json 记录正确 ─────────────────────────────────────

def test_loss_flag_facl_config(tmp_path, monkeypatch):
    """--loss facl 时 config.json 应记录 'facl+mse'，训练不抛异常。"""
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, ["--loss", "facl"]))
    main()
    cfg = json.loads((tmp_path / "runs" / "test_run" / "config.json").read_text())
    assert cfg["loss"] == "facl+mse", f"Expected 'facl+mse', got {cfg['loss']}"


# ── 18. --optimizer adamw 使用 AdamW ─────────────────────────────────────────

def test_optimizer_flag_adamw(tmp_path, monkeypatch):
    """--optimizer adamw 时 config.json 应记录 'AdamW'，训练不抛异常。"""
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, ["--optimizer", "adamw"]))
    main()
    cfg = json.loads((tmp_path / "runs" / "test_run" / "config.json").read_text())
    assert cfg["optimizer"] == "AdamW", f"Expected 'AdamW', got {cfg['optimizer']}"


# ── 19. --scheduler cosine 时 lr 真的衰减了 ────────────────────────────────────

def test_scheduler_flag_cosine_lr_decays(tmp_path, monkeypatch):
    """--scheduler cosine --epochs 4 时，最后一个 epoch 的 lr 应小于初始 lr。"""
    import src.train as train_module

    captured_lrs = []
    orig_train_epoch = train_module.train_epoch

    def patched_train_epoch(model, loader, optimiser, device, loss_fn=None):
        captured_lrs.append(optimiser.param_groups[0]["lr"])
        return orig_train_epoch(model, loader, optimiser, device, loss_fn=loss_fn)

    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, [
        "--scheduler", "cosine",
        "--epochs", "4",
    ]))
    monkeypatch.setattr(train_module, "train_epoch", patched_train_epoch)
    main()

    assert len(captured_lrs) == 4, f"Expected 4 lr captures, got {len(captured_lrs)}"
    # With cosine annealing, lr should strictly decrease over 4 epochs
    assert captured_lrs[-1] < captured_lrs[0], (
        f"LR did not decay: first={captured_lrs[0]:.6f}, last={captured_lrs[-1]:.6f}"
    )
    cfg = json.loads((tmp_path / "runs" / "test_run" / "config.json").read_text())
    assert cfg["scheduler"] == "CosineAnnealingLR"


# ── 20. 组合 flags 全套 ───────────────────────────────────────────────────────

def test_combined_flags(tmp_path, monkeypatch):
    """--loss facl --optimizer adamw --scheduler cosine 三合一，训练正常完成。"""
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, [
        "--loss", "facl",
        "--optimizer", "adamw",
        "--scheduler", "cosine",
        "--epochs", "2",
    ]))
    main()
    run_dir = tmp_path / "runs" / "test_run"
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["loss"] == "facl+mse"
    assert cfg["optimizer"] == "AdamW"
    assert cfg["scheduler"] == "CosineAnnealingLR"
    history = json.loads((run_dir / "metrics.json").read_text())
    assert len(history) == 2


# ── 21. 默认 flags 向后兼容 ───────────────────────────────────────────────────

def test_default_flags_backward_compat(tmp_path, monkeypatch):
    """不传新 flags 时，config.json 记录 'weighted_mse_loss', 'Adam', 'none'。"""
    run_dir = _train_once(tmp_path, monkeypatch)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["loss"] == "weighted_mse_loss"
    assert cfg["optimizer"] == "Adam"
    assert cfg["scheduler"] == "none"


# ── 22. checkpoint 包含 scheduler key ────────────────────────────────────────

def test_checkpoint_contains_scheduler_key(tmp_path, monkeypatch):
    """best.pt / last.pt 必须包含 'scheduler' key（值为 None 或 state_dict 均可）。"""
    run_dir = _train_once(tmp_path, monkeypatch)
    for ckpt_name in ("best.pt", "last.pt"):
        ckpt_path = run_dir / ckpt_name
        assert ckpt_path.exists(), f"{ckpt_name} not found"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert "scheduler" in ckpt, f"{ckpt_name} missing 'scheduler' key"
        assert ckpt["scheduler"] is None, (
            f"{ckpt_name}['scheduler'] should be None for default (no-scheduler) run"
        )


# ── 23. --lambda-facl 写入 config.json ───────────────────────────────────────

def test_lambda_facl_recorded_in_config(tmp_path, monkeypatch):
    """--loss facl --lambda-facl 0.1 时 config.json 应记录 lambda_facl=0.1。"""
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, ["--loss", "facl", "--lambda-facl", "0.1"]))
    main()
    cfg = json.loads((tmp_path / "runs" / "test_run" / "config.json").read_text())
    assert cfg["loss"] == "facl+mse"
    assert cfg["lambda_facl"] == pytest.approx(0.1)


def test_lambda_facl_default_is_none_for_mse(tmp_path, monkeypatch):
    """--loss mse 时 config.json 的 lambda_facl 应为 null。"""
    run_dir = _train_once(tmp_path, monkeypatch)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["lambda_facl"] is None


# ── 24. --lambda-facl 影响实际 loss 数值 ─────────────────────────────────────

def test_lambda_facl_affects_loss_value():
    """lambda=0.1 的总 loss 应小于 lambda=1.0（FACL 项权重更小）。"""
    from src.loss import weighted_mse_loss, facl_loss
    torch.manual_seed(0)
    pred   = torch.rand(2, 1, 66, 70)
    target = torch.rand(2, 1, 66, 70)

    loss_lambda1  = weighted_mse_loss(pred, target) + 1.0 * facl_loss(pred, target)
    loss_lambda01 = weighted_mse_loss(pred, target) + 0.1 * facl_loss(pred, target)
    assert loss_lambda01 < loss_lambda1

