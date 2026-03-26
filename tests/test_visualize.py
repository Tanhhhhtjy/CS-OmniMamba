"""Tests for visualization script."""
import json

import pytest
import torch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# ── Task 1: get_custom_rain_cmap ─────────────────────────────────────────────


def test_get_custom_rain_cmap_type_and_length():
    from scripts.visualize import get_custom_rain_cmap

    cmap = get_custom_rain_cmap()
    assert isinstance(cmap, ListedColormap)
    assert cmap.N == 7


def test_get_custom_rain_cmap_boundary_colors():
    """Index 6 (lowest rain) → white, index 0 (highest rain) → dark brown."""
    from scripts.visualize import get_custom_rain_cmap

    cmap = get_custom_rain_cmap()
    rgba_zero = cmap(0.0)  # 强降水端
    rgba_one = cmap(1.0)  # 无降水端
    # 白色 (254,254,254)/255 ≈ (0.996, 0.996, 0.996)
    assert rgba_one[0] > 0.99 and rgba_one[1] > 0.99 and rgba_one[2] > 0.99
    # 深褐 (97,40,31)/255 ≈ (0.38, 0.157, 0.122)
    assert rgba_zero[0] < 0.5 and rgba_zero[1] < 0.2


def test_get_pwv_cmap_type_and_endpoints():
    from scripts.visualize import get_pwv_cmap

    cmap = get_pwv_cmap(discrete=True)
    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.N == 11

    rgba_low = cmap(0.0)   # 低水汽
    rgba_high = cmap(1.0)  # 高水汽
    assert rgba_low[2] > rgba_low[0]  # blue-dominant
    assert rgba_high[0] > rgba_high[1] and rgba_high[0] > rgba_high[2]  # red-dominant


def test_get_radar_cmap_type_and_endpoints():
    from scripts.visualize import get_radar_cmap

    cmap = get_radar_cmap(discrete=True)
    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.N == 13

    rgba_strong = cmap(0.0)      # 强回波
    rgba_background = cmap(1.0)  # 无回波背景
    assert rgba_strong[0] > 0.9 and rgba_strong[1] < 0.4  # orange/red dominant
    assert rgba_background[2] > rgba_background[0] and rgba_background[2] > rgba_background[1]


def test_to_display_space_inverts_current_project_encoding():
    from scripts.visualize import _to_display_space

    x = torch.tensor([[0.0, 0.25, 1.0]])
    y = _to_display_space(x)
    assert torch.allclose(y, torch.tensor([[1.0, 0.75, 0.0]]))


# ── Task 2: plot_curves ──────────────────────────────────────────────────────


def test_plot_curves_creates_png(tmp_path):
    """plot_curves reads metrics.json and writes loss_curve.png."""
    from scripts.visualize import plot_curves

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    metrics = [
        {
            "epoch": 1,
            "train_loss": 0.01,
            "val_mse": 0.002,
            "val_csi_weak": 0.1,
            "val_far_weak": 0.9,
        },
        {
            "epoch": 2,
            "train_loss": 0.008,
            "val_mse": 0.0018,
            "val_csi_weak": 0.15,
            "val_far_weak": 0.85,
        },
    ]
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    plot_curves(run_dir)
    assert (run_dir / "vis" / "loss_curve.png").exists()


def test_plot_curves_with_persistence_line(tmp_path):
    """plot_curves accepts optional csi_persistence without error."""
    from scripts.visualize import plot_curves

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    metrics = [
        {
            "epoch": 1,
            "train_loss": 0.01,
            "val_mse": 0.002,
            "val_csi_weak": 0.1,
            "val_far_weak": 0.9,
        },
    ]
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    plot_curves(run_dir, csi_persistence=0.9276)
    assert (run_dir / "vis" / "loss_curve.png").exists()


# ── Task 3: plot_samples ─────────────────────────────────────────────────────


def test_plot_samples_creates_png(tmp_path):
    """plot_samples with synthetic data produces output file."""
    from scripts.visualize import plot_samples_from_tensors

    B, H, W = 2, 66, 70
    radar_frames = torch.rand(B, 1, H, W)
    pwv_frames = torch.rand(B, 1, H, W)
    pred = torch.rand(B, 1, H, W)
    target = torch.rand(B, 1, H, W)
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    plot_samples_from_tensors(
        run_dir, radar_frames, pwv_frames, pred, target, epoch=5, diff_vmax=0.5
    )
    assert (run_dir / "vis" / "vis_epoch_005.png").exists()


def test_plot_samples_matches_predict_final_style(tmp_path, monkeypatch):
    """Single-step samples should use predict_final-style 4-panel layout."""
    from scripts.visualize import plot_samples_from_tensors
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    calls = []
    original_imshow = Axes.imshow

    def spy_imshow(self, *args, **kwargs):
        calls.append(kwargs)
        return original_imshow(self, *args, **kwargs)

    def fail_colorbar(self, *args, **kwargs):
        raise AssertionError("sample plot should not add colorbars")

    monkeypatch.setattr(Axes, "imshow", spy_imshow)
    monkeypatch.setattr(Figure, "colorbar", fail_colorbar)

    B, H, W = 1, 66, 70
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    plot_samples_from_tensors(
        run_dir,
        torch.rand(B, 1, H, W),
        torch.rand(B, 1, H, W),
        torch.rand(B, 1, H, W),
        torch.rand(B, 1, H, W),
        epoch=1,
        diff_vmax=0.5,
    )

    assert len(calls) == 4, f"expected 4 image panels, got {len(calls)}"
    assert all(call["interpolation"] == "bicubic" for call in calls)
    assert calls[0]["cmap"].name.startswith("custom_pwv_")
    assert calls[1]["cmap"].name.startswith("custom_radar_")
    assert calls[2]["cmap"].name.startswith("custom_rain_")
    assert calls[3]["cmap"].name.startswith("custom_rain_")


def test_plot_samples_last_naming(tmp_path):
    """epoch=None → vis_last.png."""
    from scripts.visualize import plot_samples_from_tensors

    B, H, W = 2, 66, 70
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    plot_samples_from_tensors(
        run_dir,
        torch.rand(B, 1, H, W),
        torch.rand(B, 1, H, W),
        torch.rand(B, 1, H, W),
        torch.rand(B, 1, H, W),
        epoch=None,
        diff_vmax=0.5,
    )
    assert (run_dir / "vis" / "vis_last.png").exists()


def test_plot_samples_rejects_bad_num_steps(tmp_path):
    """num_steps not in {1,3} → ValueError."""
    from scripts.visualize import plot_samples_from_tensors

    B, H, W = 2, 66, 70
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="num_steps"):
        plot_samples_from_tensors(
            run_dir,
            torch.rand(B, 1, H, W),
            torch.rand(B, 1, H, W),
            torch.rand(B, 2, H, W),  # num_steps=2
            torch.rand(B, 2, H, W),
            epoch=1,
            diff_vmax=0.5,
        )


# ── Task 4: plot_threshold ───────────────────────────────────────────────────


def test_plot_threshold_creates_png(tmp_path):
    """plot_threshold_from_tensors produces threshold_curve.png."""
    from scripts.visualize import plot_threshold_from_tensors

    B, H, W = 4, 66, 70
    preds = [torch.rand(B, 1, H, W)]
    targets = [torch.rand(B, 1, H, W)]
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    plot_threshold_from_tensors(run_dir, preds, targets)
    assert (run_dir / "vis" / "threshold_curve.png").exists()


# ── Task 5: main() ───────────────────────────────────────────────────────────


def test_main_curves_mode(tmp_path, monkeypatch):
    """CLI --mode curves invokes plot_curves."""
    from scripts.visualize import main

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    metrics = [
        {
            "epoch": 1,
            "train_loss": 0.01,
            "val_mse": 0.002,
            "val_csi_weak": 0.1,
            "val_far_weak": 0.9,
        }
    ]
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr(
        "sys.argv", ["visualize.py", "--run-name", str(run_dir), "--mode", "curves"]
    )
    main()
    assert (run_dir / "vis" / "loss_curve.png").exists()


def test_main_samples_mode(tmp_path, monkeypatch):
    """CLI --mode samples invokes plot_samples."""
    from scripts.visualize import main

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    # Mock plot_samples to verify it's called
    called = []

    def fake_plot_samples(run_dir, epoch, n_samples, split, diff_vmax, device):
        called.append(True)
        # Create expected output
        vis_dir = run_dir / "vis"
        vis_dir.mkdir(exist_ok=True)
        (vis_dir / "vis_last.png").touch()

    monkeypatch.setattr("scripts.visualize.plot_samples", fake_plot_samples)
    monkeypatch.setattr(
        "sys.argv",
        ["visualize.py", "--run-name", str(run_dir), "--mode", "samples"],
    )
    main()
    assert len(called) == 1
    assert (run_dir / "vis" / "vis_last.png").exists()


def test_main_threshold_mode(tmp_path, monkeypatch):
    """CLI --mode threshold invokes plot_threshold."""
    from scripts.visualize import main

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    # Mock plot_threshold to verify it's called
    called = []

    def fake_plot_threshold(run_dir, epoch, split, device):
        called.append(True)
        # Create expected output
        vis_dir = run_dir / "vis"
        vis_dir.mkdir(exist_ok=True)
        (vis_dir / "threshold_curve.png").touch()

    monkeypatch.setattr("scripts.visualize.plot_threshold", fake_plot_threshold)
    monkeypatch.setattr(
        "sys.argv", ["visualize.py", "--run-name", str(run_dir), "--mode", "threshold"]
    )
    main()
    assert len(called) == 1
    assert (run_dir / "vis" / "threshold_curve.png").exists()


def test_load_model_missing_checkpoint(tmp_path):
    """_load_model raises FileNotFoundError if checkpoint missing."""
    from scripts.visualize import _load_model

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _load_model(run_dir, epoch=999, device=torch.device("cpu"))


def test_load_model_structured_checkpoint(tmp_path):
    """_load_model must handle structured checkpoints saved by current train.py."""
    from scripts.visualize import _load_model
    from src.config import T
    from src.model_convlstm import ConvLSTMModel

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    # Save a structured checkpoint (new format from train.py)
    model = ConvLSTMModel(t=T)
    optimiser = torch.optim.Adam(model.parameters())
    payload = {
        "model": model.state_dict(),
        "optimizer": optimiser.state_dict(),
        "epoch": 5,
        "best_csi": 0.42,
        "best_epoch": 3,
    }
    torch.save(payload, run_dir / "last.pt")

    # Should load without error
    loaded = _load_model(run_dir, epoch=None, device=torch.device("cpu"))
    assert loaded is not None
