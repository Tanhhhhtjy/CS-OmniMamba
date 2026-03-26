# Visualization Script Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `scripts/visualize.py` per the spec at `docs/superpowers/specs/2026-03-15-visualization-spec.md`

**Architecture:** Single script with 4 functions (`get_custom_rain_cmap`, `plot_curves`, `plot_samples`, `plot_threshold`) + argparse entry point. Reuses existing `src/` modules (dataset, model, config, metrics) for data loading and inference. No new dependencies.

**Tech Stack:** matplotlib, numpy, torch, existing `src.*` modules

**Spec:** `docs/superpowers/specs/2026-03-15-visualization-spec.md`

---

## Context

训练完成后需要离线可视化工具来分析模型表现。当前只有 spec，`scripts/visualize.py` 还未实现。此计划按 spec 逐函数实现，配套测试，确保服务器跑完训练后能立即出图。

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `scripts/visualize.py` | 全部可视化逻辑 |
| Create | `tests/test_visualize.py` | 单元测试 |

重用的现有模块（只读，不修改）：
- `src/config.py` — 路径常量、split 日期
- `src/dataset.py` — `RainDataset`
- `src/train.py` — `_filter_dataset_by_split`
- `src/model_convlstm.py` — `ConvLSTMModel`
- `src/metrics.py` — `compute_csi_pod_far`, `THRESH_WEAK`, `THRESH_STRONG`

---

## Task 1: `get_custom_rain_cmap()` + test

**Files:**
- Create: `scripts/visualize.py`
- Create: `tests/test_visualize.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_visualize.py
from matplotlib.colors import ListedColormap

def test_get_custom_rain_cmap_type_and_length():
    from scripts.visualize import get_custom_rain_cmap
    cmap = get_custom_rain_cmap()
    assert isinstance(cmap, ListedColormap)
    assert cmap.N == 7

def test_get_custom_rain_cmap_boundary_colors():
    """Index 6 (lowest rain) → white, index 0 (highest rain) → dark brown."""
    from scripts.visualize import get_custom_rain_cmap
    cmap = get_custom_rain_cmap()
    import numpy as np
    # 归一化值 0 → 色表最左(index 0=深褐), 归一化值 1 → 色表最右(index 6=白)
    rgba_zero = cmap(0.0)   # 强降水端
    rgba_one  = cmap(1.0)   # 无降水端
    # 白色 (254,254,254)/255 ≈ (0.996, 0.996, 0.996)
    assert rgba_one[0] > 0.99 and rgba_one[1] > 0.99 and rgba_one[2] > 0.99
    # 深褐 (97,40,31)/255 ≈ (0.38, 0.157, 0.122)
    assert rgba_zero[0] < 0.5 and rgba_zero[1] < 0.2
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/test_visualize.py -v
```

- [ ] **Step 3: Implement `get_custom_rain_cmap` + script skeleton**

```python
# scripts/visualize.py
"""Offline visualization for precipitation nowcasting runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_custom_rain_cmap() -> ListedColormap:
    """7-level discrete rain colormap (strong→none = dark brown→white)."""
    colors_rgb = [
        (97,  40,  31),   # 0: deep brown — heavy rain
        (250,  1, 246),   # 1: magenta
        (0,    0, 254),   # 2: deep blue
        (101, 183, 252),  # 3: light blue
        (61,  185,  63),  # 4: deep green
        (166, 242, 142),  # 5: light green
        (254, 254, 254),  # 6: white — no rain
    ]
    colors_norm = [(r / 255, g / 255, b / 255) for r, g, b in colors_rgb]
    return ListedColormap(colors_norm, name="custom_rain_discrete")
```

- [ ] **Step 4: Run test — expect PASS**
- [ ] **Step 5: Commit**

---

## Task 2: `plot_curves()` + test

**Files:**
- Modify: `scripts/visualize.py`
- Modify: `tests/test_visualize.py`

**Data source:** `runs/{run_name}/metrics.json` — JSON array, each entry has keys: `epoch`, `train_loss`, `val_mse`, `val_csi_weak`, `val_far_weak`

- [ ] **Step 1: Write failing test**

```python
# tests/test_visualize.py (append)
import json, tempfile, os

def test_plot_curves_creates_png(tmp_path):
    """plot_curves reads metrics.json and writes loss_curve.png."""
    from scripts.visualize import plot_curves
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    metrics = [
        {"epoch": 1, "train_loss": 0.01, "val_mse": 0.002, "val_csi_weak": 0.1, "val_far_weak": 0.9},
        {"epoch": 2, "train_loss": 0.008, "val_mse": 0.0018, "val_csi_weak": 0.15, "val_far_weak": 0.85},
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
        {"epoch": 1, "train_loss": 0.01, "val_mse": 0.002, "val_csi_weak": 0.1, "val_far_weak": 0.9},
    ]
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    plot_curves(run_dir, csi_persistence=0.9276)
    assert (run_dir / "vis" / "loss_curve.png").exists()
```

- [ ] **Step 2: Run test — expect FAIL**
- [ ] **Step 3: Implement `plot_curves`**

```python
def plot_curves(run_dir: Path, csi_persistence: float | None = None) -> None:
    """Read metrics.json and plot 3-row training curves."""
    metrics_path = run_dir / "metrics.json"
    data = json.loads(metrics_path.read_text())

    epochs     = [d["epoch"] for d in data]
    train_loss = [d["train_loss"] for d in data]
    val_mse    = [d["val_mse"] for d in data]
    val_csi    = [d["val_csi_weak"] for d in data]
    val_far    = [d["val_far_weak"] for d in data]

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
        axes[2].axhline(csi_persistence, ls="--", color="gray",
                        label=f"persistence CSI={csi_persistence:.4f}")
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
```

- [ ] **Step 4: Run test — expect PASS**
- [ ] **Step 5: Commit**

---

## Task 3: `plot_samples()` + test

**Files:**
- Modify: `scripts/visualize.py`
- Modify: `tests/test_visualize.py`

**Key dependencies:**
- `src/train.py:_filter_dataset_by_split` — split filtering
- `src/dataset.py:RainDataset` — `__getitem__` returns `{"radar": [T,1,H,W], "pwv": [1,1,H,W], "rain": [1,1,H,W], ...}`
- `src/model_convlstm.py:ConvLSTMModel` — `forward(radar, pwv)` → `[B,1,H,W]`（当前单步）
- Model output convention: `pred.shape = [B, num_steps, H, W]`；当前 `num_steps=1`

- [ ] **Step 1: Write failing test（用合成数据，不依赖真实 checkpoint）**

```python
def test_plot_samples_creates_png(tmp_path):
    """plot_samples with synthetic data produces output file."""
    from scripts.visualize import plot_samples_from_tensors
    import torch
    B, H, W = 2, 66, 70
    radar_frames = torch.rand(B, 1, H, W)  # 代表性输入帧 t
    pwv_frames   = torch.rand(B, 1, H, W)
    pred         = torch.rand(B, 1, H, W)  # num_steps=1
    target       = torch.rand(B, 1, H, W)
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    plot_samples_from_tensors(run_dir, radar_frames, pwv_frames, pred, target,
                              epoch=5, diff_vmax=0.5)
    assert (run_dir / "vis" / "vis_epoch_005.png").exists()

def test_plot_samples_last_naming(tmp_path):
    """epoch=None → vis_last.png."""
    from scripts.visualize import plot_samples_from_tensors
    import torch
    B, H, W = 2, 66, 70
    run_dir = tmp_path / "test_run"; run_dir.mkdir()
    plot_samples_from_tensors(
        run_dir, torch.rand(B,1,H,W), torch.rand(B,1,H,W),
        torch.rand(B,1,H,W), torch.rand(B,1,H,W),
        epoch=None, diff_vmax=0.5)
    assert (run_dir / "vis" / "vis_last.png").exists()

def test_plot_samples_rejects_bad_num_steps(tmp_path):
    """num_steps not in {1,3} → ValueError."""
    from scripts.visualize import plot_samples_from_tensors
    import torch, pytest
    B, H, W = 2, 66, 70
    run_dir = tmp_path / "test_run"; run_dir.mkdir()
    with pytest.raises(ValueError, match="num_steps"):
        plot_samples_from_tensors(
            run_dir, torch.rand(B,1,H,W), torch.rand(B,1,H,W),
            torch.rand(B,2,H,W), torch.rand(B,2,H,W),  # num_steps=2
            epoch=1, diff_vmax=0.5)
```

- [ ] **Step 2: Run test — expect FAIL**
- [ ] **Step 3: Implement**

设计说明：拆分为两层函数：
- `plot_samples_from_tensors(run_dir, radar, pwv, pred, target, epoch, diff_vmax)` — 纯绘图，接受 tensor，可独立测试
- `plot_samples(run_dir, epoch, n_samples, split, diff_vmax, device)` — 加载 checkpoint + DataLoader，调用上面的函数

```python
def plot_samples_from_tensors(
    run_dir: Path,
    radar_frames: torch.Tensor,   # [B, 1, H, W] — 代表性输入帧
    pwv_frames: torch.Tensor,     # [B, 1, H, W]
    pred: torch.Tensor,           # [B, num_steps, H, W]
    target: torch.Tensor,         # [B, num_steps, H, W]
    epoch: int | None,
    diff_vmax: float,
) -> None:
    num_steps = pred.shape[1]
    if num_steps not in (1, 3):
        raise ValueError(f"num_steps must be 1 or 3, got {num_steps}")

    N = pred.shape[0]
    n_cols = 2 + 3 * num_steps  # 5 for single-step, 11 for 3-step
    fig_w = 15 if num_steps == 1 else 28
    fig, axes = plt.subplots(N, n_cols, figsize=(fig_w, 4 * N),
                             squeeze=False)

    rain_cmap = get_custom_rain_cmap()
    col_titles = ["RADAR (t)", "PWV (t)"]
    for s in range(num_steps):
        label = f"t+{s+1}"
        col_titles += [f"Pred {label}", f"True {label}", f"Diff {label}"]

    for row in range(N):
        col = 0
        # RADAR
        axes[row, col].imshow(radar_frames[row, 0].cpu().numpy(),
                              cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        col += 1
        # PWV
        axes[row, col].imshow(pwv_frames[row, 0].cpu().numpy(),
                              cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        col += 1
        # Per step
        for s in range(num_steps):
            p = pred[row, s].cpu().numpy()
            t = target[row, s].cpu().numpy()
            d = p - t
            im_p = axes[row, col].imshow(p, cmap=rain_cmap, vmin=0, vmax=1, interpolation="nearest")
            col += 1
            im_t = axes[row, col].imshow(t, cmap=rain_cmap, vmin=0, vmax=1, interpolation="nearest")
            col += 1
            im_d = axes[row, col].imshow(d, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax, interpolation="nearest")
            col += 1

    # Titles on first row only
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9)

    # Remove ticks
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    vis_dir = run_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    fname = f"vis_epoch_{epoch:03d}.png" if epoch is not None else "vis_last.png"
    fig.savefig(vis_dir / fname, dpi=150)
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
    from src.config import RADAR_PREP_DIR, RAIN_DIR, PWV_DIR, T
    from src.dataset import RainDataset
    from src.model_convlstm import ConvLSTMModel as StubModel
    from src.train import _filter_dataset_by_split
    from torch.utils.data import DataLoader

    # Load checkpoint
    if epoch is not None:
        ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
    else:
        ckpt_path = run_dir / "last.pt"
    model = StubModel(t=T).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Build DataLoader
    ds = _filter_dataset_by_split(
        RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR), split
    )
    loader = DataLoader(ds, batch_size=n_samples, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    radar = batch["radar"].to(device)   # [B, T, 1, H, W]
    pwv   = batch["pwv"].to(device)     # [B, 1, 1, H, W]
    rain  = batch["rain"].to(device)    # [B, 1, 1, H, W]

    with torch.no_grad():
        pred = model(radar, pwv)        # [B, 1, H, W]

    # 代表性输入帧: radar[:, -1, :, :, :] → [B, 1, H, W]
    radar_last = radar[:, -1]           # [B, 1, H, W]
    pwv_squeezed = pwv[:, 0]            # [B, 1, H, W]
    target = rain[:, 0]                 # [B, 1, H, W]
    # pred already [B, 1, H, W] — matches spec [B, num_steps, H, W]

    plot_samples_from_tensors(run_dir, radar_last, pwv_squeezed,
                              pred, target, epoch, diff_vmax)
```

- [ ] **Step 4: Run test — expect PASS**
- [ ] **Step 5: Commit**

---

## Task 4: `plot_threshold()` + test

**Files:**
- Modify: `scripts/visualize.py`
- Modify: `tests/test_visualize.py`

**Key dependency:** `src/metrics.py:compute_csi_pod_far(pred, target, threshold)` — 返回 `{"csi", "pod", "far"}`

- [ ] **Step 1: Write failing test（用合成数据）**

```python
def test_plot_threshold_creates_png(tmp_path):
    """plot_threshold_from_tensors produces threshold_curve.png."""
    from scripts.visualize import plot_threshold_from_tensors
    import torch
    B, H, W = 4, 66, 70
    preds   = [torch.rand(B, 1, H, W)]
    targets = [torch.rand(B, 1, H, W)]
    run_dir = tmp_path / "test_run"; run_dir.mkdir()
    plot_threshold_from_tensors(run_dir, preds, targets)
    assert (run_dir / "vis" / "threshold_curve.png").exists()
```

- [ ] **Step 2: Run test — expect FAIL**
- [ ] **Step 3: Implement**

同样拆两层：
- `plot_threshold_from_tensors(run_dir, all_preds, all_targets)` — 纯绘图
- `plot_threshold(run_dir, epoch, split, device)` — 加载模型 + 推理全集

```python
def plot_threshold_from_tensors(
    run_dir: Path,
    all_preds: list[torch.Tensor],    # list of [B, 1, H, W]
    all_targets: list[torch.Tensor],  # list of [B, 1, H, W]
) -> None:
    """Sweep thresholds and plot CSI/POD/FAR curves."""
    from src.metrics import compute_csi_pod_far, THRESH_WEAK, THRESH_STRONG

    pred_cat   = torch.cat(all_preds, dim=0)    # [N, 1, H, W]
    target_cat = torch.cat(all_targets, dim=0)

    thresholds = np.arange(0, 1.01, 0.02)
    csi_vals, pod_vals, far_vals = [], [], []
    for th in thresholds:
        m = compute_csi_pod_far(pred_cat, target_cat, th)
        csi_vals.append(m["csi"])
        pod_vals.append(m["pod"])
        far_vals.append(m["far"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, csi_vals, label="CSI")
    ax.plot(thresholds, pod_vals, label="POD")
    ax.plot(thresholds, far_vals, label="FAR")
    ax.axvline(THRESH_WEAK, ls="--", color="gray", alpha=0.7,
               label=f"THRESH_WEAK={THRESH_WEAK:.4f}")
    ax.axvline(THRESH_STRONG, ls="--", color="dimgray", alpha=0.7,
               label=f"THRESH_STRONG={THRESH_STRONG:.4f}")
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
    from src.config import RADAR_PREP_DIR, RAIN_DIR, PWV_DIR, T
    from src.dataset import RainDataset
    from src.model_convlstm import ConvLSTMModel as StubModel
    from src.train import _filter_dataset_by_split
    from torch.utils.data import DataLoader

    if epoch is not None:
        ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
    else:
        ckpt_path = run_dir / "last.pt"
    model = StubModel(t=T).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    ds = _filter_dataset_by_split(
        RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR), split
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["radar"].to(device), batch["pwv"].to(device))
            all_preds.append(pred.cpu())
            all_targets.append(batch["rain"].squeeze(1).cpu())

    plot_threshold_from_tensors(run_dir, all_preds, all_targets)
```

- [ ] **Step 4: Run test — expect PASS**
- [ ] **Step 5: Commit**

---

## Task 5: `main()` argparse entry point + integration test

**Files:**
- Modify: `scripts/visualize.py`
- Modify: `tests/test_visualize.py`

- [ ] **Step 1: Write failing test**

```python
def test_main_curves_mode(tmp_path, monkeypatch):
    """CLI --mode curves invokes plot_curves."""
    from scripts.visualize import main
    import json
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    metrics = [{"epoch": 1, "train_loss": 0.01, "val_mse": 0.002,
                "val_csi_weak": 0.1, "val_far_weak": 0.9}]
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr("sys.argv", [
        "visualize.py", "--run-name", str(run_dir), "--mode", "curves"
    ])
    main()
    assert (run_dir / "vis" / "loss_curve.png").exists()
```

- [ ] **Step 2: Run test — expect FAIL**
- [ ] **Step 3: Implement `main()`**

```python
def main():
    parser = argparse.ArgumentParser(description="Visualization for precipitation nowcasting")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Run directory (absolute path or relative under runs/)")
    parser.add_argument("--mode", choices=["curves", "samples", "threshold", "all"],
                        default="all")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--csi-persistence", type=float, default=None)
    parser.add_argument("--diff-vmax", type=float, default=0.5)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Resolve run directory
    run_dir = Path(args.run_name)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = Path("runs") / args.run_name
    device = torch.device(args.device)

    if args.mode in ("curves", "all"):
        print(f"Plotting training curves → {run_dir / 'vis' / 'loss_curve.png'}")
        plot_curves(run_dir, csi_persistence=args.csi_persistence)

    if args.mode in ("samples", "all"):
        print(f"Plotting sample comparison → {run_dir / 'vis'}/")
        plot_samples(run_dir, args.epoch, args.n_samples,
                     args.split, args.diff_vmax, device)

    if args.mode in ("threshold", "all"):
        print(f"Plotting threshold curve → {run_dir / 'vis' / 'threshold_curve.png'}")
        plot_threshold(run_dir, args.epoch, args.split, device)

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test — expect PASS**
- [ ] **Step 5: Run full test suite**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/ -v
```

Expected: all existing 35 tests + new visualization tests PASS.

- [ ] **Step 6: Commit**

---

## Verification

1. **单元测试**：`python -m pytest tests/test_visualize.py -v` — 全部 PASS
2. **回归测试**：`python -m pytest tests/ -v` — 35+ tests 全部 PASS
3. **Smoke test（本地，用 convlstm_smoke run）**：
   ```bash
   python scripts/visualize.py --run-name convlstm_smoke --mode curves
   ```
   验证 `runs/convlstm_smoke/vis/loss_curve.png` 生成且可打开
4. **samples/threshold 模式**需要 GPU + 完整数据，在远程服务器验证
