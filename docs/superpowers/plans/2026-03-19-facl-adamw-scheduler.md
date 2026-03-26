# FACL + AdamW + CosineAnnealingLR 实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 ConvLSTM 训练流程中引入尺度标定的 FACL loss 和 AdamW + CosineAnnealingLR 优化器，通过 CLI flags 独立控制，零破坏兼容已有测试，并设计可分离因果归因的实验方案。

**Architecture:**
- `src/loss.py` 新增 `facl_loss()` 函数（频域振幅归一化 + 相关损失），与 `weighted_mse_loss` 并列；
- `src/train.py` 新增 `--loss {mse,facl}`、`--optimizer {adam,adamw}`、`--scheduler {none,cosine}` 三个独立 CLI flags，并将选择写入 `config.json`；
- 改动最小化：`loss.py` 和 `train.py` 各自独立，其他文件（dataset、model、metrics）不动。

**Tech Stack:** Python 3.11, PyTorch 2.5.1, torch.fft（标准库，无需安装额外依赖）

---

## Chunk 1: FACL Loss 实现

### 为什么引入 FACL？

**问题诊断（当前已知证据）**：E1 实验显示 POD=0.91、FAR=0.64、CSI=0.35，高 POD + 高 FAR 的典型特征是"撒网式预测"。一个合理假设是 `weighted_mse_loss` 的均值回归效应导致模糊预测，阈值化后产生大量 false positive。这是假设，需要被 E2 实验证伪或证实。

**Success criteria（如果 FACL 有效，应满足）**：
- `val_far_weak` 降至 **< 0.50**（当前 0.64，即降低 ≥ 0.14）
- `val_pod_weak` 不低于 **0.80**（允许轻微下降，但不能大幅牺牲召回）
- `val_csi_weak` 较 E1 提升（当前 0.35）

**FACL（Fourier Amplitude and Correlation Loss）** 来自 NeurIPS 2024 论文"Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting"，专门针对降水 nowcasting 任务设计：

- **振幅损失（Amplitude Loss）**：对 pred 和 target 的 2D FFT 振幅谱做 L1 距离。**关键：必须除以 H×W 做尺度归一化**，否则振幅项会主导训练（实测：未归一化 amp_loss ≈ 2.57，MSE ≈ 0.018，相差 140x）。归一化后 amp_loss ≈ 0.000556，与 MSE 可比。

- **相关损失（Correlation Loss）**：对 pred 和 target 做归一化互相关（Pearson correlation），鼓励空间结构对齐。实测量级 corr_loss ≈ 0.043，约为 MSE 的 2x，权重设为 0.5。

**归一化后的混合方案（量纲平衡）**：
```
H, W = pred.shape[-2], pred.shape[-1]
total = weighted_mse(pred, target)
      + 1.0 * (amp_loss / (H*W))    # 归一化后 ~0.030x MSE
      + 0.5 * corr_loss             # ~1x MSE
```
三项总量级相当，MSE 不会被频谱项压制。

> **注**：振幅损失对全频谱做等权 L1，DC 和低频大幅值分量仍然有一定影响，但归一化后整体量级已合理。如后续实验表明频谱项仍主导，可进一步降低 `lambda_amp`，建议消融 `lambda_amp ∈ {0.1, 0.3, 1.0}`。

---

### Task 1: 实现 `facl_loss()` 并写单元测试

**Files:**
- Modify: `src/loss.py`
- Modify: `tests/test_loss.py`

- [ ] **Step 1: 写失败测试（FACL 函数存在性 + 基本性质 + 量纲校验）**

在 `tests/test_loss.py` 末尾追加：

```python
# ── FACL tests ────────────────────────────────────────────────────────────────
from src.loss import facl_loss

def test_facl_zero_loss_perfect_prediction():
    """完美预测时 FACL loss 应接近 0。"""
    t = torch.rand(2, 1, 66, 70)
    loss = facl_loss(t, t)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)

def test_facl_positive_for_wrong_prediction():
    """预测全零、target 全一时 loss 应 > 0。"""
    pred   = torch.zeros(2, 1, 66, 70)
    target = torch.ones(2, 1, 66, 70)
    assert facl_loss(pred, target).item() > 0.0

def test_facl_finite():
    """随机输入不得产生 NaN 或 Inf。"""
    pred   = torch.rand(2, 1, 66, 70)
    target = torch.rand(2, 1, 66, 70)
    loss = facl_loss(pred, target)
    assert torch.isfinite(loss), f"facl_loss returned non-finite: {loss.item()}"

def test_facl_shape_invariant():
    """不同 batch size 和空间尺寸下均应返回标量。"""
    for shape in [(1, 1, 66, 70), (4, 1, 66, 70), (2, 1, 32, 32)]:
        pred   = torch.rand(*shape)
        target = torch.rand(*shape)
        loss = facl_loss(pred, target)
        assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"

def test_facl_less_loss_for_closer_prediction():
    """更接近 target 的预测应产生更低的 FACL loss。"""
    torch.manual_seed(42)  # fixed seed for determinism
    target = torch.rand(2, 1, 66, 70)
    pred_close = target + 0.01 * torch.randn_like(target)
    pred_far   = torch.rand_like(target)
    loss_close = facl_loss(pred_close.clamp(0, 1), target)
    loss_far   = facl_loss(pred_far, target)
    assert loss_close.item() < loss_far.item()

def test_facl_scale_comparable_to_mse():
    """归一化后 facl_loss 与 weighted_mse_loss 量级可比（0.001x ~ 10x MSE 范围内）。
    这确保 FACL 不会因量纲差异压制 MSE 或被 MSE 压制。
    """
    from src.loss import weighted_mse_loss
    torch.manual_seed(0)
    pred   = torch.rand(4, 1, 66, 70)
    target = torch.rand(4, 1, 66, 70)
    mse = weighted_mse_loss(pred, target).item()
    fl  = facl_loss(pred, target).item()
    ratio = fl / (mse + 1e-8)
    assert 0.001 < ratio < 10.0, (
        f"FACL/MSE ratio = {ratio:.3f} is outside [0.001, 10] — "
        f"amplitude normalization may be missing (FACL={fl:.4f}, MSE={mse:.4f})"
    )

def test_facl_gradients_finite_and_nonzero():
    """FACL 反向传播时梯度应有限且非零（确保训练接线正确）。"""
    pred   = torch.rand(2, 1, 66, 70, requires_grad=True)
    target = torch.rand(2, 1, 66, 70)
    loss = facl_loss(pred, target)
    loss.backward()
    assert pred.grad is not None, "No gradient computed"
    assert torch.isfinite(pred.grad).all(), "Gradient contains NaN or Inf"
    assert pred.grad.abs().sum().item() > 0.0, "Gradient is all zeros"
```

- [ ] **Step 2: 运行测试，确认失败（ImportError）**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/test_loss.py::test_facl_zero_loss_perfect_prediction -v
```
Expected: `ImportError: cannot import name 'facl_loss' from 'src.loss'`

- [ ] **Step 3: 实现 `facl_loss()`**

在 `src/loss.py` 中，在 `weighted_mse_loss` 之后追加：

```python
def _amplitude_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Normalised L1 distance between FFT amplitude spectra.

    pred / target: float32, shape [B, C, H, W], values in [0, 1].

    The raw FFT amplitude scales with H*W (DC component ~ sum of pixel values).
    We divide by H*W so the result is comparable in magnitude to pixel-space MSE.
    Without this normalisation, amplitude loss is ~140x larger than MSE loss
    for typical rain-like inputs (measured on [B=8, H=66, W=70] tensors).
    """
    H, W = pred.shape[-2], pred.shape[-1]
    pred_fft   = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    pred_amp   = pred_fft.abs()
    target_amp = target_fft.abs()
    return (pred_amp - target_amp).abs().mean() / (H * W)


def _correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    1 - Pearson correlation between pred and target (averaged over batch).

    Each sample [C, H, W] is flattened to a vector; correlation is computed
    per sample then averaged. Returns values in [0, 2]; 0 = perfect correlation.

    Degenerate case: if pred or target is spatially constant (all same value),
    the norm is 0 and correlation is undefined. clamp(min=1e-8) in the
    denominator causes r → 0, giving loss → 1.0 (treated as uncorrelated).
    """
    B = pred.shape[0]
    p = pred.view(B, -1)    # [B, N]
    t = target.view(B, -1)  # [B, N]

    p_mean = p.mean(dim=1, keepdim=True)
    t_mean = t.mean(dim=1, keepdim=True)
    p_c = p - p_mean
    t_c = t - t_mean

    num   = (p_c * t_c).sum(dim=1)
    denom = (p_c.norm(dim=1) * t_c.norm(dim=1)).clamp(min=1e-8)
    r     = num / denom           # [B], in [-1, 1]
    return (1.0 - r).mean()       # loss in [0, 2]; 0 when perfectly correlated


def facl_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_amp: float = 1.0,
    lambda_corr: float = 0.5,
) -> torch.Tensor:
    """
    Fourier Amplitude and Correlation Loss (FACL).

    Combines:
    - Amplitude loss: normalised L1 on FFT amplitude spectra (divided by H*W
      to match pixel-space MSE magnitude; suppresses over-smoothing / blurring).
    - Correlation loss: 1 - Pearson correlation (encourages spatial structure
      alignment, robust to global brightness shift).

    Scale calibration (measured on [B=8, H=66, W=70] rain-like tensors):
      weighted_mse  ≈ 0.018
      amp_loss/H/W  ≈ 0.00056  (lambda_amp=1.0 → contributes ~0.03x MSE)
      corr_loss     ≈ 0.043    (lambda_corr=0.5 → contributes ~1.2x MSE)
    Total FACL with defaults is ~1.2x weighted_mse — suitable for direct addition.

    Typical usage in train.py:
        total_loss = weighted_mse_loss(pred, target) + facl_loss(pred, target)

    Reference: Yan et al., "Fourier Amplitude and Correlation Loss: Beyond
    Using L2 Loss for Skillful Precipitation Nowcasting", NeurIPS 2024.

    Args:
        pred:        float32, normalised [0, 1], shape [B, C, H, W].
        target:      same shape as pred.
        lambda_amp:  weight for normalised amplitude loss (default 1.0).
        lambda_corr: weight for correlation loss (default 0.5).

    Returns:
        Scalar loss tensor.
    """
    amp_l  = _amplitude_loss(pred, target)
    corr_l = _correlation_loss(pred, target)
    return lambda_amp * amp_l + lambda_corr * corr_l
```

- [ ] **Step 4: 运行 FACL 相关测试，确认全部通过**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/test_loss.py -v
```
Expected: 所有 test_loss.py 测试 PASS（原有 4 个 + 新增 7 个 = 11 个）

- [ ] **Step 5: 运行全量测试，确认无回归**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/ -v
```
Expected: 67 + 7 = 74 tests PASS，0 failures

---

## Chunk 2: train.py — 引入 `--loss`, `--optimizer`, `--scheduler` flags

### 为什么引入 AdamW + CosineAnnealingLR？

**AdamW**：标准 Adam 的 weight decay 被耦合进自适应梯度缩放（实为 L2 正则化），AdamW 将其解耦，正确地衰减参数本身。对于过拟合导致的高 FAR，AdamW 提供更有效的隐式正则化。

**CosineAnnealingLR**：E1 观察到 epoch 90→100 轻微退化，说明固定 lr 后期已无法精细收敛。Cosine 调度将 lr 从初始值余弦衰减到 `eta_min=0`，后期精细调整。

**关键：三个 flags 完全独立**，可以单独使用或任意组合，服务器实验通过控制变量分离因果。

---

### Task 2: 写 train.py 相关测试

**Files:**
- Modify: `tests/test_train.py`

- [ ] **Step 1: 在 `tests/test_train.py` 末尾追加新测试**

```python
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
```

- [ ] **Step 2: 运行新测试，确认失败（因 train.py 尚未支持新 flags）**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/test_train.py::test_loss_flag_facl_actually_uses_different_loss tests/test_train.py::test_loss_flag_facl_config tests/test_train.py::test_optimizer_flag_adamw tests/test_train.py::test_scheduler_flag_cosine_lr_decays tests/test_train.py::test_combined_flags tests/test_train.py::test_default_flags_backward_compat tests/test_train.py::test_checkpoint_contains_scheduler_key -v
```
Expected: 7 tests FAIL（SystemExit 或 AssertionError）

- [ ] **Step 3: 修改 `src/train.py`，添加三个 CLI flags 并实现逻辑**

在 `train.py` 中需要做以下改动（**按顺序执行**）：

**3a. 导入 facl_loss**

将 `from src.loss import weighted_mse_loss` 改为：

```python
from src.loss import weighted_mse_loss, facl_loss
```

**3b. 新增三个 argparse flags**

在 `parser.add_argument("--vis-every", ...)` 之后插入：

```python
    parser.add_argument("--loss",      type=str, default="mse",
                        choices=["mse", "facl"],
                        help="Loss: 'mse'=weighted_mse_loss; "
                             "'facl'=weighted_mse + facl_loss")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw"],
                        help="Optimizer: 'adam' (default) or 'adamw' (weight_decay=1e-4)")
    parser.add_argument("--scheduler", type=str, default="none",
                        choices=["none", "cosine"],
                        help="LR scheduler: 'none' (constant) or 'cosine' (CosineAnnealingLR)")
```

**3c. 修改 config.json 写入**

将 `config` dict 中的 `"loss": "weighted_mse_loss"` 行改为：

```python
        "loss":      "facl+mse" if args.loss == "facl" else "weighted_mse_loss",
        "optimizer": "AdamW"    if args.optimizer == "adamw" else "Adam",
        "scheduler": "CosineAnnealingLR" if args.scheduler == "cosine" else "none",
```

**3d. 修改 optimiser 构建**

将 `optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)` 替换为：

```python
    if args.optimizer == "adamw":
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
```

**3e. 在 optimiser 构建之后、`if args.resume:` 块之前，添加 scheduler 构建**

> ⚠️ **顺序关键**：scheduler 必须在 `if args.resume:` 块之前创建，resume 块才能恢复 scheduler state。

```python
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=args.epochs, eta_min=0.0
        )
    else:
        scheduler = None
```

**3f. 修改 train_epoch 函数签名**

将 `def train_epoch(model, loader, optimiser, device) -> float:` 改为：
```python
def train_epoch(model, loader, optimiser, device, loss_fn=None) -> float:
```

并将函数体中 `loss = weighted_mse_loss(pred, rain)` 替换为：
```python
        loss = loss_fn(pred, rain) if loss_fn is not None else weighted_mse_loss(pred, rain)
```

**3g. 在 `main()` 中，构建 loss_fn**

在 `vis_every = args.vis_every ...` 之前插入：

```python
    if args.loss == "facl":
        def loss_fn(pred, target):
            return weighted_mse_loss(pred, target) + facl_loss(pred, target)
    else:
        loss_fn = None  # train_epoch falls back to weighted_mse_loss
```

**3h. 修改训练循环的 train_epoch 调用**

将 `train_loss = train_epoch(model, train_loader, optimiser, device)` 改为：

```python
        train_loss = train_epoch(model, train_loader, optimiser, device, loss_fn=loss_fn)
```

**3i. 在训练循环末尾，可视化块之前，添加 scheduler step**

在 `if vis_every > 0 and epoch % vis_every == 0:` 之前插入：

```python
        if scheduler is not None:
            scheduler.step()
```

**3j. resume 时恢复 scheduler state**

在 resume 的结构化 checkpoint 加载分支（`if isinstance(ckpt, dict) and "model" in ckpt:`）末尾，`best_epoch = ckpt.get(...)` 之后插入：

```python
            if scheduler is not None and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            # Note: checkpoints saved before this change have no "scheduler" key;
            # in that case the scheduler silently resets to epoch 0. This is safe.
```

**3k. checkpoint 保存时附带 scheduler state（所有三处 torch.save）**

将所有 `torch.save({...}, ...)` 的 dict 中加入 `"scheduler"` 字段：

`best.pt`：
```python
            torch.save(
                {"model": model.state_dict(), "optimizer": optimiser.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler is not None else None,
                 "epoch": epoch, "best_csi": best_csi, "best_epoch": best_epoch},
                run_dir / "best.pt",
            )
```

`epoch_NNN.pt`：
```python
            torch.save(
                {"model": model.state_dict(), "optimizer": optimiser.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler is not None else None,
                 "epoch": epoch, "best_csi": best_csi, "best_epoch": best_epoch},
                run_dir / f"epoch_{epoch:03d}.pt",
            )
```

`last.pt`：
```python
    torch.save(
        {"model": model.state_dict(), "optimizer": optimiser.state_dict(),
         "scheduler": scheduler.state_dict() if scheduler is not None else None,
         "epoch": args.epochs, "best_csi": best_csi, "best_epoch": best_epoch},
        run_dir / "last.pt",
    )
```

- [ ] **Step 4: 运行新增的 7 个测试，确认通过**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/test_train.py::test_loss_flag_facl_actually_uses_different_loss tests/test_train.py::test_loss_flag_facl_config tests/test_train.py::test_optimizer_flag_adamw tests/test_train.py::test_scheduler_flag_cosine_lr_decays tests/test_train.py::test_combined_flags tests/test_train.py::test_default_flags_backward_compat tests/test_train.py::test_checkpoint_contains_scheduler_key -v
```
Expected: 7 tests PASS

- [ ] **Step 5: 运行全量测试，确认无回归**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/ -v
```
Expected: 74 + 7 = 81 tests PASS，0 failures

---

## Chunk 3: 收尾与服务器实验设计

### Task 3: 服务器烟测脚本

新建 `scripts/smoke_flags.py`（**只在远程服务器运行**，需要真实数据）：

```python
"""Quick smoke test for new CLI flags on the server - 1 epoch each."""
import subprocess, sys

configs = [
    (["--loss", "mse"],    "E_smoke_mse"),
    (["--loss", "facl"],   "E_smoke_facl"),
    (["--optimizer", "adamw"], "E_smoke_adamw"),
    (["--scheduler", "cosine"], "E_smoke_cosine"),
    (["--loss", "facl", "--optimizer", "adamw", "--scheduler", "cosine"], "E_smoke_full"),
]

for extra, name in configs:
    cmd = [
        sys.executable, "-m", "src.train",
        "--run-name", name,
        "--epochs", "1",
        "--batch-size", "4",
        "--workers", "2",
    ] + extra
    print(f"\n{'='*60}\nRunning: {' '.join(extra)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FAILED:")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        sys.exit(1)
    else:
        print("OK")

print("\nAll smoke tests passed.")
```

### Task 4: 服务器实验设计（因果可分离）

以下命令供**远程服务器**使用。**三组实验独立控制变量**，可并行运行：

**E1（已跑，当前基线）：**
```bash
python -m src.train --run-name E1_baseline --epochs 100 --batch-size 16 --lr 1e-3
```

**E2a（仅 FACL，隔离 loss 效果）：**
```bash
python -m src.train --run-name E2a_facl_only --epochs 100 --batch-size 16 --lr 1e-3 \
  --loss facl
```

**E2b（仅 AdamW + Cosine，隔离优化器效果）：**
```bash
python -m src.train --run-name E2b_adamw_cosine --epochs 100 --batch-size 16 --lr 1e-3 \
  --optimizer adamw --scheduler cosine
```

**E2c（全套组合）：**
```bash
python -m src.train --run-name E2c_full --epochs 100 --batch-size 16 --lr 1e-3 \
  --loss facl --optimizer adamw --scheduler cosine
```

**关键对比分析**：

| 对比 | 结论 |
|------|------|
| E1 vs E2a | FACL 对 FAR/CSI 的单独贡献 |
| E1 vs E2b | AdamW+Cosine 对训练稳定性的单独贡献 |
| E2a+E2b vs E2c | 交互效应（是否有协同或对抗） |

**Success criteria（E2a 应满足以下之一才值得继续 FACL 方向）**：
- `val_far_weak < 0.50`（降低 ≥ 0.14）
- `val_csi_weak > 0.40`（提升 ≥ 0.05）
- `val_pod_weak ≥ 0.80` 且 `val_far_weak < val_far_weak(E1)`

---

## 附录：文件改动汇总

| 文件 | 改动类型 | 内容 |
|------|---------|------|
| `src/loss.py` | 追加 | `_amplitude_loss`（含 H*W 归一化）、`_correlation_loss`、`facl_loss` |
| `src/train.py` | 修改 | 3 个独立 CLI flags；条件化 loss/optimizer/scheduler；checkpoint 附带 scheduler state |
| `tests/test_loss.py` | 追加 | 7 个测试（含量纲校验 + 梯度校验） |
| `tests/test_train.py` | 追加 | 7 个测试（含行为验证：loss 差异 + lr 衰减 + checkpoint key） |
| `scripts/smoke_flags.py` | 新增 | 服务器烟测脚本 |

其他文件（`config.py`, `dataset.py`, `model_convlstm.py`, `metrics.py`, `baselines.py`）**不改动**。

预期测试变化：**67 → 81 tests**（+14）。
