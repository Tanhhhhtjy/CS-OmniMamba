# 真实模型实现方案 — 2026-03-15

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用一个真实的可训练模型替换 StubModel，使其在验证集上的 CSI_weak 显著高于 ZeroBaseline（0.0000）并逼近 Persistence 基准线（0.9276）。

**Architecture:** 以轻量 ConvLSTM 为首选模型（见 §1 选型分析），保持已有 `forward(radar, pwv) -> Tensor[B,1,H,W]` 接口不变，只新增 `src/model_convlstm.py` 一个文件。`train.py` 修改两处：import 行 + checkpoint 路径（共 2 行）。`ConvLSTMModel.__init__` 接受 `**kwargs` 以兼容现有 `StubModel(t=T)` 调用，行 116 不需要改动。

**Tech Stack:** Python 3.11, PyTorch 2.5.1, conda env `pytorch_gpu`，项目根 `e:\CS-OmniMamba-2025`

---

## § 1 选型分析与建议

### 1.1 当前基准线

| 基线 | CSI_weak | MSE |
|------|---------|-----|
| Persistence | **0.9276** | 0.00003 |
| Zero | 0.0000 | 0.00190 |
| StubModel | 0.0000 | 0.00190 |

Persistence 极强的根本原因：6 分钟步长下降水场几乎不变，直接复制上一帧即可得到极高 CSI。
**实际目标**：第一个真实模型应先超越 Zero，能预测出雨区位置，再迭代逼近/超越 Persistence。

### 1.2 候选架构对比

| 架构 | 参数量 | 能捕捉时序 | 实现难度 | 收敛速度 | 推荐程度 |
|------|--------|----------|---------|---------|---------|
| **ConvLSTM（推荐）** | ~500K | ✅ 显式 | 中 | 快 | ⭐⭐⭐ |
| 轻量时空 CNN（S3D 变体） | ~300K | ⚠️ 隐式 | 低 | 很快 | ⭐⭐ |
| UNet-3D | ~2M+ | ⚠️ 隐式 | 中 | 慢 | ⭐ |
| OmniMamba | ~5M+ | ✅ SSM | 高 | 慢 | ❌（留后续） |

**推荐：ConvLSTM**

理由：
1. **显式时序建模**：对 t-9…t 共 10 帧做循环推进，每步隐状态携带历史，直接针对"序列→1 帧预测"的任务结构
2. **小图适配**：66×70 的空间分辨率很低，ConvLSTM 在当前小空间设置下参数效率更高
3. **可解释性**：隐状态可视化便于调试（哪些历史帧对预测最重要）
4. **接口简洁**：天然适配 `forward(radar[B,T,1,H,W], pwv[B,1,1,H,W])` 签名
5. **文献充分**：ShiNet(2015) 已在雷达外推任务上验证，社区实现丰富，不依赖新框架

**备选：轻量时空 CNN（若 ConvLSTM 收敛问题难以解决时切换）**
保留 T 帧时间维，用 3D 卷积在时空联合域隐式建模时序，比 ConvLSTM 更易优化但时序感知能力弱。

**OmniMamba 不作为首个真实模型**：其优势在于长程依赖和高分辨率场景，在 66×70 × 10 帧的设置下优势难以充分体现，且调试成本较高，优先级较低。待 ConvLSTM 稳定后，可作为 E3/E4 消融实验对象加以验证。

### 1.3 ConvLSTM 设计规格

```
输入 radar: [B, 10, 1, 66, 70]
输入 pwv:   [B,  1, 1, 66, 70]

流程：
1. 合并：在时间维度末尾拼接 PWV，形成 [B, 11, 1, 66, 70]
   （PWV 作为第 11 个"伪时序帧"输入，让模型自行学习如何利用背景场）
2. ConvLSTM 层（2层，hidden_dim=64, kernel=3×3）：
   逐帧推进，输出最后一步隐状态 h_{11}: [B, 64, 66, 70]
3. 输出头：Conv2d(64→1) + Sigmoid → [B, 1, 66, 70]

参数量估算：~480K（在 GPU 上 batch=8 约占 ~50MB）
```

**ConvLSTM Cell 公式（标准实现）：**
```
i = σ(W_xi * x_t + W_hi * h_{t-1} + b_i)   # input gate
f = σ(W_xf * x_t + W_hf * h_{t-1} + b_f)   # forget gate
g = tanh(W_xg * x_t + W_hg * h_{t-1} + b_g) # cell gate
o = σ(W_xo * x_t + W_ho * h_{t-1} + b_o)   # output gate
c_t = f ⊙ c_{t-1} + i ⊙ g
h_t = o ⊙ tanh(c_t)
其中 * 为二维卷积（padding=kernel//2 保持空间不变）
```

---

## § 2 文件结构

```
e:\CS-OmniMamba-2025\
├── src/
│   ├── model_convlstm.py      ← 新建（ConvLSTM Cell + 多层 ConvLSTM + 主模型）
│   └── train.py               ← 修改多处（见 Task 3 详细说明）
├── tests/
│   └── test_model_convlstm.py ← 新建（shape / forward / gradient 测试）
└── runs/                      ← 自动创建，每次训练一个子目录
    └── {run_name}/
        ├── epoch_005.pt       ← 每 --ckpt-every 轮保存一次
        ├── epoch_010.pt
        ├── ...
        ├── last.pt            ← 训练结束时保存
        └── metrics.json       ← 每 epoch 追加一行指标
```

只新增 1 个源文件 + 1 个测试文件，`train.py` 修改多处（import、argparse、训练循环、checkpoint 逻辑），其余文件不动。

---

## § 3 实现计划

### Chunk 1：ConvLSTM 模型实现（TDD）

---

#### Task 1：写失败的测试

**Files:**
- Create: `tests/test_model_convlstm.py`

- [ ] **Step 1：新建测试文件**

```python
# tests/test_model_convlstm.py
"""Tests for ConvLSTMModel."""
import torch
import pytest
from src.config import T, H, W

B = 2   # batch size for all tests


class TestConvLSTMCell:
    def test_output_shape(self):
        """ConvLSTMCell 输出 h 和 c 的 shape 应与输入一致。"""
        from src.model_convlstm import ConvLSTMCell
        cell = ConvLSTMCell(in_channels=1, hidden_channels=16, kernel_size=3)
        x = torch.zeros(B, 1, H, W)
        h = torch.zeros(B, 16, H, W)
        c = torch.zeros(B, 16, H, W)
        h_new, c_new = cell(x, h, c)
        assert h_new.shape == (B, 16, H, W)
        assert c_new.shape == (B, 16, H, W)

    def test_cell_not_all_zero(self):
        """非零输入应产生非零隐状态（验证门控激活正常）。"""
        from src.model_convlstm import ConvLSTMCell
        cell = ConvLSTMCell(in_channels=1, hidden_channels=16, kernel_size=3)
        x = torch.ones(B, 1, H, W)
        h = torch.zeros(B, 16, H, W)
        c = torch.zeros(B, 16, H, W)
        h_new, c_new = cell(x, h, c)
        assert h_new.abs().sum() > 0


class TestConvLSTMModel:
    def test_forward_output_shape(self):
        """主模型 forward 输出 shape 应为 [B, 1, H, W]。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel()
        radar = torch.zeros(B, T, 1, H, W)
        pwv   = torch.zeros(B, 1, 1, H, W)
        out = model(radar, pwv)
        assert out.shape == (B, 1, H, W), f"Expected ({B},1,{H},{W}), got {out.shape}"

    def test_output_range(self):
        """输出值应在 [0, 1]（Sigmoid 激活）。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel()
        radar = torch.rand(B, T, 1, H, W)
        pwv   = torch.rand(B, 1, 1, H, W)
        out = model(radar, pwv)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_gradient_flows(self):
        """反向传播应能更新所有参数（无梯度断裂）。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel()
        radar = torch.rand(B, T, 1, H, W)
        pwv   = torch.rand(B, 1, 1, H, W)
        target = torch.rand(B, 1, H, W)
        out = model(radar, pwv)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN grad for {name}"

    def test_deterministic_eval(self):
        """eval 模式下两次 forward 结果完全一致（无 Dropout）。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel()
        model.eval()
        torch.manual_seed(42)
        radar = torch.rand(B, T, 1, H, W)
        pwv   = torch.rand(B, 1, 1, H, W)
        with torch.no_grad():
            out1 = model(radar, pwv)
            out2 = model(radar, pwv)
        assert torch.allclose(out1, out2)

    def test_custom_hidden_dim(self):
        """支持自定义 hidden_dim 参数。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel(hidden_dim=32, num_layers=1)
        radar = torch.zeros(B, T, 1, H, W)
        pwv   = torch.zeros(B, 1, 1, H, W)
        out = model(radar, pwv)
        assert out.shape == (B, 1, H, W)
```

- [ ] **Step 2：确认测试失败**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/test_model_convlstm.py -v
```

预期结果：`ModuleNotFoundError: No module named 'src.model_convlstm'`（7 tests collected, 7 errors）
（TestConvLSTMCell × 2 + TestConvLSTMModel × 5 = 7）

---

#### Task 2：实现 ConvLSTMModel

**Files:**
- Create: `src/model_convlstm.py`

- [ ] **Step 3：新建 `src/model_convlstm.py`**

```python
"""
ConvLSTM-based precipitation nowcasting model.

Interface (identical to StubModel):
    forward(radar[B,T,1,H,W], pwv[B,1,1,H,W]) -> Tensor[B,1,H,W]

Architecture:
    1. Concatenate PWV as the (T+1)-th "pseudo-frame" → [B, T+1, 1, H, W]
    2. Multi-layer ConvLSTM processes frames sequentially
    3. Final hidden state h: [B, hidden_dim, H, W]
    4. Output head: Conv2d(hidden_dim→1) + Sigmoid → [B, 1, H, W]
"""
from __future__ import annotations

import torch
import torch.nn as nn
from src.config import T, H, W


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al., 2015)."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        # Gates: i, f, g, o — fused into a single 4× conv for efficiency
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
        )

    def forward(
        self,
        x: torch.Tensor,          # [B, in_channels, H, W]
        h: torch.Tensor,          # [B, hidden_channels, H, W]
        c: torch.Tensor,          # [B, hidden_channels, H, W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)         # [B, in+hidden, H, W]
        gates = self.conv(combined)                  # [B, 4*hidden, H, W]
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMModel(nn.Module):
    """
    Multi-layer ConvLSTM for Seq→1 precipitation nowcasting.

    Args:
        hidden_dim:  Number of feature channels in each ConvLSTM hidden state.
        num_layers:  Number of stacked ConvLSTM layers.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, **kwargs) -> None:
        # **kwargs 兼容 train.py 中遗留的 StubModel(t=T) 调用，t 参数被静默忽略
        super().__init__()
        self.num_layers = num_layers

        # Build stacked ConvLSTM cells
        # Layer 0: input channels = 1 (grayscale)
        # Layer k>0: input channels = hidden_dim (from previous layer's h)
        cells = []
        for i in range(num_layers):
            in_ch = 1 if i == 0 else hidden_dim
            cells.append(ConvLSTMCell(in_ch, hidden_dim, kernel_size=3))
        self.cells = nn.ModuleList(cells)

        # Output projection: hidden → 1 channel rain map
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        radar: torch.Tensor,   # [B, T, 1, H, W]
        pwv:   torch.Tensor,   # [B, 1, 1, H, W]
    ) -> torch.Tensor:         # [B, 1, H, W]
        B, T_in, C, H, W = radar.shape

        # Concatenate PWV as the (T+1)-th frame → [B, T+1, 1, H, W]
        seq = torch.cat([radar, pwv], dim=1)          # [B, T+1, 1, H, W]
        T_seq = seq.shape[1]

        # Initialise hidden states for all layers
        device = radar.device
        h = [torch.zeros(B, cell.hidden_channels, H, W, device=device)
             for cell in self.cells]
        c = [torch.zeros(B, cell.hidden_channels, H, W, device=device)
             for cell in self.cells]

        # Step through the sequence
        for t in range(T_seq):
            x_t = seq[:, t]           # [B, 1, H, W]
            for layer_idx, cell in enumerate(self.cells):
                inp = x_t if layer_idx == 0 else h[layer_idx - 1]
                h[layer_idx], c[layer_idx] = cell(inp, h[layer_idx], c[layer_idx])

        # Use last layer's final hidden state as feature map
        out = self.output_conv(h[-1])   # [B, 1, H, W]
        return torch.sigmoid(out)
```

- [ ] **Step 4：运行测试，确认全部通过**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/test_model_convlstm.py -v
```

预期结果：`7 passed`

- [ ] **Step 5：确认全部 27+7=34 个测试仍然全绿**

```bash
conda run -n pytorch_gpu --no-capture-output python -m pytest tests/ -v
```

预期结果：`34 passed`

---

### Chunk 2：接入训练循环

---

#### Task 3：修改 train.py 切换到真实模型

**Files:**
- Modify: `src/train.py`（import、argparse、训练循环、checkpoint 逻辑，共 4 处）

- [ ] **Step 6：修改 train.py**

**改动 1 — import（第 26 行）**
```python
# 原
from src.model_stub import StubModel
# 改为
from src.model_convlstm import ConvLSTMModel as StubModel
```
> 别名保持 `StubModel` 不变，第 116 行 `StubModel(t=T)` 无需修改（`**kwargs` 已兼容）。

**改动 2 — argparse 新增两个参数（在现有 `--device` 行之后）**
```python
parser.add_argument("--run-name",   type=str, default="convlstm",
                    help="子目录名，输出写入 runs/{run_name}/")
parser.add_argument("--ckpt-every", type=int, default=5,
                    help="每隔多少 epoch 保存一次分段 checkpoint")
```

**改动 3 — 训练前创建 run 目录**
```python
import json
run_dir = Path("runs") / args.run_name
run_dir.mkdir(parents=True, exist_ok=True)
```

**改动 4 — 训练循环内每 epoch 结尾追加 metrics.json + 每 ckpt_every 轮保存分段 checkpoint**

在 `print(f"Epoch {epoch:03d} ...")` 之后插入：
```python
# 追加 metrics
entry = {"epoch": epoch, "train_loss": train_loss,
         "val_csi_weak": csi, "val_far_weak": far, "val_mse": mse}
metrics_path = run_dir / "metrics.json"
history = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
history.append(entry)
metrics_path.write_text(json.dumps(history, indent=2))

# 分段 checkpoint
if epoch % args.ckpt_every == 0:
    torch.save(model.state_dict(), run_dir / f"epoch_{epoch:03d}.pt")
```

**改动 5 — 训练结束后保存 last.pt（替换原 checkpoints/stub_last.pt 逻辑）**
```python
# 原 checkpoints/stub_last.pt 逻辑全部替换为：
torch.save(model.state_dict(), run_dir / "last.pt")
print(f"Checkpoint saved → {run_dir / 'last.pt'}")
```

- [ ] **Step 7：确认 train.py 可以 import（无语法错误）**

```bash
conda run -n pytorch_gpu --no-capture-output python -c "from src.train import main; print('OK')"
```

预期：打印 `OK`。

---

#### Task 4：3-epoch 快速验证训练

- [ ] **Step 8：运行 3 epoch 快速验证**

```bash
conda run -n pytorch_gpu --no-capture-output python -m src.train --epochs 3 --batch-size 8 --workers 0 --run-name convlstm_smoke
```

**关注指标（判断标准）：**

| 指标 | 通过标准 | 说明 |
|------|---------|------|
| `train_loss` | 呈下降趋势（epoch 3 < epoch 1） | 模型在学习 |
| `val_csi_weak` | > 0.0000（任意正值） | 模型开始预测出雨区 |
| `val_mse` | < 0.00190（低于 Zero baseline） | 优于什么都不预测 |
| 无 crash / NaN loss | ✅ | 数值稳定性 |

> **如果 3 epoch 内 `val_csi_weak` 仍然 = 0**，先确认：① `train_loss` 是否在下降；② 模型输出最大值 `pred.max()` 是否已高于 `THRESH_WEAK`（2/255≈0.0078）。若 loss 在下降但 CSI 仍为 0，说明预测值尚低于阈值，继续训练即可。若 loss 不下降或 pred 始终全零，需先排查梯度是否正常（Step 9），再决定是否延长 epoch。

- [ ] **Step 9：若 3 epoch 内 loss 出现 NaN，执行诊断**

```bash
# 诊断命令（降低 LR 重试）
conda run -n pytorch_gpu --no-capture-output python -m src.train --epochs 3 --batch-size 8 --workers 0 --lr 1e-4 --run-name convlstm_debug
```

NaN loss 最常见原因：LR 过大（默认 1e-3 是合理起点，ConvLSTM 偶发梯度爆炸可降至 1e-4）。

---

### Chunk 3：20-epoch 基准训练

---

#### Task 5：完整基准训练

- [ ] **Step 10：启动 20-epoch 基准训练**

```bash
conda run -n pytorch_gpu --no-capture-output python -m src.train --epochs 20 --batch-size 8 --workers 0 --run-name convlstm_e1
```

预期耗时：约 20-40 分钟（GPU）。

- [ ] **Step 11：记录结果**

训练完成后，记录以下数值到本文档或交接文档：

```
Epoch 020 | train_loss=? | val_csi_weak=? | val_far_weak=? | val_mse=?
[Persistence] csi_weak=0.9276 | mse=0.00003
[Zero      ]  csi_weak=0.0000 | mse=0.00190
```

**成功标准（20 epoch 后）：**

| 指标 | 最低通过线 | 说明 |
|------|----------|------|
| `val_csi_weak` | > 0.30 | 明显超越 Zero，证明模型能预测雨区 |
| `val_mse` | < 0.00100 | 低于 Zero baseline MSE 的一半 |
| `train_loss` 趋势 | 持续下降 | 未过早收敛/震荡 |

> **如果 val_csi_weak < 0.30**：不意味着模型失败，而是需要更多 epoch 或调参。继续到 50 epoch 再评估，同时可尝试提高 `RAIN_WEIGHT`（改为 15 或 20）。

---

## § 4 超参数调参建议

若 20 epoch 后效果不理想，按以下顺序尝试：

| 问题 | 调整建议 | 入口 |
|------|---------|------|
| loss 下降太慢 | lr 从 1e-3 → 5e-4 | `--lr` CLI 参数 |
| val_csi 始终 0 | RAIN_WEIGHT 从 10 → 20 | `src/config.py` 第 19 行 |
| FAR 过高（虚警多） | RAIN_WEIGHT 从 10 → 5 | `src/config.py` 第 19 行 |
| 过拟合（train↓ val↑） | hidden_dim 从 64 → 32，或 num_layers 从 2 → 1 | `src/model_convlstm.py` 中 `ConvLSTMModel.__init__` 默认值 |
| GPU OOM | batch_size 从 8 → 4 | `--batch-size` CLI 参数 |

**注意**：`hidden_dim` 和 `num_layers` 目前不是 CLI 参数，需直接修改 `src/model_convlstm.py` 中的构造函数默认值；`lr`/`batch_size`/`workers`/`run_name`/`ckpt_every` 通过命令行参数传入；`RAIN_WEIGHT` 通过 `src/config.py`。

消融实验建议命名规范：`--run-name convlstm_e1`（主实验）/ `convlstm_e2_no_pwv`（去 PWV）/ `convlstm_e4_ssim`（+SSIM loss），与 spec §6 实验编号对应。

---

## § 5 后续路径（本方案范围之外）

按计划完成本方案后，下一步选项（留 review 后决定）：

1. **消融实验 E0~E2**（见 spec §6）：对比 Persistence / RADAR-only / RADAR+PWV
2. **OmniMamba 实现**：在 ConvLSTM 已有结果的对比基础上，评估是否值得引入
3. **Loss 改进**：+SSIM 项（已延期的 Phase 2 指标）
4. **测试集评估**：仅在模型选型稳定后运行一次测试集，避免信息泄漏

---

## § 6 执行检查清单

实现完成后，确认以下所有项：

- [ ] `tests/test_model_convlstm.py` — 7 tests pass（TestConvLSTMCell × 2 + TestConvLSTMModel × 5）
- [ ] `python -m pytest tests/ -v` — 34 tests pass（原 27 + 新 7）
- [ ] `src/train.py` import 已改为 ConvLSTMModel，新增 `--run-name`、`--ckpt-every` 参数
- [ ] `runs/convlstm_smoke/` 目录存在（3-epoch smoke 产物）
- [ ] 3-epoch 快速验证：无 NaN，loss 下降
- [ ] `runs/convlstm_e1/` 目录存在，包含 `epoch_005.pt`、`epoch_010.pt` ... `epoch_020.pt`、`last.pt`、`metrics.json`
- [ ] 20-epoch 基准训练完成，结果已记录至本文档

---

*文档版本：2026-03-15 rev.2（新增 run_name/ckpt_every/metrics.json 支持）| 作者：Tanh*
