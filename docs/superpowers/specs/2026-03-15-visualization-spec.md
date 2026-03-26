# 可视化方案规范

> 日期：2026-03-15
> 适用范围：`CS-OmniMamba-2025` 项目所有可视化输出
> 脚本目标路径：`scripts/visualize.py`
> 输出目录：`runs/{run_name}/vis/`
>
> **当前阶段**：单步输出（仅预测 t+1）
> **未来扩展**：计划扩展至三步输出（t+1 / t+2 / t+3），本规范已预留扩展接口，见 §2.2 和 §3

---

## 1. 配色方案

### 1.1 降水图（预测 / 真实）

使用 7 级离散色表，继承自 `北斗极端降水预测(新)/优化模型/ver_1.0.py`：

```python
from matplotlib.colors import ListedColormap
colors_rgb = [
    (97,  40,  31),   # 0: 深褐 — 强降水
    (250,  1, 246),   # 1: 紫红
    (0,    0, 254),   # 2: 深蓝
    (101, 183, 252),  # 3: 浅蓝
    (61,  185,  63),  # 4: 深绿
    (166, 242, 142),  # 5: 浅绿
    (254, 254, 254),  # 6: 白色 — 无降水
]
colors_norm = [(r/255, g/255, b/255) for r, g, b in colors_rgb]
cmap = ListedColormap(colors_norm, name="custom_rain_discrete")
```

- `vmin=0, vmax=1`（归一化降水值）
- `interpolation='nearest'`（保持离散感，不做双线性模糊）

### 1.2 RADAR 输入帧

- `cmap='gray'`，`vmin=0, vmax=1`
- `interpolation='nearest'`

### 1.3 PWV 输入帧

- `cmap='gray'`，`vmin=0, vmax=1`
- `interpolation='nearest'`

### 1.4 差值图（预测 − 真实）

- `cmap='RdBu_r'`（红=正偏差/虚报，蓝=负偏差/漏报）
- `vmin=-diff_vmax, vmax=diff_vmax`，`vcenter=0`（`diff_vmax` 默认 0.5，可通过 `--diff-vmax` CLI 参数覆盖）

---

## 2. 图表类型与内容

### 2.1 训练曲线图 `loss_curve.png`

- **内容**：`train_loss`、`val_mse`、`val_csi_weak`、`val_far_weak` 随 epoch 变化
- **子图**：3 行 × 1 列
  - 上：`train_loss` vs epoch
  - 中：`val_mse` vs epoch
  - 下：`val_csi_weak` 和 `val_far_weak` vs epoch（同一子图，双线）
- **参考线**：若传入 `--csi-persistence`，在下图画水平虚线标注 persistence baseline CSI 值；未传入则不画
- **数据来源**：读取 `runs/{run_name}/metrics.json`
- **调用方式**：`python scripts/visualize.py --run-name convlstm_e1 --mode curves --csi-persistence 0.9276`

### 2.2 样本对比图 `vis_epoch_{epoch:03d}.png`

每张图展示 N 个样本（默认 N=4），每个样本一行。

**列布局随输出步数自动扩展：**

#### 当前：单步输出（`num_steps=1`），共 5 列

| 列 | 内容 | 色表 |
|---|---|---|
| 0 | RADAR 代表性输入帧 `radar[:,9]`（t 时刻，仅展示 10 帧序列中的最新一帧） | gray |
| 1 | PWV 输入 | gray |
| 2 | 预测 RAIN(t+1) | custom_rain_cmap |
| 3 | 真实 RAIN(t+1) | custom_rain_cmap |
| 4 | 差值（预测−真实） | RdBu_r |

#### 未来：三步输出（`num_steps=3`），共 2+3×3=11 列

| 列 | 内容 | 色表 |
|---|---|---|
| 0 | RADAR 代表性输入帧（t 时刻） | gray |
| 1 | PWV 输入 | gray |
| 2 | 预测 RAIN(t+1) | custom_rain_cmap |
| 3 | 真实 RAIN(t+1) | custom_rain_cmap |
| 4 | 差值 t+1 | RdBu_r |
| 5 | 预测 RAIN(t+2) | custom_rain_cmap |
| 6 | 真实 RAIN(t+2) | custom_rain_cmap |
| 7 | 差值 t+2 | RdBu_r |
| 8 | 预测 RAIN(t+3) | custom_rain_cmap |
| 9 | 真实 RAIN(t+3) | custom_rain_cmap |
| 10 | 差值 t+3 | RdBu_r |

**实现说明**：模型输出约定为 `pred.shape = [B, num_steps, H, W]`。`plot_samples` 通过 `pred.shape[1]` 判断 `num_steps`（当前仅支持 1 或 3，其他值 raise `ValueError`）。

- 图标题行只在第一行显示
- 每个子图右侧加 colorbar（同行统一 colorbar 即可）
- 图尺寸：单步 `figsize=(15, 4*N)`；三步 `figsize=(28, 4*N)`
- **调用方式**：`python scripts/visualize.py --run-name convlstm_e1 --mode samples --epoch 20 --n-samples 4`
- **文件命名**：
  - 指定 `--epoch 20` → `vis_epoch_020.png`
  - 省略 `--epoch`（读 `last.pt`）→ `vis_last.png`

### 2.3 阈值敏感性图 `threshold_curve.png`（可选，Phase 1 后期）

- X 轴：阈值 `thresh` 从 0 到 1（步长 0.02）
- Y 轴：对应的 `CSI / POD / FAR`
- 标注 `THRESH_WEAK`（2/255）和 `THRESH_STRONG`（55/255）两条竖虚线
- **数据来源**：`plot_threshold` 自行加载 checkpoint + val/test DataLoader，实时推理全集并统计各阈值下的指标（不保存中间 pred.npy，不依赖 metrics.json）

---

## 3. 脚本结构

```
scripts/visualize.py
├── get_custom_rain_cmap()       # 返回 7 级离散 ListedColormap
├── plot_curves(run_name, csi_persistence)
│                                # 读 metrics.json，画 3 行训练曲线
├── plot_samples(run_name, epoch, n_samples, split, diff_vmax)
│   # pred.shape = [B, num_steps, H, W]，依 pred.shape[1] 判断步数
│   # num_steps=1 → 5 列；num_steps=3 → 11 列；其他 raise ValueError
├── plot_threshold(run_name, epoch, split)
│   # 加载 checkpoint + DataLoader，实时推理全集，统计各阈值下指标
└── main()                       # argparse 入口
    --run-name        str    (必填)
    --mode            {curves, samples, threshold, all}  (默认 all)
    --epoch           int    (samples/threshold 模式用，默认读 last.pt)
    --n-samples       int    (默认 4)
    --split           {val, test}  (默认 val)
    --csi-persistence float  (可选，persistence baseline CSI 参考线值)
    --diff-vmax       float  (默认 0.5，差值图色标范围)
```

`plot_samples` 流程：
1. 加载 `runs/{run_name}/last.pt`（或指定 epoch 的 checkpoint）
2. 构建 val/test DataLoader（`shuffle=False, batch_size=n_samples`）
3. 取第一个 batch，前向推理
4. 绘制对比图并保存到 `runs/{run_name}/vis/`
   - 指定 `--epoch` → `vis_epoch_{epoch:03d}.png`
   - 未指定 → `vis_last.png`

---

## 4. 输出文件约定

```
runs/{run_name}/
├── metrics.json
├── last.pt
├── epoch_005.pt
└── vis/
    ├── loss_curve.png
    ├── vis_epoch_020.png     # --epoch 20
    ├── vis_last.png          # 省略 --epoch 时
    └── threshold_curve.png   # 可选，--mode threshold
```

---

## 5. 不做的事（范围约束）

- 不做动图（GIF/MP4）
- 不做交互式图表（plotly/streamlit）
- 不在训练循环内调用可视化（避免拖慢训练；可视化是离线独立脚本）
- `plot_samples` 不在服务器上自动执行，由用户手动调用
- 可视化脚本既可在远程服务器上对刚训完的结果运行，也可在本地对已同步的 `runs/{run_name}/` 目录运行

---

## 6. 依赖

仅使用已有环境中的库：
- `matplotlib`
- `numpy`
- `torch`
- `Pillow`（通过 dataset 间接已有）

无需新增依赖。
