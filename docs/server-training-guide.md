# 远程服务器训练对接说明

**写给远程服务器上的 Claude**
**日期**: 2026-03-19（更新：2026-03-19）
**项目**: CS-OmniMamba-2025 — 河北降水临近预报

---

## 项目概述

- **任务**: 序列→单步降水临近预报（6 分钟后）
- **输入**: RADAR[B,10,1,66,70]（t-9…t）+ PWV[B,1,1,66,70]（t）
- **输出**: RAIN[B,1,1,66,70]（t+1）
- **模型**: ConvLSTMModel（`src/model_convlstm.py`）
- **数据集**: 28,359 个样本，train=20,950 / val=3,589 / test=3,820

---

## 目录结构（关键部分）

```
CS-OmniMamba-2025/
├── src/
│   ├── config.py          # 路径、常量、split 日期
│   ├── dataset.py         # RainDataset
│   ├── model_convlstm.py  # ConvLSTMModel（当前模型）
│   ├── train.py           # 训练入口（含 eval-only / resume / --loss / --optimizer / --scheduler）
│   ├── metrics.py         # CSI/POD/FAR/MSE/MAE
│   └── loss.py            # weighted_mse_loss + facl_loss
├── scripts/
│   ├── visualize.py       # 可视化工具
│   └── smoke_flags.py     # 服务器烟测脚本（验证新 flags）
├── tests/                 # 81 个测试（本地已全部通过）
├── runs/                  # 训练输出目录（自动创建）
├── radar_preprocessed/    # 预处理后的 RADAR（28,710 张 PNG）
├── 70×66_rain_pic_hebei_2025/   # RAIN 数据
├── PWV_2025_S/            # PWV 数据
└── requirements.txt
```

---

## 环境准备

```bash
# 确认 Python 版本（需要 3.10+）
python --version

# 安装依赖
pip install -r requirements.txt

# 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 运行测试确认代码无问题（可选，约 30s）
python -m pytest tests/ -q
```

---

## 数据路径说明

`src/config.py` 中的路径基于项目根目录自动推导，**无需手动修改**。
确认以下目录存在且非空：

```bash
ls radar_preprocessed/ | wc -l   # 应为 28710
ls 70×66_rain_pic_hebei_2025/ | wc -l   # 应为 28710
ls PWV_2025_S/ | wc -l           # 应为 28710（或接近）
```

---

## 训练命令

### 新增 CLI Flags（2026-03-19）

`train.py` 新增三个独立 flags，可任意组合：

| Flag | 选项 | 默认 | 说明 |
|------|------|------|------|
| `--loss` | `mse` / `facl` | `mse` | `facl` = weighted_mse + FACL（频域振幅+相关损失） |
| `--optimizer` | `adam` / `adamw` | `adam` | `adamw` 使用 weight_decay=1e-4 |
| `--scheduler` | `none` / `cosine` | `none` | `cosine` = CosineAnnealingLR，lr 从初始值衰减到 0 |

---

### E1 Baseline（已跑 / 参考基线）

```bash
python -m src.train \
  --run-name E1_baseline \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --ckpt-every 5 \
  --vis-every 10 \
  --device cuda
```

---

### E2a — 仅 FACL loss（隔离 loss 效果）

```bash
python -m src.train \
  --run-name E2a_facl_only \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --loss facl \
  --ckpt-every 5 \
  --vis-every 10 \
  --device cuda
```

### E2b — 仅 AdamW + CosineAnnealingLR（隔离优化器效果）

```bash
python -m src.train \
  --run-name E2b_adamw_cosine \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --optimizer adamw \
  --scheduler cosine \
  --ckpt-every 5 \
  --vis-every 10 \
  --device cuda
```

### E2c — 全套组合

```bash
python -m src.train \
  --run-name E2c_full \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --loss facl \
  --optimizer adamw \
  --scheduler cosine \
  --ckpt-every 5 \
  --vis-every 10 \
  --device cuda
```

> E2a / E2b / E2c 可并行运行（各自独立 run_dir）。

### E3 — FACL lambda=0.1 + AdamW + Cosine（当前推荐）

```bash
python -m src.train \
  --run-name E3_facl_lambda01 \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --loss facl \
  --lambda-facl 0.1 \
  --optimizer adamw \
  --scheduler cosine \
  --ckpt-every 5 \
  --vis-every 10 \
  --device cuda
```

**动机**: E2a/E2c 中 lambda=1.0 导致 POD 从 0.91 跌至 0.75，best_epoch 仅 6/24。调小到 0.1 预期 POD 回升至 0.85+，CSI 突破 0.50。

**实验对比矩阵**：

| 对比 | 结论 |
|------|------|
| E1 vs E2a | FACL 对 FAR/CSI 的单独贡献 |
| E1 vs E2b | AdamW+Cosine 对训练稳定性的单独贡献 |
| E2a+E2b vs E2c | 交互效应（协同或对抗） |

**FACL success criteria（E2a 应满足以下之一才值得继续）**：
- `val_far_weak < 0.50`（当前 E1 = 0.64）
- `val_csi_weak > 0.40`（当前 E1 = 0.35）
- `val_pod_weak ≥ 0.80` 且 `val_far_weak` 低于 E1

---

### 烟测（首次运行前验证新 flags）

```bash
python scripts/smoke_flags.py
```

每个 flag 组合跑 1 epoch，确认无报错后再启动正式训练。

---

**预期输出目录** `runs/{run_name}/`：
- `config.json` — 训练配置（训练开始时写出，含 loss/optimizer/scheduler 字段）
- `metrics.json` — 每 epoch 指标（追加写入）
- `best.pt` — 最优 checkpoint（val_csi_weak 最高时更新，含 scheduler state）
- `last.pt` — 最终 checkpoint
- `epoch_005.pt`, `epoch_010.pt`, ... — 周期 checkpoint
- `baselines.json` — Persistence / Zero baseline 对比（训练结束后写出）
- `vis/vis_epoch_010.png`, `vis_epoch_020.png`, ... — 样本预测图

### 如果训练中断，续训

```bash
python -m src.train \
  --epochs 100 \
  --resume runs/E1_baseline/last.pt \
  --batch-size 16 \
  --lr 1e-3 \
  --ckpt-every 5 \
  --vis-every 10 \
  --device cuda
```

注意：`--resume` 时 **不需要** 传 `--run-name`，输出目录自动从 checkpoint 路径推导。续训时如果原来用了 `--loss facl` 等 flags，**必须重新传入**，否则会用默认值。

### 训练完成后评估 val（验证）

```bash
python -m src.train \
  --eval-only \
  --ckpt runs/E1_baseline/best.pt \
  --split val \
  --device cuda
```

输出：`runs/E1_baseline/eval_val.json`

### 最终评估 test（仅在确认模型选型后执行）

```bash
python -m src.train \
  --eval-only \
  --ckpt runs/E1_baseline/best.pt \
  --split test \
  --device cuda
```

输出：`runs/E1_baseline/eval_test.json`

> ⚠️ test set 是"封印"的，只在所有实验对比完成、模型选型确定后才执行一次。

---

## 训练监控

每个 epoch 会打印：
```
Epoch 001 | train_loss=0.0312 | val_csi_weak=0.0870 | val_far_weak=0.9200 | val_mse=0.00412
  → New best: epoch 1, val_csi_weak=0.0870
```

**关注指标**：
- `val_csi_weak` 上升 → 模型在学习（目标：接近 Persistence baseline 的 0.9276）
- `train_loss` 下降但 `val_csi_weak` 不涨 → 过拟合
- 出现 `NaN` → 立即停止，报告

---

## 回传内容与格式约定

每个实验跑完后，请回传以下内容：

### 必须回传（文件，每个实验一份）

| 文件 | 说明 |
|------|------|
| `runs/{run_name}/metrics.json` | 完整训练曲线 |
| `runs/{run_name}/baselines.json` | baseline 对比 |
| `runs/{run_name}/config.json` | 训练配置（含 loss/optimizer/scheduler） |
| `runs/{run_name}/eval_val.json` | val 最终评估 |
| `runs/{run_name}/best.pt` | 最优模型权重 |
| `runs/{run_name}/vis/vis_epoch_*.png` | 所有可视化图 |

### 必须回传（文字报告，每个实验一份）

```
## {run_name} 训练报告

**训练状态**: 完成 / 中断（说明原因）
**配置**: loss={loss} optimizer={optimizer} scheduler={scheduler}
**总 epoch**: 100
**最优 epoch**: XX（val_csi_weak=X.XXXX）
**最终 epoch**: 100（val_csi_weak=X.XXXX）

**关键指标（best checkpoint）**:
- val_csi_weak:   X.XXXX
- val_pod_weak:   X.XXXX
- val_far_weak:   X.XXXX
- val_csi_strong: X.XXXX
- val_mse:        X.XXXXX

**Persistence baseline（val）**:
- csi_weak: X.XXXX
- mse:      X.XXXXX

**异常情况**: 无 / （描述）

**训练曲线观察**:
（简述 loss 是否收敛、CSI 趋势、是否有震荡等）
```

### 可选回传

- `runs/{run_name}/eval_test.json` — 仅在用户明确要求时执行并回传
- `runs/{run_name}/epoch_*.pt` — 仅在需要分析特定 epoch 时回传

---

## 常见问题

**Q: CUDA out of memory**
减小 batch size：`--batch-size 4`

**Q: DataLoader 报错 / 找不到文件**
检查数据目录路径，确认 `radar_preprocessed/` 存在

**Q: val_csi_weak 一直是 NaN**
说明 val set 里没有雨像素，检查数据归一化是否正确：
```bash
python -c "
from src.dataset import RainDataset
from src.config import RADAR_PREP_DIR, PWV_DIR, RAIN_DIR
ds = RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR)
print(len(ds))
"
```

**Q: 训练速度很慢**
增加 workers：`--workers 4`（或根据服务器 CPU 核数调整）

---

## 指标定义参考

| 指标 | 含义 | NaN 条件 |
|------|------|---------|
| `csi_weak` | 弱降水 CSI（阈值=2/255） | val set 无雨时 |
| `pod_weak` | 弱降水命中率 | val set 无雨时 |
| `far_weak` | 弱降水误报率 | 模型完全不预测雨时 |
| `csi_strong` | 强降水 CSI（阈值=55/255） | val set 无强雨时 |
| `mse` | 全图均方误差 | 不会 NaN |
| `mae_rain` | 雨区平均绝对误差 | val set 无雨时 |

**Persistence baseline 参考值（val set）**: `csi_weak ≈ 0.9276`
这是"直接用当前帧预测下一帧"的性能上界，模型需要接近这个值才算有效。
