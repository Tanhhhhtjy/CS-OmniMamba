# CS-OmniMamba

基于 PWV（可降水量）+ 雷达回波时序的短临降水预报深度学习框架。

模型输入当前时刻的大气水汽图像（单帧）与过去 66 分钟的雷达回波序列（12帧），输出 **T+1h / T+2h / T+3h** 三个时间步的降水预测。

---

## 目录结构

```
CS-OmniMamba/
├── train.py                  # 训练入口（CLI）
├── data/
│   ├── PWV/                  # 可降水量图像，6分钟一帧
│   ├── RADAR/                # 雷达反射率图像，6分钟一帧
│   └── RAIN/                 # 降水量真值图像，6分钟一帧
├── omnimamba/
│   ├── config.py             # 超参数配置
│   ├── constants.py          # 路径常量
│   ├── data_match.py         # 数据对齐与时序匹配
│   ├── dataset.py            # PyTorch Dataset / DataLoader
│   ├── splits.py             # 训练/验证/测试集时间切分
│   ├── model.py              # 网络结构（CrossAttentionMamba）
│   ├── losses.py             # 损失函数
│   ├── metrics.py            # 评估指标
│   ├── train_loop.py         # 训练与验证循环
│   └── viz.py                # 可视化工具
├── scripts/                  # 辅助脚本（数据重建、评估对比等）
└── tests/                    # 单元测试
```

---

## 数据格式

### 文件命名

所有图像统一命名为时间戳格式：

```
YYYY-MM-DD-HH-mm-SS.png
```

例：`2023-08-01-12-30-00.png`

### 目录布局

```
data/
  PWV/      2023-02-01-00-00-00.png  2023-02-01-00-06-00.png  ...
  RADAR/    2023-05-01-00-00-00.png  ...
  RAIN/     2023-05-01-00-00-00.png  ...
```

### 样本匹配规则（`data_match.py`）

对于每个 PWV 时刻 $t$：
1. 在 RADAR 中找时间戳最接近 $t$（容差 ≤60分钟）的帧作为锚点
2. 从锚点向前取最近 **12 帧**雷达（时间间隔 ≤10分钟容差），不足时前向填充（repeat padding）
3. RAIN 目录中 $t$+1h、$t$+2h、$t$+3h 三帧必须同时存在
4. 非图像文件自动跳过，输出按时间戳升序排列

数据集划分（时间不重叠）：

| 集合 | 时间范围 | 样本数 |
|------|----------|--------|
| 训练集 | 2023-04-30 → 2023-07-30 | ~21,850 |
| 验证集 | 2023-07-31 → 2023-08-06 | ~1,680 |
| 测试集 | 2023-08-07 → 2023-08-31 | ~5,971 |

---

## 模型结构（`model.py`）

### 总体架构：CrossAttentionMamba

```
输入：
  PWV(t)         [B, 1, 66, 70]          单帧可降水量
  Radar Seq      [B, 12, 1, 66, 70]      过去 66 分钟雷达序列

                ┌─────────────────────┐    ┌──────────────────────────────┐
                │   PWV 编码器         │    │   Radar 时序编码器            │
                │  patch_embed1       │    │  shared patch_embed2         │
                │  + pos_embed        │    │  + pos_embed  (×12帧)        │
                │  OmniMambaBlock ×4  │    │  OmniMambaBlock ×4           │
                │  [B, L, 128]        │    │  temporal GRU (逐空间位置)    │
                └──────────┬──────────┘    └──────────────┬───────────────┘
                           │                              │
                           └──────────┬───────────────────┘
                                      │
                          GatedCrossAttentionBlock
                          (query=PWV, key/value=Radar)
                          gate = Sigmoid(Linear([q, context]))
                          output = q + gate × (context − q)
                                      │
                              LayerNorm → Linear
                              → Pixel Shuffle → Bilinear Upsample
                              → PrecipitationEnhancementCNN
                              → Sigmoid
                                      │
                              输出 [B, 3, 66, 70]
                              (T+1h / T+2h / T+3h)
```

### 子模块说明

#### OmniMambaBlock
全向状态空间模块，支持两种后端（自动选择）：

| 后端 | 条件 | 实现方式 |
|------|------|----------|
| 真实 SSM | `mamba_ssm` 已安装（需 CUDA） | 水平扫描 + 垂直转置扫描，结果融合 |
| GRU 伪实现 | 无 `mamba_ssm`（CPU 可用） | 双向 GRU 水平扫描 + 垂直转置扫描，来自 ver_1.2 验证实现 |

两套后端接口完全一致，外部无感知。

#### GatedCrossAttentionBlock
门控交叉注意力融合模块。PWV tokens 作为 Query，Radar tokens 作为 Key/Value。门控初始化偏置为 -2（初始倾向保留 PWV 信息，逐步学习融合比例）。

#### PrecipitationEnhancementCNN
四路多尺度卷积细化（3×3 / 5×5 / 7×7 / dilated 3×3），通过 1×1 融合后加残差连接，强化降水边界细节。

---

## 损失函数（`losses.py`）

$$\mathcal{L} = 1.0 \cdot L_\text{MAE}^{\text{weighted}} + 0.05 \cdot L_\text{FFT} + 0.1 \cdot L_\text{SSIM}$$

| 分量 | 作用 |
|------|------|
| $L_\text{MAE}^{\text{weighted}}$ | 像素级 L1，强降水区域权重 $= 1 + 6 \times \text{target}$（最大7倍惩罚） |
| $L_\text{FFT}$ | 频域 MAE，约束降水纹理的空间频率结构 |
| $L_\text{SSIM}$ | 滑动窗口结构相似度损失，保持局部一致性 |

---

## 评估指标（`metrics.py`）

每个指标对 T+1h / T+2h / T+3h 分别计算，阈值 0.1（归一化像素值）：

| 指标 | 公式 | 含义 |
|------|------|------|
| MAE | $\frac{1}{N}\sum\|p-t\|$ | 像素绝对误差，越小越好 |
| CSI | $\frac{TP}{TP+FP+FN}$ | 命中率，气象标准，越大越好 |
| ETS | $\frac{TP-TP_\text{rand}}{TP+FP+FN-TP_\text{rand}}$ | 去随机命中后的 CSI，更抗稀疏性 |
| PSNR | $20\log_{10}(1/\sqrt{MSE})$ | 峰值信噪比，越大越好 |
| SSIM | 结构相似度 | 局部纹理相似度，越大越好 |

---

## 超参数配置（`config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `img_size` | 66 | 图像高度（像素） |
| `img_size_w` | 70 | 图像宽度（像素） |
| `patch_size` | 2 | Patch 边长 |
| `stride` | 2 | Patch 步长 |
| `dim` | 128 | Token 特征维度 |
| `depth` | 4 | OmniMambaBlock 层数 |
| `d_state` | 32 | SSM/GRU 隐状态维度 |
| `radar_seq_len` | 12 | 雷达历史帧数（12帧=66分钟） |
| `batch_size` | 8 | 批大小 |
| `epochs` | 1000 | 最大训练轮数 |
| `lr` | 1e-4 | AdamW 初始学习率 |
| `num_workers` | 0（Windows）/ 4（Linux） | DataLoader 工作进程数 |

---

## 安装依赖

```bash
pip install -r requirements.txt
```

GPU 加速（可选，启用真实 mamba_ssm）：
```bash
pip install mamba-ssm   # 需要 CUDA 环境
```

无 `mamba_ssm` 时自动回退到 GRU 实现，CPU 下可正常训练。

---

## 训练

> ⚠️ 团队约定：本地仅开发与测试，**不要在本地执行完整训练**。  
> 训练请走云端流程，见 [docs/CLOUD_TRAINING_WORKFLOW.md](docs/CLOUD_TRAINING_WORKFLOW.md)。

### 快速开始

```bash
python train.py --confirm-train
```

### 常用参数

```bash
# 指定数据路径和输出目录
python train.py --confirm-train --data-root ./data --results-dir ./results

# 自定义超参数
python train.py --confirm-train --epochs 200 --batch-size 4 --lr 5e-5

# 指定设备
python train.py --confirm-train --device cuda   # GPU
python train.py --confirm-train --device cpu    # CPU

# 固定随机种子
python train.py --confirm-train --seed 42
```

### 训练输出

训练结束后，`./results/` 目录下包含：

| 文件 | 内容 |
|------|------|
| `best_model.pth` | 验证集 loss 最优的模型权重 |
| `final_model.pth` | 训练结束时的最终权重 |
| `eval_report.json` | 测试集上三时间步的完整评估指标 |
| `loss_curve.png` | 训练/验证 loss 曲线 |
| `gate_curve.png` | 交叉注意力门控均值变化曲线 |
| `results_epoch_*.png` | 每5个 epoch 的预测可视化（逐次覆盖） |

### 训练策略

- **优化器**：AdamW，lr=1e-4
- **调度器**：CosineAnnealingWarmRestarts（T_0=50）
- **梯度裁剪**：max_norm=1.0
- **早停**：`patience = 4 × T_0`（默认 200 个 epoch 无改善触发）
- **评估频率**：每 5 个 epoch 打印详细指标并可视化

---

## 测试

```bash
python -m pytest tests/ -v
```

主要测试覆盖：配置加载、时间切分逻辑、模型前向传播 shape 验证、训练/验证循环冒烟测试、CLI 入口。

---

## 模块速查

| 模块 | 职责 |
|------|------|
| `config.py` | 所有超参数的统一数据类，修改此处即可调整实验配置 |
| `data_match.py` | 按时间戳对齐三路数据，构建雷达历史序列，过滤无效文件 |
| `dataset.py` | `TripleChannelDataset`：加载并转换图像，输出 `(pwv, radar_seq, targets)` |
| `splits.py` | 按时间范围将 SampleRecord 列表切分为 train/val/test |
| `model.py` | 完整网络：OmniMambaBlock、CrossAttentionMamba、RadarTemporalEncoder 等 |
| `losses.py` | SpectralStructuralWeightedLoss（MAE + FFT + SSIM + 强降水加权）|
| `metrics.py` | MAE、CSI、ETS、PSNR、SSIM |
| `train_loop.py` | `train_epoch` / `validate_epoch` / `train`（完整训练主循环） |
| `viz.py` | loss 曲线、门控曲线、预测对比图的绘制与保存 |
