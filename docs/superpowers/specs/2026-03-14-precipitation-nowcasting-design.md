# 降水临近预报任务设计规格文档

> **版本**：v1.1
> **日期**：2026-03-14
> **数据集**：2025年5–8月河北气象多模态数据集（PWV / RADAR / RAIN）
> **状态**：已通过 brainstorming 评审，待实现

---

## 1. 任务概述

### 1.1 目标

利用过去 60 分钟的雷达反射率序列和当前时刻的大气可降水量背景场，预测未来 6 分钟（下一时刻）的地面降水分布图。

### 1.2 任务类型

**Seq→1 临近预报（Nowcasting）**

```
输入：RADAR(t-9) ... RADAR(t)   ← 10 帧，步长 6 分钟，共 60 分钟历史
      PWV(t)                    ← 当前时刻背景水汽场（单帧）
输出：RAIN(t+1)                 ← 未来 6 分钟降水分布（单帧）
```

### 1.3 任务性质说明

- **预测量**：连续灰度值（降水强度），非离散分类
- **核心挑战**：空间极稀疏（89.80% 像素无雨）、零膨胀分布、6 分钟步长下持续基准（Persistence）极强
- **本设计不涉及模型架构**：模型选择留作后续独立决策；本文档规定输入输出格式、数据管线、损失函数和评估协议

---

## 2. 输入输出规格

### 2.1 张量形状约定

全项目统一使用 `[B, T, C, H, W]` 五维约定，其中：
- `B`：batch size
- `T`：时间步数（单帧输入时 T=1）
- `C`：通道数（灰度图 C=1）
- `H`：图像高度（行数）= **66**
- `W`：图像宽度（列数）= **70**

> **注意**：原始 PNG 数组 shape 为 `(H, W) = (66, 70)`（行×列），即高=66、宽=70。PIL 的 `Image.size` 返回 `(width, height) = (70, 66)`，两者约定不同，代码中须注意区分。

### 2.2 输入张量

| 名称 | 形状 | 数据类型 | 说明 |
|------|------|---------|------|
| `radar` | `[B, 10, 1, 66, 70]` | float32 | t-9 至 t 共 10 帧，预处理下采样后单通道 |
| `pwv`   | `[B,  1, 1, 66, 70]` | float32 | 仅 t 时刻，单帧背景场 |

### 2.3 输出张量

| 名称 | 形状 | 数据类型 | 说明 |
|------|------|---------|------|
| `rain_pred` | `[B, 1, 1, 66, 70]` | float32 | t+1 时刻预测降水强度（**当前阶段**） |
| `rain_gt`   | `[B, 1, 1, 66, 70]` | float32 | t+1 时刻真值（**当前阶段**） |

> **多步扩展预留**：计划后续扩展至三步输出（t+1 / t+2 / t+3）。届时输出形状将变为：
> - `rain_pred`: `[B, 3, 1, 66, 70]`（或等价的 `[B, 3, 66, 70]`）
> - `rain_gt`:   `[B, 3, 1, 66, 70]`
>
> 扩展时需同步修改：`src/dataset.py`（增加 t+2/t+3 target 读取）、`src/loss.py`（对每步分别计算再求均值）、`src/metrics.py`（对每步分别统计）、`src/train.py`（squeeze 逻辑调整）。可视化脚本已预留自动适配接口，见 `docs/superpowers/specs/2026-03-15-visualization-spec.md`。

### 2.4 归一化方案（所有模态统一）

```python
# 统一变换：高值 = 强信号（正向编码）
def normalize(pixel: np.ndarray) -> np.ndarray:
    """pixel: uint8 array, return float32 in [0, 1]"""
    return (255.0 - pixel.astype(np.float32)) / 255.0

# RAIN：0 = 无雨，1 = 最强降水（原始255→0, 0→1）
# PWV ：0 = 低水汽（干），1 = 高水汽（湿）（原始255→0, 0→1）
# RADAR：0 = 无回波，1 = 最强回波（原始255→0, 0→1）
```

> **编码验证**：PWV 反转编码方向已通过 ERA5 TCWV 空间均值对比得到支持（Pearson r = −0.997），可作为反转编码假设的佐证，但不能作为像素级物理量的精确验证。

---

## 3. 数据管线

### 3.1 RADAR 预处理（离线完成）

原始 RADAR 为 700×660（宽×高）RGBA 图像，需在构建 Dataset 前完成以下操作：

```python
import PIL.Image
import numpy as np

def load_radar(path: str) -> np.ndarray:
    img = PIL.Image.open(path)
    arr = np.array(img)[:, :, 0]   # 仅取 R 通道（R=G=B，Alpha=255），shape=(660, 700)

    # PIL.resize 参数为 (width, height)，目标 width=70, height=66
    # 输出 np.array shape = (66, 70)，与 RAIN/PWV 一致
    # 使用 LANCZOS（等价于 area averaging）保留强回波核心，优于双线性
    downsampled = PIL.Image.fromarray(arr).resize(
        (70, 66), resample=PIL.Image.LANCZOS
    )
    return np.array(downsampled, dtype=np.uint8)  # shape: (66, 70)
```

> **下采样方法选择**：此处为精确 10× 缩放（700→70, 660→66）。LANCZOS（等价于 area pooling 的近似）比双线性更好地保留强回波的峰值，不会将回波核心过度平滑。

**推荐做法**：离线将所有 RADAR 图像预处理并保存为 66×70 npy 或灰度 png，避免每次训练重复下采样。

### 3.2 样本索引构建

样本 = 时间窗口 `(t-9, t-8, ..., t)` + 目标 `(t+1)`，共需 **11 个**连续时间步（10 输入 + 1 目标）。

**约束条件**：

1. **三模态文件交集**：对 PWV / RADAR / RAIN 三个目录各取文件名集合，仅保留三者均存在的时间戳
2. **不跨数据缺失边界**：通过检查窗口内相邻时间戳的实际差值（应为 360 秒）验证连续性

```python
from datetime import datetime

def build_sample_index(radar_dir, pwv_dir, rain_dir, T=10):
    """
    Returns list of start indices into valid_ts where:
      inputs  = valid_ts[i : i + T]       # T=10 frames: t-9 ... t
      target  = valid_ts[i + T]           # t+1
    Window requires T+1 = 11 consecutive timestamps.
    """
    radar_ts = set(parse_timestamps(radar_dir))
    pwv_ts   = set(parse_timestamps(pwv_dir))
    rain_ts  = set(parse_timestamps(rain_dir))
    valid_ts = sorted(radar_ts & pwv_ts & rain_ts)

    samples = []
    for i in range(len(valid_ts) - T):
        window = valid_ts[i : i + T + 1]   # 11 timestamps
        diffs = [
            (window[j+1] - window[j]).total_seconds()   # total_seconds() 避免跨天丢失天数
            for j in range(len(window) - 1)
        ]
        if all(d == 360.0 for d in diffs):   # 360 秒 = 6 分钟
            samples.append(i)
    return samples
```

> **`total_seconds()` vs `.seconds`**：`timedelta.seconds` 只返回 seconds 字段（不含天），跨天缺口（如 23:54 → 次日 00:06）会被错误地计算为负数或小值。`total_seconds()` 正确处理跨天情形，是这里的必须选择。

### 3.3 数据划分

```
训练集：2025-05-01 ~ 2025-07-31（3个月）
验证集：2025-08-01 ~ 2025-08-15（前半月）
测试集：2025-08-16 ~ 2025-08-31（后半月）
```

**已知严重缺失日（由交集过滤 + 连续性检查自动排除）**：

| 日期 | RADAR 帧数 / 240 | 说明 |
|------|----------------|------|
| 2025-07-01 | 168 | 严重缺失，窗口自然截断 |
| 2025-07-02 | 26 | 极度缺失 |
| 2025-07-03 | 27 | 极度缺失 |
| 2025-06-06 | 97 | 缺约半天，影响跨日窗口 |
| 2025-06-11 | 122 | 同上 |

---

## 4. 损失函数

### 4.1 基线损失：雨区加权 MSE

```python
EPS = 2.0 / 255.0   # 有雨阈值（对应原始灰度 < 253，即归一化后 > ~0.0078）
RAIN_WEIGHT = 10.0  # 雨区像素权重倍数（初始值，可调）

def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: float32, shape [B, 1, H, W], normalized to [0, 1]
    rain pixels  : target > EPS  → weight = RAIN_WEIGHT
    no-rain pixels: target <= EPS → weight = 1.0
    """
    rain_mask = (target > EPS).float()
    weight = 1.0 + (RAIN_WEIGHT - 1.0) * rain_mask   # no-rain=1, rain=10
    loss = (weight * (pred - target) ** 2).mean()
    return loss
```

**权重设计依据**：
- 像素级零膨胀约 89.80%（无雨） vs 10.20%（有雨），按像素占比估算（不考虑误差幅值分布）
- weight=10 使雨区贡献约 49% 的总 loss，近似平衡两类像素
- 模型不会退化为全零输出，同时保留无雨帧"抑制虚警"的监督信号

### 4.2 权重调参策略

使用验证集 CSI 和 FAR 作为调参依据：

| 现象 | 调整方向 |
|------|---------|
| CSI 低，POD 低（漏报多） | 增大 RAIN_WEIGHT（如 15, 20） |
| FAR 高（虚警多） | 减小 RAIN_WEIGHT（如 5, 7） |
| 两者均可接受 | 保持当前权重 |

### 4.3 模糊偏置的渐进改善路径

MSE 的均值偏置会导致预测图模糊，按以下顺序尝试（前一步不满足时推进）：

```
Step 0（基线）：weighted_mse
Step 1（如预测图明显模糊）：+ SSIM 项
    loss = weighted_mse + 0.1 * (1 - ssim(pred, target, data_range=1.0))
Step 2（如高强度雨核仍不清晰）：对 target 做幂次变换
    gamma = 0.5
    loss = weighted_mse(pred ** gamma, target ** gamma)
Step 3（如 CSI 提升遇瓶颈，分析确认边界模糊是主因）：
    升级为双头（分类头 Focal Loss + 回归头 masked MSE）
```

> 空间分辨率仅 66×70，模糊影响相对有限，**先观察基线结果再决定是否升级**。

---

## 5. 评估协议

### 5.1 降水阈值定义

| 级别 | 归一化阈值 | 原始像素值 | 物理含义 |
|------|-----------|----------|---------|
| 弱降水 | > 2/255 ≈ 0.0078 | < 253 | 可见降水信号 |
| 中强降水 | > 55/255 ≈ 0.216 | < 200 | 较强降水区域 |

> **注意**：以上为基于灰度分布的经验阈值，未经雨量站数据校准。建议后续通过实测数据对比来校准，不宜完全依赖灰度分布。

### 5.2 主要评估指标

| 指标 | 说明 | 计算阈值 |
|------|------|---------|
| **CSI（Critical Success Index）** | `TP / (TP + FP + FN)`，主指标 | 弱降水、中强降水各算一次 |
| **POD（Probability of Detection）** | `TP / (TP + FN)`，漏报分析 | 同上 |
| **FAR（False Alarm Ratio）** | `FP / (TP + FP)`，虚警分析 | 同上 |
| **MSE** | 像素级均方误差（全图） | — |
| **MAE（rain only）** | 仅在有雨区域计算 MAE | target > EPS |
| **SSIM** *(Phase 2)* | 全图结构相似性，延期至有可比较模型输出后实现 | — |
| **FSS（Fractions Skill Score）** *(Phase 2)* | 邻域技巧分（邻域半径建议 3×3, 5×5），延期至 Phase 2 | 弱降水 |

### 5.3 必须计算的基准线（Baseline）

| 基准 | 定义 | 说明 |
|------|------|------|
| **Persistence** | `RAIN_pred(t+1) = RAIN(t)` | 直接复制最近一帧；由于降水持续性强（7月实测中位持续时长约 294 分钟，来源：dataset_analysis.md），这是极强的强基线，**不是上界** |
| **Zero** | `RAIN_pred(t+1) = 0`（全无雨） | 平凡退化基准，像素准确率 ~90%，CSI = 0 |

**模型必须在 CSI 指标上明显超过 Persistence Baseline，否则无实际意义。**

### 5.4 分天气型评估

所有指标必须同时报告以下两个子集，以**样本窗口**为粒度分层（而非按日均值）：

| 子集 | 定义 | 说明 |
|------|------|------|
| **强对流样本** | 目标帧 `RAIN(t+1)` 中有雨像素覆盖率 > 20% | 降水信号充分，能体现模型真实能力 |
| **弱回波/无雨样本** | 目标帧中有雨像素覆盖率 < 1% | 接近 Persistence/Zero 的退化条件 |

> 按样本窗口分层比按日均 RADAR 覆盖率分层更精确：同一天内的短时强对流不会被日均值稀释。

---

## 6. 消融实验计划

| 实验编号 | 变体 | 目的 |
|---------|------|------|
| E0 | Persistence Baseline | 强基线参考 |
| E1 | **基线**：RADAR(t-9:t) + PWV(t) → RAIN(t+1)，weighted MSE | 主实验 |
| E2 | RADAR(t-9:t) only（去掉 PWV） | 验证 PWV 对模型的边际贡献 |
| E3 | RADAR(t-9:t) + PWV(t-8:t, stride=2)（5帧稀疏 PWV） | 验证 PWV 趋势信息是否有效 |
| E4 | E1 + SSIM 损失 | 验证模糊改善效果 |

> **E2 vs E1 / E3 vs E1 的判断标准**：不以固定数值（如 0.5%）作为拍板阈值，而是在多个随机种子（建议 ≥3）重复实验后，若 CSI 差异在实验方差范围内（可用 bootstrap 区间估计），则认为无显著收益，保留更简单的方案。

---

## 7. 已知风险与开放问题

### 7.1 已知风险

| 风险 | 影响 | 缓解策略 |
|------|------|---------|
| Persistence 基准极强 | 模型难以展示超额收益 | 按样本窗口分层评估，强对流样本单独报告 |
| MSE 模糊偏置 | 高强度雨核学不准 | 见 4.3 渐进路径 |
| RADAR 空间对应未验证 | 700×660 → 66×70 的地理对齐假设 | 暂以文件名时间戳对齐为代理；后续补充地理投影验证 |
| PWV 数据来源未知 | 无法确认是否含真实 6 分钟信息 | E2/E3 消融实验量化 PWV 贡献；即便仅用 Option C，结论也有效 |

### 7.2 开放问题（不阻塞实现，但需后续跟进）

1. **PWV 数据来源与生成算法**：影响对 E3 结果的解读
2. **降水阈值校准**：建议与雨量站数据对比验证灰度-降水量对应关系
3. **SSIM 实现选择**：推荐使用 `pytorch-msssim` 库，注意 `data_range=1.0` 参数设置
4. **RAIN/RADAR 空间像素对应**：当前假设 66×70 对应同一地理域，无投影元数据验证

---

## 8. 决策日志（关键选择的理由记录）

| 决策点 | 选择 | 否决的选项 | 理由 |
|--------|------|----------|------|
| RAIN 历史是否作为输入 | 否 | 是 | 任务定义简化：单步预测场景中 RAIN(t) 可用，但 RAIN 时序信息量低于 RADAR，且加入后会增加输入维度与训练复杂度；留作后续消融扩展 |
| 历史窗口长度 T | 10（60 分钟） | T=5（30分钟）/ T=20 | 行业常用，覆盖大多数对流发展时间尺度 |
| PWV 时序策略 | Option C（单帧） | Option A（T=10）/ Option B（稀疏序列） | PWV 连续帧高度相关（MAE ~0.34–0.63），单帧已能代表背景场；冗余帧浪费计算 |
| RADAR 分辨率对齐 | 预处理下采样 66×70 | 在网络内部学习型下采样 | 简单可控，不引入额外可学习参数；可离线缓存 |
| RADAR 下采样方法 | LANCZOS | 双线性（BILINEAR） | 10× 精确缩放场景下 LANCZOS 更好地保留强回波峰值，双线性会过度平滑回波核心 |
| 损失函数 | Weighted MSE（α） | Masked MSE（β）/ 双头（γ） | β 丢失虚警抑制信号；γ 过早锁定头部结构；α 权重可控，与模型解耦 |
| 无雨样本过滤 | 不过滤 | 过滤全部无雨帧 | 61.4% 有雨帧比例不低；无雨帧提供抑制虚警的监督信号 |

---

*本规格文档基于 brainstorming 会话中的迭代讨论制定，覆盖数据分析结论（dataset_analysis.md）和时序特性调查（temporal_analysis.md）。模型架构选择不在本文档范围内，待后续独立设计。*
