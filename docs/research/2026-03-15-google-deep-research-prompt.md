# Google Deep Research 检索任务说明（给外部搜索代理）

请围绕以下任务进行**系统性文献调研**，输出一份适合科研选型决策的综述。

---

## 任务背景

我当前在做一个**降水临近预报**任务，数据形式如下：

- 输入：
  - `RADAR`：过去 10 帧，6 分钟间隔，张量形状约 `[B,10,1,66,70]`
  - `PWV`：当前 1 帧，张量形状约 `[B,1,1,66,70]`
- 输出：
  - `RAIN(t+1)`：未来 6 分钟的降水图，形状 `[B,1,66,70]`

当前已有基线：

- `ConvLSTM`
- `weighted MSE`
- 评估以 `CSI / POD / FAR / MSE / rain-only MAE` 为主
- 目标不仅是提升 CSI，还要**降低误报（false alarm / FAR）**

另有一个本地启发模型 `sota_plus.py`，其中出现了这些机制：

- gated cross-attention
- pseudo-Mamba（depthwise conv + bidirectional GRU）
- 多尺度增强 CNN
- AdamW + cosine warm restarts
- MSE + SSIM/PSNR 等验证指标

**注意：**
我不是要你推荐“能实现的随便什么技巧”，而是要你尽可能搜集**科研界已有论文支持、已被验证有效**的方法。

---

## 你的输出目标

请输出一份结构化调研报告，重点回答：

1. **有哪些模型结构优化方法可能提升这个任务的预测能力？**
2. **有哪些机制特别可能降低误报/虚警（FAR）？**
3. **哪些方法最适合当前这种“小分辨率 + 短序列 + 多模态（RADAR+PWV）+ 单步预测”的设置？**
4. **哪些方法更适合第二阶段/长期研究，而不适合第一阶段就上？**

---

## 重点检索方向

请至少覆盖以下 8 类：

### A. 时空预测主干模型

- ConvLSTM / ConvGRU
- PredRNN / PredRNN++ / MIM / E3D-LSTM
- TrajGRU
- UNet + ConvLSTM
- Earthformer / Transformer 类时空模型
- Mamba / SSM / selective state space 用于 precipitation nowcasting
- diffusion / cascaded / hybrid models

### B. 多模态融合机制

重点搜：

- radar + satellite
- radar + NWP
- radar + weather station
- radar + GNSS PWV / PWV / moisture field

需要关注：

- concat early fusion
- dual-encoder
- gated fusion
- cross-attention
- conditional fusion

### C. 降低误报 / FAR 的方法

重点搜：

- detection + regression 双头
- focal loss / class-balanced loss / asymmetric loss
- calibration / uncertainty / threshold tuning
- hard negative mining
- post-processing methods
- probability calibration for precipitation nowcasting

### D. 改善模糊预测 / 提高极端降水能力的方法

重点搜：

- perceptual loss
- SSIM / LPIPS / structure-aware loss
- Fourier-domain losses
- diffusion refinement
- cascaded deterministic + probabilistic models
- extreme precipitation focused objectives

### E. 物理约束 / physics-guided 方法

重点搜：

- advection-constrained models
- physically conditioned neural networks
- conservation-law-aware nowcasting
- gray-box / hybrid physics-AI nowcasting

### F. 优化器 / 训练策略

重点搜：

- Adam vs AdamW
- lr scheduler
- curriculum learning
- sample reweighting / oversampling extremes
- heavy-rain focused training

### G. 后处理与可部署增强

重点搜：

- zero-shot deblurring / postprocessing
- calibration after training
- blur correction
- morphology / connected-component filtering

### H. 2024–2026 最新工作

请特别关注近两年：

- ICML / NeurIPS / ICLR / CVPR / Nature / Nature Communications / PMLR / OpenReview / arXiv

---

## 来源要求

请尽量优先使用**一手来源**：

- 会议官网
- 期刊官网
- OpenReview
- PMLR
- NeurIPS proceedings
- arXiv
- Google Research official pages
- Nature / Science / AGU / AMS 等期刊官网

如果引用二手综述，请明确标注其为综述，不要把综述结论当作原始实验结论。

---

## 输出格式要求

请按以下结构输出：

### 1. Executive Summary

- 3–8 条结论
- 直接告诉我哪些方向最值得优先试

### 2. Method Taxonomy

按类别整理：

- 模型结构
- 融合机制
- 损失函数
- 降误报方法
- 物理约束方法
- 后处理方法

### 3. Paper-by-Paper Table

表格至少包含：

- 论文名
- 年份/ venue
- 核心机制
- 解决的痛点
- 是否报告对极端降水/高阈值 CSI/FAR 有帮助
- 实现复杂度（低/中/高）
- 对我当前任务的适配度（高/中/低）

### 4. False Alarm Reduction Focus

专门用一节回答：

- 哪些方法最可能降低 FAR
- 它们为什么有效
- 更适合训练阶段还是推理后处理阶段

### 5. Recommended Roadmap

请按下面 3 层给建议：

- **近期可落地**
- **中期值得做**
- **长期研究方向**

### 6. Gaps / Open Questions

例如：

- 是否已有 radar + PWV / GNSS-PWV 的直接 nowcasting 工作？
- 哪些方法在小图、短序列场景下仍然有效？
- 哪些方法虽然论文结果强，但实现成本过高？

---

## 特别提醒

请不要只给“大模型更强”这种泛泛结论。  
我更需要的是：

- **为什么它适合当前任务**
- **为什么它可能降低误报**
- **它是否真的有论文实验支持**
- **它在当前设置下是否值得优先实现**

如果某个方法只在大分辨率、长 lead time、多步生成里表现突出，但不适合我当前这种 `66×70 + 10帧 + 单步预测`，请明确指出。

