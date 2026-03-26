# 降水临近预报模型优化调研（初版）

> 日期：2026-03-15  
> 目的：围绕当前项目（`RADAR(10帧)+PWV(1帧) -> RAIN(t+1)`）系统调研科研界已验证、可能适用的模型优化方法与降低误报方案。  
> 范围：参考本地 `D:\北斗极端降水预测(新)\sota_plus.py` 中出现的机制，并结合公开论文原文进行扩展。  
> 说明：本版聚焦“方法地图 + 适用性判断 + 下一步建议”；后续可结合 Google Deep Research 返回结果汇总为终版。

---

## 1. 先说结论

对当前项目最值得优先考虑的方向，不是简单“堆更大模型”，而是按以下优先级推进：

1. **先把当前 ConvLSTM 做强**
   - 训练更久
   - 做好雨区不平衡处理
   - 增强时空特征提取
   - 增加误报约束
2. **再引入更适合 nowcasting 的结构**
   - `PredRNN / TrajGRU / Earthformer / CasCast 风格级联框架`
3. **若重点是极端降水和减少模糊/漏报**
   - 引入 `感知约束 / Fourier loss / diffusion refinement / physics-guided evolution`
4. **若重点是降低误报（false alarm）**
   - 优先考虑 `双头检测-回归`、`概率校准`、`多模态物理约束融合`

对于你当前这个 **小分辨率（66×70）+ 短历史（10帧）+ 多模态（RADAR + PWV）+ 单步预测** 的任务，最现实的高收益路线是：

- **短期可落地**：ConvLSTM + 更好的 loss/训练策略 + 门控融合
- **中期强升级**：PredRNN / Earthformer / radar+PWV gated fusion
- **长期冲击 SOTA**：NowcastNet / CasCast / DiffCast / PreDiff 一类“物理+生成式/级联式”框架

---

## 2. 本地 `sota_plus.py` 给出的启发

本地参考实现里出现的关键思路有：

### 2.1 门控跨模态融合

- `GatedCrossAttentionBlock`
- 核心思想：`query` 模态与 `key/value` 模态先做注意力交互，再通过 `gate` 控制保留多少跨模态信息
- 启发：
  - 非常适合你当前的 `RADAR + PWV`
  - 可避免 `PWV` 这种慢变量在融合时“压制”或“污染”高频雷达动态

### 2.2 伪 Mamba / SSM 风格时序建模

- `BiMambaBlock_Pseudo`
- 实际上是 `depthwise conv + bidirectional GRU + gating`
- 启发：
  - 如果不想立即上真 Mamba，可以先用 `GRU/ConvGRU + 局部卷积 + gating`
  - 对短序列 nowcasting 也是可行的中间方案

### 2.3 多尺度增强 CNN

- `PrecipitationEnhancementCNN`
- 多分支卷积核和膨胀卷积
- 启发：
  - 可作为输出头/后处理头，强化雨核、边界和局地对流结构

### 2.4 训练侧经验

- `AdamW`
- `CosineAnnealingWarmRestarts`
- 额外可视化 `gate` 历史
- 验证指标里加入 `MAE / MAPE / PSNR / SSIM`

**结论**：`sota_plus.py` 最值得迁移的不是整套 patch-token 架构，而是：

- `门控融合`
- `多尺度增强头`
- `更好的优化器/学习率调度`
- `更丰富的结构与感知类指标`

---

## 3. 科研界已验证、对当前任务可能有价值的方法地图

下面按“解决什么问题”而不是按“论文时间线”来组织。

---

## 4. 提升时序建模能力的方法

### 4.1 ConvLSTM：当前基线，优点是稳

**代表论文**

- ConvLSTM（NeurIPS 2015）  
  https://proceedings.neurips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html

**核心价值**

- 把卷积和 LSTM 结合起来
- 直接为降水临近预报提出
- 对时空局部结构建模有效

**对当前项目的意义**

- 作为第一代真实模型完全合理
- 但它常见问题是：
  - 预测偏平滑
  - 长程运动建模有限
  - 对快速位移和形变不够强

**建议**

- 保留为基线
- 不建议停留在“裸 ConvLSTM”

---

### 4.2 PredRNN：增强记忆传递

**代表论文**

- PredRNN（NeurIPS 2017）  
  https://papers.nips.cc/paper/6689-predrnn-recurrent-neural-networks-for-predictive-learning-using-spatiotemporal-lstms

**核心思想**

- 引入 `Spatiotemporal LSTM`
- 让记忆不仅沿时间传播，也沿层间传播

**对当前任务的潜在收益**

- 比 ConvLSTM 更强的时空记忆
- 对连续回波演化通常更稳
- 在 radar nowcasting 语境里是非常经典的升级方向

**适配判断**

- **高适配**
- 如果你想在“仍保持 recurrent 范式”下升级，`PredRNN` 是最值得优先查的

---

### 4.3 TrajGRU：显式学习位移轨迹

**代表论文**

- Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model（NeurIPS 2017，TrajGRU）  
  https://proceedings.neurips.cc/paper/7145-deep-learning-for-precipitation-nowcasting-a-benchmark-and-a-new-model.pdf

**核心思想**

- 不再使用固定卷积邻域
- 学习动态采样轨迹，适应回波平移/旋转/变形

**为什么重要**

- 降水回波不是静态纹理，而是会移动、伸缩、扭曲
- TrajGRU 直接针对这一点设计

**适配判断**

- **很适合 radar 主导的 nowcasting**
- 如果你发现 ConvLSTM 主要输在“位置偏差”和“运动跟不上”，TrajGRU 很值得尝试

---

### 4.4 Earthformer：空间-时间注意力

**代表论文**

- Earthformer（NeurIPS 2022 / OpenReview）  
  https://openreview.net/forum?id=lzZstLVGVGW

**核心思想**

- 用 `Cuboid Attention` 在局部 3D 时空块上做注意力
- 再通过全局向量连通局部和全局信息

**已验证信息**

- 原文明确提到在真实降水 nowcasting benchmark 上达到 SOTA

**对当前项目的适配判断**

- **中高适配**
- 优点：
  - 比 recurrent 模型更擅长全局依赖
  - 容易融合多模态
- 风险：
  - 对你当前这种 66×70 小图，纯 Transformer 的优势未必完全释放
  - 实现复杂度明显高于 PredRNN/TrajGRU

---

## 5. 提升极端降水与高强度雨核预测的方法

### 5.1 物理约束 + 条件学习：NowcastNet

**代表论文**

- NowcastNet（Nature 2023）  
  https://www.nature.com/articles/s41586-023-06184-4

**核心思想**

- 把物理演化规律和深度学习统一进一个框架
- 强调对极端降水、对流系统和多尺度结构的建模

**原文关键信息**

- 强调：
  - 现有方法容易模糊、耗散、位置错
  - 需要把物理约束嵌入数据驱动模型

**对当前项目的启发**

- 你的任务是 `PWV + RADAR -> RAIN`
- 最自然的 physics-guided 路线包括：
  - 对雷达分支学习运动/平流
  - 对 PWV 分支施加“背景场/条件场”作用
  - 把二者通过条件机制耦合，而不是简单拼接

**适配判断**

- **高价值，但实现复杂**
- 更适合作为第二阶段目标，不适合第一版直接上

---

### 5.2 级联式框架：CasCast

**代表论文**

- CasCast（ICML 2024 / PMLR）  
  https://proceedings.mlr.press/v235/gong24a.html

**核心思想**

- 把“中尺度确定性结构”和“小尺度随机细节”拆开建模
- 用 deterministic + probabilistic cascaded pipeline

**为什么重要**

- 极端降水往往既需要大尺度位置对，又需要局地小尺度细节锐利
- 单一 deterministic 模型容易平滑
- 单一 generative 模型容易位置飘

**对当前项目的适配判断**

- **非常值得关注**
- 尤其适合未来你要冲击：
  - 极端降水
  - 细结构清晰度
  - FSS / 高阈值 CSI

---

### 5.3 扩散模型路线：PreDiff / DiffCast / STLDM / PostCast

**代表论文**

- PreDiff（NeurIPS 2023）  
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/f82ba6a6b981fbbecf5f2ee5de7db39c-Abstract-Conference.html
- DiffCast（CVPR 2024，arXiv 版本）  
  https://arxiv.org/abs/2312.06734
- PostCast（ICLR 2025）  
  https://openreview.net/forum?id=v2zcCDYMok
- STLDM（2025，OpenReview/arXiv）  
  https://openreview.net/forum?id=f4oJwXn3qg

**共同动机**

- precipitation nowcasting 是多峰、随机、易模糊的问题
- deterministic 回归常常产生“平均化未来”

**各自特征**

- `PreDiff`：latent diffusion + 显式知识对齐
- `DiffCast`：把系统分成全局确定性运动 + 局地随机残差
- `STLDM`：先做 deterministic forecast，再做 latent diffusion enhancement
- `PostCast`：把“去模糊”单独做成可泛化后处理，不要求主模型重新训练

**对当前项目的适配判断**

- **中长期高价值**
- 若当前目标是尽快获得一个可靠、可复现、可对比的模型，不应第一步就上 diffusion
- 但如果后面重点转向：
  - 极端降水
  - 锐利结构
  - 不确定性
  则 diffusion 系是最值得重点追踪的路线之一

---

## 6. 解决“模糊预测”问题的方法

### 6.1 感知约束 / 频域损失

**代表论文**

- Fourier Amplitude and Correlation Loss（NeurIPS 2024）  
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/b54532b0e57eb963b19e00583376cda3-Abstract-Conference.html
- Perceptually Constrained Precipitation Nowcasting Model（ICML 2025）  
  https://proceedings.mlr.press/v267/feng25h.html

**核心思想**

- 单纯 `MSE / weighted MSE` 容易导致模糊
- 频域约束可增强高频结构
- 感知/后验约束可在不完全牺牲像素精度的前提下提升结构真实性

**为什么与你相关**

- 你当前就已经发现：
  - Zero baseline 的 MSE 可能不差
  - 但真正重要的是 CSI、结构、雨核、误报

**适配判断**

- **高适配**
- 如果你不想立刻换掉 ConvLSTM 主体，最容易增益的可能正是：
  - `weighted MSE + 结构约束`
  - `weighted MSE + Fourier 类损失`

---

### 6.2 后处理去模糊

**代表论文**

- PostCast（ICLR 2025）  
  https://openreview.net/forum?id=v2zcCDYMok

**核心思想**

- 不一定非要把主模型变复杂
- 可以把 nowcast 模型输出的模糊视为“可逆 blur”
- 再用专门后处理去增强

**适配判断**

- **非常适合作为后期低侵入增强**
- 优点：
  - 不必推翻当前主干
  - 可以后挂在 ConvLSTM / PredRNN / Transformer 后面

---

## 7. 降低误报（false alarm）的主要方法

这是你特别关心的一类。

### 7.1 双头：检测 + 强度回归

**证据来源**

- 来自 nowcasting 研究的常见设计范式
- 也被多篇 precipitation 论文通过“分类优于回归处理极端事件”间接支持
- IMERG/GPM nowcasting 论文明确指出：
  - 对极端降水，classification + focal-loss 可能优于 regression + MSE  
  - 见：  
    https://arxiv.org/abs/2307.10843

**原理**

- 先学“会不会下雨 / 哪儿会下雨”
- 再学“下多大”

**为什么能减少误报**

- 把“是否触发降水”单独建模
- 可以用更偏向精确率/虚警控制的 loss 或阈值

**对你当前任务的建议**

- **这是最值得优先考虑的降误报路线之一**
- 比一开始就上复杂 diffusion 更务实

---

### 7.2 概率校准

**代表论文**

- Probability calibration for precipitation nowcasting（arXiv 2025）  
  https://arxiv.org/abs/2510.00594

**核心思想**

- 模型预测的概率往往不校准
- 即便排序对，阈值一切就会出现虚警偏多或偏少

**对当前项目的意义**

- 如果后面模型输出扩展为：
  - 概率雨图
  - 多阈值分类概率
  则 calibration 是降低误报非常关键的一步

**适配判断**

- 当前你还是 deterministic 单图回归
- 但一旦转双头/概率模型，**校准应被纳入必做项**

---

### 7.3 多模态物理约束融合

**代表论文/证据**

- SmaAt-fUsion / SmaAt-Krige-GNet（radar + station）  
  https://arxiv.org/abs/2502.16116
- Global MetNet / MetNet 类工作强调：
  额外气象变量有助于更稳的 precipitation prediction  
  https://research.google/pubs/metnet-a-neural-weather-model-for-precipitation-forecasting/

**对你的任务的具体含义**

- 你已经有 `PWV`
- `PWV` 本身就是一种“是否具备降水水汽背景条件”的物理变量
- 如果融合得好，它应当帮助减少：
  - “雷达有点回波但其实不该下雨”的虚警
  - “缺乏水汽背景还乱报雨”的误报

**建议**

- 不要只做简单拼接
- 优先尝试：
  - `gated fusion`
  - `cross-attention fusion`
  - `PWV-conditioned radar decoder`

---

### 7.4 阈值导向训练

**现象**

- 你的评估主指标是 `CSI / POD / FAR`
- 但训练目标是 `weighted MSE`
- 这两者存在天然错位

**可行路线**

- 增加 threshold-aware auxiliary loss
- 或加“雨区检测头”
- 或在 validation 上做校准阈值搜索

**结论**

- 若你的目标是“减少误报”，单靠 MSE 往往不够

---

## 8. 多模态融合：对你当前任务最关键的一类优化

你的任务不是纯 radar nowcasting，而是 `PWV + RADAR -> RAIN`。这意味着多模态融合质量会强烈影响上限。

### 8.1 早融合：最简单，但通常不是最优

- 直接 `concat` 后送入主干
- 优点：简单
- 缺点：
  - 无法区分“快变量（RADAR）”与“慢变量（PWV）”
  - 易让模态间互相干扰

### 8.2 门控融合：最推荐

可参考 `sota_plus.py` 的 `GatedCrossAttentionBlock`

**优点**

- 能学会“什么时候信雷达，什么时候信 PWV”
- 特别适合当前这个“一个高频、一个低频”的组合

### 8.3 注意力融合：适合第二阶段

- cross-attention
- region attention
- recall attention

**相关证据**

- RAP-Net 强调注意力与强降水区域建模  
  https://arxiv.org/abs/2110.01035

**建议**

- 短期：先做 gated fusion
- 中期：尝试 cross-attention / region attention

---

## 9. 训练与优化器层面的可用改进

### 9.1 AdamW + scheduler

**本地 `sota_plus.py` 的做法**

- `AdamW`
- `CosineAnnealingWarmRestarts`

**当前项目状态**

- 还是 `Adam`

**建议**

- 这是一个低风险、低成本、值得立即尝试的改动
- 尤其当你开始训练更复杂模型时，`AdamW` 往往更稳

### 9.2 curriculum / lead-time curriculum

虽然你现在是单步 `t+1`，但未来若做多步输出：

- 先学短 lead time
- 再逐渐扩展到长 lead time

对 nowcasting 很常见，也更稳定。

### 9.3 hard example mining / heavy-rain oversampling

对于极端降水和减少漏报/误报：

- 适当增加强降水样本权重
- 对强对流窗口 oversample

这类策略常常比“换一个更大模型”更划算。

---

## 10. 面向当前项目的推荐路线图

### 10.1 近期待做（最值得）

1. **把 ConvLSTM 做成更强 baseline**
   - Adam → AdamW
   - 加 scheduler
   - 训练更久
2. **加入更强的多模态融合**
   - `PWV` 不再只当伪帧拼接
   - 改为门控融合 / cross-attention 融合
3. **优化 loss**
   - `weighted MSE` 保留
   - 增加结构项（SSIM / Fourier loss / perceptual constraint 中选一）
4. **专门降低误报**
   - 加 detection head
   - 或做回归+分类双头

### 10.2 中期升级

1. `ConvLSTM -> PredRNN / TrajGRU`
2. `ConvRNN -> Earthformer`
3. `concat fusion -> gated cross-modal fusion`

### 10.3 长期冲击性能

1. `NowcastNet` 风格 physics-guided
2. `CasCast / DiffCast / PreDiff / STLDM` 风格生成式/级联式
3. 概率输出 + calibration

---

## 11. 对当前项目最可能有用的方法清单（按优先级）

### A 级：强烈建议优先尝试

1. **门控融合（PWV 条件化 / gated fusion）**
2. **双头：降水检测 + 强度回归**
3. **AdamW + scheduler**
4. **结构损失：Fourier / perceptual / SSIM 类**
5. **PredRNN 或 TrajGRU 替代裸 ConvLSTM**

### B 级：很有潜力，但实现成本更高

1. **Earthformer / Transformer 类时空注意力**
2. **CasCast 风格 deterministic + probabilistic 级联**
3. **PostCast 风格后处理去模糊**

### C 级：更偏中长期研究方向

1. **NowcastNet 风格 physics-guided evolution**
2. **PreDiff / DiffCast / STLDM / SRNDiff 等 diffusion 路线**
3. **概率校准 + 不确定性输出**

---

## 12. 我建议 Google Deep Research 重点补充的空白

我这一轮已经能确认的大方向是：

- recurrent、attention、diffusion、physics-guided、multimodal fusion 都有充分文献支持

但还需要进一步深挖的包括：

1. **是否已有“radar + PWV / GNSS-PWV”直接用于 nowcasting 的代表工作**
2. **哪些论文明确报告了 FAR/误报下降，而不仅是 CSI 上升**
3. **哪些 loss 在 extreme precipitation 下最稳**
4. **哪些方法在小分辨率、小样本、多模态设置下真正划算**
5. **2025–2026 最新工作里，哪些已经开始把 calibration / postprocessing / perception loss 系统化**

---

## 13. 参考来源（初版）

### 本地参考

- `D:\北斗极端降水预测(新)\sota_plus.py`

### 论文与官方页面

- ConvLSTM (NeurIPS 2015)  
  https://proceedings.neurips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html
- PredRNN (NeurIPS 2017)  
  https://papers.nips.cc/paper/6689-predrnn-recurrent-neural-networks-for-predictive-learning-using-spatiotemporal-lstms
- TrajGRU benchmark paper (NeurIPS 2017)  
  https://proceedings.neurips.cc/paper/7145-deep-learning-for-precipitation-nowcasting-a-benchmark-and-a-new-model.pdf
- DGMR (Nature 2021)  
  https://www.nature.com/articles/s41586-021-03854-z
- MetNet (Google Research)  
  https://research.google/pubs/metnet-a-neural-weather-model-for-precipitation-forecasting/
- Earthformer (OpenReview / NeurIPS 2022)  
  https://openreview.net/forum?id=lzZstLVGVGW
- NowcastNet (Nature 2023)  
  https://www.nature.com/articles/s41586-023-06184-4
- Global precipitation nowcasting with U-Net ConvLSTM + focal loss (arXiv 2023)  
  https://arxiv.org/abs/2307.10843
- PreDiff (NeurIPS 2023)  
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/f82ba6a6b981fbbecf5f2ee5de7db39c-Abstract-Conference.html
- DiffCast (CVPR 2024 / arXiv)  
  https://arxiv.org/abs/2312.06734
- CasCast (ICML 2024)  
  https://proceedings.mlr.press/v235/gong24a.html
- Probability calibration for precipitation nowcasting (arXiv 2025)  
  https://arxiv.org/abs/2510.00594
- SmaAt-fUsion / SmaAt-Krige-GNet (arXiv 2025)  
  https://arxiv.org/abs/2502.16116
- Fourier Amplitude and Correlation Loss (NeurIPS 2024)  
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/b54532b0e57eb963b19e00583376cda3-Abstract-Conference.html
- Perceptually Constrained Precipitation Nowcasting Model (ICML 2025)  
  https://proceedings.mlr.press/v267/feng25h.html
- PostCast (ICLR 2025)  
  https://openreview.net/forum?id=v2zcCDYMok
- STLDM (2025)  
  https://openreview.net/forum?id=f4oJwXn3qg

