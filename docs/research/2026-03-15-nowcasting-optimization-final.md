# 降水临近预报模型优化最终调研结论

> 日期：2026-03-15  
> 任务：`RADAR(10帧) + PWV(1帧) -> RAIN(t+1)`  
> 当前设置：`66×70` 小分辨率、6 分钟步长、单步预测、重点关注 `CSI / POD / FAR`，尤其是**降低误报（FAR）**。  
> 本文档来源：综合 `docs/research/2026-03-15-nowcasting-optimization-survey.md`、Google Deep Research 返回结果，以及对关键论文真实性/任务适配性的二次核查。

---

## 1. 最终结论

### 1.1 可直接指导当前项目的结论

1. **`RADAR + PWV / GNSS-PWV` 多模态融合路线有直接文献支持，且与当前任务高度相关。**
   - 这一点不是推断，已有 IEEE TGRS 论文直接验证。
2. **在重构复杂融合机制前，应先做一次 `with PWV vs without PWV` 的 ablation。**
   - 目前“PWV 在现有 early fusion 中被浪费了”仍是合理假设，不是已被当前项目证明的结论。
3. **当前项目最值得优先尝试的不是“更大模型”，而是“更低成本、可直接验证 ROI 的改动”。**
   - 如：`PWV ablation`、`AdamW + scheduler`、`FACL`、残差预测、样本级平衡采样、EMA、形态学后处理
4. **仅靠 `weighted MSE` 不足以解决虚警和模糊问题。**
   - 频域损失、结构损失、后处理和概率校准更值得关注。
5. **在现阶段，最应优先考虑的主干升级是：**
   - `ConvLSTM -> SimVP / SmaAt-UNet` 一类低成本强 baseline
   - `ConvLSTM -> MambaCast / PredRNN / TrajGRU` 一类真正面向 radar nowcasting 的时空主干
6. **有些热门方法真实存在，但不适合当前任务：**
   - `NowcastNet`：大尺度、物理演化框架，当前小图单步任务优先级低
   - `TransMambaCNN`：任务时间尺度不匹配
   - `MCB-Net`：是 QPE，不是 nowcasting
   - `RainBalance`：更适合 1D GNSS 站点序列，不是 2D 雷达网格的即插即用方案
7. **“扩散模型会增加 FAR”没有直接文献证据。**
   - 这条只能视为工程担忧，不能作为最终报告结论。

---

## 2. 证据分级规则

- **Strong**
  - 论文真实存在
  - 任务类型与当前项目高度匹配
  - 结论能直接支持当前推荐/反对判断
- **Moderate**
  - 论文真实存在
  - 任务部分匹配，或证据来自 workshop / preprint / 相邻任务
  - 可作为候选方向，但不应写成定论
- **Weak**
  - 论文虽存在，但任务偏差较大
  - 或当前结论主要来自间接推断
  - 不适合直接写进最终推荐

---

## 3. 当前项目的最终方法清单

### 3.1 直接推荐（Phase 1 / 当前项目优先）

#### A. 多模态延迟融合 / 交叉注意力融合

- **结论**：强烈推荐
- **原因**
  - 当前任务天然是“高频动态雷达 + 慢变背景水汽场”的非对称多模态问题
  - 直接 early fusion 容易让 PWV 信号被淹没或与雷达动态纠缠
  - 更合理的方向是：
    - 先独立编码 `RADAR` 和 `PWV`
    - 再在中高层做 `gated fusion / delayed fusion / cross-attention`
- **对 FAR 的潜在帮助**
  - 可以让 PWV 充当“物理条件掩膜”
  - 抑制“有一点雷达结构、但无足够水汽条件”的伪降水触发
- **建议优先实现形式**
  - 可优先尝试 `gated fusion`
  - 若做 attention 版，可尝试 `PWV -> Query`、`RADAR 特征 -> Key/Value`
  - 但这里的**具体实现形式**属于工程设计，不是文献直接定论
- **证据强度**：
  - **融合方向**：`Strong`
  - **具体 cross-attention 设计**：`Moderate`

#### B. PWV 模态消融（Ablation）先于复杂融合重构

- **结论**：必须先做
- **原因**
  - 目前尚未证明 `PWV` 在当前项目中提供了显著边际信息
  - 如果 `PWV` 对当前任务帮助很弱，那么优先投入复杂融合机制的收益会很可疑
- **建议实验**
  - `RADAR only`
  - `RADAR + PWV(current early fusion)`
  - `RADAR + PWV(better fusion)` 仅在前两者差异明确后再做
- **证据强度**：`Moderate`
  - 多模态 nowcasting 文献普遍做 ablation；对当前项目而言，这是一条强工程约束

#### C. 形态学后处理（Morphological Post-processing）

- **结论**：强烈推荐，作为低成本防线
- **原因**
  - 不需要改训练
  - 直接在推理后对弱小散点、毛刺和虚假连接桥进行清理
- **建议形式**
  - 阈值化后：
    - `erosion`
    - `opening`
    - 小连通域过滤
  - 若后续输出为连续雨图，可先阈值成弱降水 mask 再处理
- **对 FAR 的潜在帮助**
  - 对“孤立虚警”“桥接型虚警”最直接
- **证据强度**：`Moderate`
  - 方法本身工程价值高，但当前可核查的一手证据还不如雷达/PWV 融合和 AANet 那样扎实

#### D. FACL：优先尝试的 loss 升级

- **结论**：强烈推荐
- **原因**
  - `Fourier Amplitude and Correlation Loss` 直接针对 `L2/MSE` 导致的模糊问题
  - 不要求你先换主干，也不要求新增复杂后处理
  - 已在 precipitation nowcasting 场景验证，且代码开源
- **对 FAR 的潜在帮助**
  - 不是直接优化 FAR
  - 但可通过减少广域模糊晕染，间接改善 `CSI/FSS` 并抑制“均值回归型虚警”
- **证据强度**
  - **频域损失替代 `MSE/L2` 的方向**：`Strong`
  - **FACL 这一具体实现对当前 `66×70` 单步多模态任务的直接证据**：`Moderate`

#### E. 残差/增量预测（Residual Prediction）

- **结论**：强烈推荐
- **核心形式**
  - 预测 `ΔRAIN = RAIN(t+1) - RAIN(t)`
  - 推理时再加回 `RAIN(t)`，即以 persistence 为基线学习残差
- **为什么重要**
  - 当前强基线本来就是 persistence
  - 让模型默认输出“无变化”比直接回归绝对场更符合短时单步任务
  - 要产生误报，模型必须主动在当前无雨区域预测正增量，门槛更高
- **重要边界**
  - 这里的增量应相对 `RAIN(t)`，不是相对 `RADAR(t)`
  - `RADAR` 与 `RAIN` 不是同一物理量，不能直接做物理上不自洽的差分
- **证据强度**：`Moderate-Strong`

#### F. 样本级类别平衡采样（Class-Balanced Sampling）

- **结论**：强烈推荐
- **原因**
  - 当前项目已有像素级加权，但没有样本级平衡
  - 大量晴空/近乎无雨样本可能让模型学到“输出无雨最安全”的捷径
- **建议形式**
  - 用 `WeightedRandomSampler`
  - 按每个样本的雨区像素占比或“是否含有效降水”加权
- **注意**
  - 训练可重采样
  - 验证/测试必须保持原始分布
  - 要监控是否出现 `POD↑, FAR↑` 的副作用
- **证据强度**：`Moderate`

#### G. EMA 模型平均

- **结论**：推荐
- **原因**
  - 成本极低
  - 对泛化和阈值稳定性通常有帮助
  - 与 `AdamW + scheduler` 不冲突，可并行使用
- **建议形式**
  - 使用 EMA 权重做验证/推理
- **证据强度**：`Moderate`

#### H. 用真正适配 radar nowcasting 的主干替代裸 ConvLSTM

- **结论**：推荐，但优先级排在“融合优化”之后
- **排序说明**
  - 下面的候选顺序主要按**当前任务的成本修正 ROI / 落地难度**排序
  - 不是单纯按文献证据强弱排序
- **候选顺序**
  1. `SimVP`
  2. `SmaAt-UNet`
  3. `MambaCast / PredRNN / TrajGRU`
  4. `Earthformer`
- **原因**
  - 当前 ConvLSTM 已经训通，但现有结果仍明显落后于强基线
    - 当前可见 run 中 `val_csi_weak ≈ 0.087`
    - `Persistence baseline csi_weak ≈ 0.928`
  - 这说明主干仍有明显提升空间
  - `MambaCast` 这类专为雷达序列设计的状态空间主干更值得关注
- **证据强度**
  - `SimVP`：`Moderate`
  - `SmaAt-UNet`：`Moderate`
  - `MambaCast`：`Moderate-Strong`
  - `PredRNN / TrajGRU`：`Strong`
  - `Earthformer`：`Moderate`

---

### 3.2 候选方向（Phase 2 / 中期研发）

#### I. Dice Loss + 结构损失（SSIM / gradient / perceptual）

- **结论**：值得尝试，但应作为候选，不宜写成已被当前任务充分证明
- **原因**
  - `weighted MSE` 容易导致：
    - 回归到均值
    - 预测模糊
    - 雨区边界外扩，间接推高 FAR
  - Dice 类损失对稀疏事件更友好
  - SSIM / Fourier / perceptual constraint 更有助于保持结构清晰度
- **边界**
  - Dice 的强证据更多来自“降水纠偏/分割型任务”
  - 对你这个 nowcasting 任务，结论是“合理迁移”，不是“已被直接证实”
- **建议形式**
  - 优先尝试：
    - `weighted MSE + λ * SSIM`
    - 或 `weighted MSE + λ * gradient-aware loss`
  - 第二优先：
    - 双头/分割头后再引入 `Dice / BCE / CE`
- **证据强度**：`Moderate`

#### J. 学习型降水掩膜（Learned Precipitation Mask）

- **结论**：保留为 Phase 2 候选
- **机制**
  - 用一个 `sigmoid mask head` 学“哪里会下雨”
  - 回归头学“下多大”
  - 最终输出由 mask 门控回归图
- **优点**
  - 比完全独立的双头更紧耦合
  - 能直接在无雨区压低回归输出，抑制虚警
- **证据强度**：`Moderate`

#### K. WADEPre 风格的频域解耦

- **结论**：是非常有吸引力的中期方向
- **原因**
  - 频域/小波解耦能直接针对“平流背景”和“局地对流细节”分开建模
  - 理论上很适合解决单纯回归造成的模糊
- **边界**
  - 目前核心证据来自预印本
  - 实现成本明显高于改 loss 或改融合
- **证据强度**：`Moderate`

#### L. 概率校准 / ETCE 路线

- **结论**：值得保留为中期后处理方向
- **原因**
  - 一旦模型输出可解释为概率或弱阈值置信度，校准可以直接影响 FAR
  - 在阈值型评价指标下，校准往往比纯改主干更有性价比
- **边界**
  - 当前论文证据来自 workshop / arXiv 路线
  - 对你当前 deterministic regression pipeline 需要中间改造
- **证据强度**：`Moderate`

#### M. 双头 rain / no-rain + intensity 回归

- **结论**：保留为 Phase 2 候选，不作为当前主线
- **原因**
  - 当前主指标本质上包含检测属性（`CSI / POD / FAR`）
  - 从任务形式看，把 `rain/no-rain` 作为语义分割头并非不合理
  - 但当前文献对其在你这个设置下的直接优势证据仍不足
- **推荐定位**
  - 作为 Phase 2 候选
  - 不宜在当前阶段优先于 `FACL / morphology / ablation / better fusion`
- **证据强度**：`Weak`

#### N. 非对称损失 / 可微 CSI 类损失

- **结论**：保留为 Phase 2 候选
- **原因**
  - 当前任务核心目标就是控制 `CSI / FAR`
  - 非对称损失或可微分 CSI/Jaccard 类目标，理论上更贴近最终评价指标
- **注意**
  - 这类 loss 可能更难优化
  - 建议在 `FACL / SSIM / better sampling` 稳定后再尝试
- **证据强度**：`Moderate`

#### O. PostCast / 去模糊后处理

- **结论**：可保留为 Phase 2 候选，但优先级不高
- **原因**
  - 它确实有助于减轻模糊
  - 但对当前任务而言，关键问题不是“算不动”，而是：
    - 需要额外训练/维护去模糊模型
    - 相比 `FACL / morphology / better fusion`，其投入产出比未必更优
- **重要边界**
  - 不能说它“会增加 FAR”
  - 正确表述应是：
    - **没有证据证明它会增加 FAR**
    - 但对当前任务未必是最具性价比的第一步
- **证据强度**：`Moderate`

---

### 3.3 不建议当前阶段投入

#### P. Focal Loss 作为当前主线

- **结论**：不建议作为优先方向
- **原因**
  - 已检索到的降水相关研究中，对它的收益评价偏弱
  - 一些证据甚至表明它可能伴随更高虚警
- **注意**
  - 更稳妥的最终表述不是“已被证伪”
  - 而是：
    - **当前任务中优先级低，不推荐先试**
- **证据强度**：`Moderate`（against priority, not against existence）

#### Q. NowcastNet 级别物理演化框架

- **结论**：不建议当前阶段投入
- **原因**
  - 任务尺度不匹配
  - 当前图像太小、lead time 太短
  - 工程复杂度过高
- **边界**
  - 不能说“论文证明它在小图上失效”
  - 正确写法应是：
    - **从任务尺度与工程投入产出比看，当前不适合**
- **证据强度**：`Moderate`

#### R. TransMambaCNN、MCB-Net、RainBalance（当前直接迁移）

- **结论**：不建议直接引入当前主线
- **原因**
  - `TransMambaCNN`：时间尺度和任务类型不匹配
  - `MCB-Net`：QPE，不是 nowcasting
  - `RainBalance`：1D GNSS 站点序列，不是 2D 雷达网格
- **证据强度**：`Strong`

---

## 4. 面向当前项目的研发路线图

### Phase 1：近期可落地（建议优先顺序）

1. **先做 `with PWV vs without PWV` ablation**
2. **把 `Adam -> AdamW`，并加 scheduler**
3. **尝试 `FACL`**
4. **加入残差/增量预测（相对 `RAIN(t)`）**
5. **加入样本级平衡采样**
6. **引入 EMA**
7. **引入推理后形态学过滤**
8. **若 ablation 证明 PWV 有价值，再做 delayed / gated fusion**
9. **用低成本强 baseline 替换或补充 ConvLSTM**
   - `SimVP / SmaAt-UNet / PredRNN / TrajGRU / MambaCast`

### Phase 2：中期值得做

1. `Dice + SSIM / gradient-aware loss`
2. 学习型降水掩膜
3. `WADEPre` 风格频域解耦
4. 概率校准 / ETCE
5. 双头 `rain/no-rain + intensity`
6. 非对称 / 可微 CSI 类损失
7. `PostCast` / 去模糊后处理
8. 更系统的 FAR 定向后处理

### Phase 3：长期研究方向

1. `NowcastNet` 类 physics-guided 路线
2. 更强概率模型 / ensemble 方案

---

## 5. 对当前项目最值得优先试的 6 个方向

### 1. `with PWV vs without PWV` ablation

- **推荐级别**：最高
- **为什么现在就该做**
  - 它决定了后续是否值得投入复杂融合机制
  - 这是当前所有多模态改造的前置验证
  - 若不先做，后续所有“PWV 条件化融合”都建立在未验证前提上

### 2. AdamW + scheduler

- **推荐级别**：最高
- **为什么现在就该做**
  - 成本最低
  - 风险最低
  - 对训练稳定性和收敛速度通常有直接帮助

### 3. FACL

- **推荐级别**：最高
- **为什么现在就该做**
  - 直接针对当前最明确的问题：`MSE` 导致的模糊
  - 无需改网络结构
  - 与当前任务和数据形式高度兼容

### 4. 残差/增量预测（相对 `RAIN(t)`）

- **推荐级别**：最高
- **为什么现在就该做**
  - 与 persistence 基线天然一致
  - 对当前单步短时预测非常贴合
  - 以“无变化”为默认输出，天然抬高误报门槛

### 5. 形态学后处理

- **推荐级别**：最高
- **为什么现在就该做**
  - 零训练成本
  - 对 FAR 直接有效
  - 很适合作为 baseline enhancement

### 6. Radar + PWV 的 delayed / gated fusion

- **推荐级别**：高
- **为什么现在就该做**
  - 如果 ablation 证明 PWV 有价值，这将是最直接的主线增强
  - 但其具体实现（如 cross-attention）应视作工程候选，而不是文献定论

---

## 6. 对 FAR 最可能有帮助的 5 类方法

> 以下按**预期对 FAR 的帮助程度**排序，**不等于执行优先级**。  
> 实际研发顺序请以 §4 的 Phase 1 / Phase 2 路线为准。

按我当前整合后的判断，优先级如下：

1. **PWV 条件化融合 / 门控融合**
2. **残差/增量预测（相对 `RAIN(t)`）**
3. **形态学后处理**
4. **非对称 / 可微 CSI 类损失**
5. **概率校准**

---

## 7. 实验设计与评估注意事项

### 7.1 `PWV ablation` 的控制变量

为了让 `with PWV vs without PWV` 的结论可信，建议：

- 使用**相同训练/验证划分**
- 使用**相同 epoch 数**
- 使用**相同随机种子**
- 使用**相同优化器与学习率策略**
- 主要比较应在 **validation set** 上完成，而不是先看 test set

建议最小对比组：

1. `RADAR only`
2. `RADAR + PWV (current simple fusion)`
3. `RADAR + PWV (better fusion)`  
   仅在 1 vs 2 已证明 `PWV` 有边际价值后再做

**关于残差预测的额外说明：**

- 残差/增量预测应视为**独立实验轴**
- 即先比较：
  - `RADAR only` vs `RADAR + PWV`
- 再比较：
  - `absolute target`
  - `residual target = RAIN(t+1) - RAIN(t)`

也就是说：

- **不要**把“是否使用残差预测”和“是否使用 PWV”混成一个实验
- 否则很难判断收益到底来自：
  - 多模态信息
  - 还是 residual formulation 本身

### 7.2 形态学后处理的保守参数起点

当前图像只有 `66×70`，不宜一开始就使用强腐蚀。

建议从最保守配置开始：

- `1-pixel` 等级的弱腐蚀
- 或等价的小核 `3×3`
- 先尝试 `opening`
- 小连通域阈值从极低值开始（例如 1–3 像素量级）

目标是：

- 先移除孤立散点和细小桥接
- 避免误删真实弱降水区域

### 7.3 阈值敏感性分析

当前项目以 `CSI / POD / FAR` 为主，而这些指标对阈值非常敏感。

因此建议：

- 不只看单一 `THRESH_WEAK`
- 至少补做：
  - 多阈值 `CSI / POD / FAR`
  - 阈值邻域敏感性分析
- 条件允许时加入：
  - `FSS`（对轻微空间偏移更鲁棒）

### 7.4 数据增强的当前定位

数据增强值得提及，但**不建议作为当前主线**。

原因：

- 某些常规增强会破坏气象场物理合理性
- 尤其是强旋转、强混合类增强不一定合理

当前更稳妥的做法是：

- 仅保留**轻量几何增强**为候选
- 待主干、融合、loss、后处理稳定后再单独评估收益

### 7.5 关于主干替换的建议

`SimVP / SmaAt-UNet / PredRNN / TrajGRU / MambaCast` 依然是很值得做的结构升级，
但从当前 ROI 看，它们更适合作为：

- 在 `ablation + optimizer + loss + sampling + morphology` 稳定后再推进的下一波实验
- 或与当前 ConvLSTM 并行作为强 baseline 对照

---

## 8. 最终保留参考文献（已核实存在）

1. Liu et al.  
   `A Deep Learning-Based Precipitation Nowcasting Model Fusing GNSS-PWV and Radar Echo Observations`  
   *IEEE Transactions on Geoscience and Remote Sensing*, 2025  
   https://ieeexplore.ieee.org/document/10942428

2. Cai et al.  
   `Improving Nowcasting of Intense Convective Precipitation by Incorporating Dual-Polarization Radar Variables into Generative Adversarial Networks`  
   *Sensors*, 2024  
   https://www.mdpi.com/1424-8220/24/15/4895

3. Shi et al.  
   `Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting`  
   *NeurIPS*, 2015  
   https://proceedings.neurips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html

4. Wang/Jin et al.  
   `MambaCast`  
   *IEEE Geoscience and Remote Sensing Letters*, 2026  
   （书目信息可核查，最终引用时建议补 DOI/正式页）

5. Yan, C.-W., et al.  
   `Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting`  
   *NeurIPS*, 2024  
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/b54532b0e57eb963b19e00583376cda3-Abstract-Conference.html

6. Liu et al.  
   `WADEPre: A Wavelet-based Decomposition Model for Extreme Precipitation Nowcasting with Multi-Scale Learning`  
   arXiv, 2026  
   https://arxiv.org/abs/2602.02096

7. Gao, Z., et al.  
   `SimVP: Simpler Yet Better Video Prediction`  
   *CVPR*, 2022  
   https://openaccess.thecvf.com/content/CVPR2022/html/Gao_SimVP_Simpler_Yet_Better_Video_Prediction_CVPR_2022_paper.html

8. Trebing, K., et al.  
   `SmaAt-UNet: Precipitation Nowcasting Using a Small Attention-UNet Architecture`  
   *Pattern Recognition Letters*, 2021  
   https://www.sciencedirect.com/science/article/pii/S0167865521000556

9. Kurki et al.  
   `Probability calibration for precipitation nowcasting`  
   arXiv, 2025；NeurIPS 2025 Workshop  
   https://arxiv.org/abs/2510.00594

10. Zhang et al.  
   `RainBalance: Alleviating Dual Imbalance in GNSS-based Precipitation Nowcasting via Continuous Probability Modeling`  
   arXiv, 2026  
   https://arxiv.org/abs/2601.06137

11. Zhang et al.  
   `Skilful nowcasting of extreme precipitation with NowcastNet`  
   *Nature*, 2023  
   https://www.nature.com/articles/s41586-023-06184-4

12. Gong et al.  
   `PostCast: Leveraging the Breaking of Blurry Effect for Robust Precipitation Nowcasting`  
   *ICLR*, 2025  
   https://arxiv.org/abs/2410.05805

13. Zhang et al.  
   `TransMambaCNN`  
   *Remote Sensing*, 2025  
   https://www.mdpi.com/2072-4292/17/18/3200

14. `CPrecNet`  
    *IEEE Geoscience and Remote Sensing Letters*, 2025  
    （用于支撑短时窗 residual / increment prediction 路线；最终归档时建议补 DOI/正式页）

15. `ImbCRPF`  
    *Journal of Hydrology*, 2025  
    （用于支撑样本级类别平衡采样与不平衡处理；最终归档时建议补 DOI/正式页）

16. `Improving Short-Range Precipitation Forecast through a Deep Learning-Based Mask Approach`  
    *Advances in Atmospheric Sciences*, 2024  
    （用于支撑学习型降水掩膜；最终归档时建议补 DOI/正式页）

17. `RainNet2024` / Jaccard-loss 路线  
    *Geoscientific Model Development*, 2024  
    （用于支撑可微 CSI / Jaccard 类目标；最终归档时建议补准确题名与 DOI）

---

## 9. 最后一句话

对当前项目而言，**最值得优先投入的不是“更大更复杂的模型”，而是“先验证 PWV 是否有用，再用低成本高 ROI 的方法稳步提升”**。  
如果只选一条主线，我建议：

> **先做 `with PWV vs without PWV` ablation；若 PWV 确有边际价值，再做门控/延迟融合，同时配上 FACL 与低成本形态学后处理。**

这条路线与当前任务最匹配，也最符合证据强度与投入产出比。
