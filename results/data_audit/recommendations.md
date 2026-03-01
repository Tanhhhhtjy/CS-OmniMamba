# 数据与切分改进方案（可执行）

## A. 立刻执行（1~2天内）

### A1. 增加时间净空（purge gap）

- 规则：边界两侧至少留出
  - `gap_minutes >= history_minutes + future_minutes = 66 + 180 = 246`
- 实践建议：直接取 `6小时`（保守，且按整点易维护）
- 目标：将审计中的 `train_val_overlaps` 与 `val_test_overlaps` 压到 `0`

### A2. 调整验证集构成（从“单连续块”改为“多窗口拼接”）

- 当前验证集（2023-07-31~2023-08-20）强对流偏多，事件率过高。
- 建议采用 `3~4` 个不相邻时间窗构建 val，每个窗口 5~7 天，并在窗口间保留 purge。
- 目标约束：
  - `val heavy_event_ratio(>=16mm/h)` 调整到 train 的 `0.8~1.2x`
  - 即目标区间约 `10% ~ 15%`（以当前 train `12.45%` 为锚）

### A3. 切分验收阈值（上线门槛）

- `train_val_overlaps = 0`
- `abs(val_heavy_ratio - train_heavy_ratio) <= 0.03`
- `rain_bin_jsd(train,val) <= 0.01`（现状远低于该值，但保留硬约束）

## B. 训练阶段同步改造（2~4天）

### B1. 早停从单指标改为双轨监控

- 保留主优化：`val_loss`
- 新增门控：`ETS@0.04`（至少每 5 epoch）
- 触发策略：`val_loss_ema + min_delta`，并记录 ETS 不退化约束

### B2. 验证采样稳定化

- val DataLoader 固定顺序（已满足）
- 按“事件分层抽样”构建 `val_probe`（固定 512~1024 条）用于高频监控，完整 val 低频评估

## C. 中期方案（1~2周）

### C1. 引入 Rolling-Origin 评估

- 示例（含 purge）：
  - Fold1: train 到 7月中旬；val 7月下旬
  - Fold2: train 到 8月上旬；val 8月中旬
  - Fold3: train 到 8月中旬；val 8月下旬
- 用 3-fold 的均值/方差作为早停与模型比较依据，避免单窗口偶然性。

### C2. 样本权重与切分协同

- train 保持原始分布（避免过采样过强）
- 若需提升强降水召回，仅对 train 使用温和 `WeightedRandomSampler`，val/test 绝不重采样

## D. 代码改造建议（最小侵入）

1. `omnimamba/config.py`
   - 新增 `split_purge_gap_minutes: int = 360`
   - 支持 `val_windows: list[tuple[start,end]]`（可选）
2. `omnimamba/splits.py`
   - 增加“多窗口切分 + purge 过滤”的 `split_records_with_purge(...)`
3. `scripts/data_audit_split_drift.py`
   - 增加“建议切分配置评估模式”，自动输出候选方案评分

## E. 本次数据下的结论摘要

- 你的 `val_loss` 平台化更像是“验证集过难 + 单窗口偏态”问题，而不是模型完全学不动。
- 建议先改切分，再做超参微调；否则很容易把“数据偏移”误判为“模型欠拟合”。


正在优化工具选择...

我会先按仓库现有流程快速梳理：读取 plan.md、训练配置、划分逻辑和最近训练日志，定位 val_loss 长期不变的根因，再给出可直接落地的数据与切分改进方案。接下来先做只读排查，不改代码。

我已经完成第一轮文档和配置核对：当前代码里验证窗口已被拉长（到 8/20），weight_decay=5e-4、T0=50 也已调整。下一步我会读取 splits、train_loop 和实际审计结果文件，确认是否仍存在划分漂移或验证集“无雨主导”导致 val_loss 平台化。

我已确认训练循环本身没有明显“验证损失不更新”的代码级错误，下一步会直接读取已生成的审计 JSON 和训练日志，基于真实分布给出可量化的切分优化建议。

我拿到了关键审计结果：分布漂移很小，但 val 的强降水占比是 train 的约 2.5 倍，这非常可能造成 val_loss 平台或高波动。现在我再看审计脚本与数据匹配实现，确认边界泄漏统计是否可信，并给出具体可执行的划分方案参数。

关键信号已经很清楚：val 集强降水事件占比 31.55%，而 train/test 都约 12.5%，这是导致 val_loss 长期“不降或震荡”的首要嫌疑；另外边界泄漏统计 41 也对应了“未设置 purge gap”。我会把这些结论整理成两份审计结论文档，并给出可直接执行的切分改造方案（含参数建议）。

基于你仓库里的审计结果，我判断 val_loss 长期基本不变的主因是验证集分布失衡，不是数据读取或匹配故障。

关键数据

强降水事件率（>=16mm/h）：train 12.45%、val 31.55%、test 12.64%，val 明显偏“暴雨期”。
切分边界风险：history=66min、future=180min，当前审计有 train_val_overlaps=41、val_test_overlaps=41（说明缺少 purge gap）。
数据质量本身整体健康：图像可读性正常、雷达序列 padding 触发率仅 0.071%、锚点偏移 p99=0s。
训练集/切分改进方案（优先顺序）

先加 purge gap：边界净空至少 246 分钟，建议直接 6小时，把 overlap 压到 0。
把 val 从“单连续时间块”改成“多时间窗拼接”（34 个窗口），目标让 val 强降水率回到 train 的 `0.81.2x（约 10%~15%`）。
切分验收阈值：overlap=0、|val_heavy-train_heavy|<=0.03、rain_bin_jsd(train,val)<=0.01。
训练监控改成双轨：val_loss + ETS@0.04，早停建议用 val_loss_ema + min_delta，减少“高噪声窗口”误停。
我已整理好的结论文档

summary.md
recommendations.md
要不要我下一步直接帮你实现“带 purge gap + 多窗口分层切分”的代码（改 config.py 和 splits.py），并加一个自动评估候选切分的脚本？

