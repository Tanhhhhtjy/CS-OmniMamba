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



