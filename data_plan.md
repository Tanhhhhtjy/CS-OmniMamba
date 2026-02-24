# CS-OmniMamba 数据质量与分布综合审计计划

> 目标：对 `PWV/RADAR/RAIN` 三路数据进行尽可能全面的质量分析与分布诊断，形成可执行改进建议，支撑训练稳定性与泛化能力提升。  
> 约束：本地仅开发，数据扫描与统计在云端执行；**本计划不包含完整训练**。

---

## 0. 总体交付物

本计划完成后应产出以下结果（最少交付）：

1. `results/data_audit/summary.md`：结论总览与建议清单（按优先级）
2. `results/data_audit/quality_metrics.json`：核心质量指标（完整性、覆盖率、异常率）
3. `results/data_audit/split_diagnostics.json`：train/val/test 划分统计与漂移指标
4. `results/data_audit/figures/`：关键分布图与时间序列图
5. `results/data_audit/recommendations.md`：可实施改造方案（短期/中期/长期）

---

## 1. 分析维度（覆盖清单）

### 1.1 数据完整性与一致性
- 文件存在性：三路目录是否完整，是否含非图像文件
- 时间戳合法性：命名格式是否统一、是否可解析
- 三路对齐可用率：`PWV(t)` 能匹配 `RADAR(seq)` 与 `RAIN(t+1/2/3h)` 的比例
- 缺帧结构：
  - radar 时间序列历史帧不足比例
  - repeat padding 触发比例
  - `best_time` 与 `PWV(t)` 偏差分布（秒）

### 1.2 值域与图像质量
- 灰度值范围（min/max/percentiles）
- 饱和像素比例（0 与 255 占比）
- 常量图像比例（几乎全同值）
- 空白/损坏图像比例（读取失败、shape 异常）
- 模态间统计一致性（PWV/RADAR/RAIN 的均值与方差稳定性）

### 1.3 时间分布与时序连续性
- 各月份/各天/各小时样本量分布
- 连续缺口（>6min、>30min、>1h）计数与位置
- train/val/test 的时间覆盖长度、边界间隔
- 边界泄漏风险：历史窗口+未来标签是否跨集合边界

### 1.4 标签分布（降水强度）
- 全像素分布：无雨/毛毛雨/小雨/中雨/大雨/暴雨占比
- 样本级事件分布：
  - 每张图最大降水强度分布
  - 强降水事件样本占比（例如 >=16mm/h）
- 三个预测时距（T+1h/T+2h/T+3h）的分布差异

### 1.5 集合间分布偏移（Train vs Val vs Test）
- 像素强度分布偏移（KS/JSD/EMD）
- 事件频率偏移（强降水占比差异）
- 昼夜结构偏移（小时分布偏移）
- 季节/月份偏移

### 1.6 样本权重与学习难度
- 无雨主导程度（类别不平衡度）
- 有效样本密度（含降水像素样本占比）
- 验证集方差风险评估（窗口长度是否足以稳定早停）

---

## 2. 执行阶段与步骤

## 阶段 A：数据盘点与基础体检

### A1. 文件层扫描
- 扫描 `data/PWV`, `data/RADAR`, `data/RAIN`
- 统计：总文件数、可解析时间戳文件数、异常文件数、重复时间戳数
- 输出：`inventory.json`

### A2. 可读性与格式检查
- 随机抽样+全量轻检查：
  - 图像可打开率
  - shape 是否一致（目标 `66x70` 前后）
  - 灰度通道一致性
- 输出：`image_health.json`

### A3. 时间轴构建
- 构建三路时间轴，统计缺口和不连续段
- 输出：`timeline_stats.json` + `figures/timeline_coverage.png`

---

## 阶段 B：样本匹配与可用率分析（核心）

### B1. 使用 `match_samples` 全量匹配
- 记录总 `PWV` 数、可用样本数、过滤原因分解：
  - 无可用 radar
  - radar 偏差超 60min
  - 缺失任一目标 `t+1/2/3h`
- 输出：`match_summary.json`

### B2. Radar 序列质量诊断
- 统计 `radar_seq_len` 实际历史深度
- 统计 repeat padding 比例
- 统计相邻帧间隔分布（6/12/18min...）
- 输出：`radar_seq_quality.json` + `figures/radar_gap_hist.png`

### B3. 锚点偏移质量
- `|radar_anchor_time - pwv_time|` 分布
- 重点关注 >15min、>30min 比例
- 输出：`anchor_offset.json`

---

## 阶段 C：标签与特征分布分析

### C1. 值域反编码与雨强分箱
- 采用与 `scripts/eval_thresholds.py` 一致的反编码逻辑
- 统计各分箱占比（像素级 + 样本级）
- 输出：`rain_bins_global.json`

### C2. 分集合分布（train/val/test）
- 分别统计：
  - 无雨像素率
  - 强降水像素率
  - 强降水样本率
- 输出：`rain_bins_by_split.json` + `figures/rain_bins_by_split.png`

### C3. 多时距分布差异
- 对 T+1/T+2/T+3 分别建分布
- 检查随时距变远是否出现分布断崖
- 输出：`rain_bins_by_horizon.json`

---

## 阶段 D：切分合理性与漂移诊断

### D1. 当前切分可解释性
- 基于 `TrainingConfig` 复现 split
- 统计各集合时长、样本量、事件数
- 输出：`split_summary.json`

### D2. 边界泄漏风险检查
- 检查样本的历史窗口和未来标签是否跨分界
- 若跨界，统计比例与具体时间段
- 输出：`split_leakage_check.json`

### D3. 分布漂移量化
- 计算 train-val、train-test 在关键指标上的偏移：
  - 雨强分箱 JSD
  - 小时分布 JSD
  - 强降水事件率差
- 输出：`split_drift_metrics.json`

---

## 阶段 E：结论与建议输出

### E1. 问题分级
- P0（必须立即修复）：会造成训练失真/指标虚高
- P1（高优先级）：明显影响泛化或早停稳定
- P2（优化项）：提升上限但非阻塞

### E2. 建议清单（按投入-收益排序）
- 数据层：补齐缺失时段、修复异常样本、扩展验证窗口
- 切分层：加入 purge gap、事件分层切分、延长 val
- 训练层：早停判据调整（EMA + ETS）、采样策略（重雨 oversampling）

### E3. 形成执行路线
- 1周内可完成（低风险）
- 2~4周可完成（中等改造）
- 长期建设（跨季节/跨年数据）

---

## 3. 指标口径与阈值（建议）

### 3.1 关键质量指标（KQI）
- 匹配可用率 >= 85%
- 雷达锚点偏差 > 30min 的比例 <= 5%
- repeat padding 触发比例 <= 10%
- 验证集强降水样本率不低于训练集的 70%

### 3.2 划分合理性阈值
- train-val 雨强分箱 JSD <= 0.10（经验阈值）
- train-test 雨强分箱 JSD <= 0.15（经验阈值）
- 边界跨集合窗口泄漏比例 = 0

> 注：阈值用于预警与排序，不是绝对物理定律；最终以业务目标和可获取数据为准。

---

## 4. 产出图表清单

1. 三路时间覆盖总览图（按天）
2. train/val/test 样本量与事件量柱状图
3. 雨强分箱堆叠图（像素级、样本级）
4. radar 锚点偏移直方图
5. radar 序列间隔与 padding 触发率图
6. 小时分布对比图（train/val/test）
7. 漂移热力图（JSD/事件率差）

---

## 5. 执行方式（云端）

建议在云端新建审计目录：

```bash
mkdir -p results/data_audit/{figures,logs}
```

执行时原则：
- 优先复用现有逻辑（`omnimamba/data_match.py`, `scripts/eval_thresholds.py`）
- 统计脚本单独输出到 `results/data_audit/`
- 不启动完整训练，不覆盖训练结果目录

---

## 6. 计划里程碑

- M1（D+1）：完成 A/B 阶段（完整性与匹配质量）
- M2（D+2）：完成 C/D 阶段（分布与切分漂移）
- M3（D+3）：输出 E 阶段建议与实施优先级

---

## 7. 风险与兜底

- 若数据量过大导致全量统计耗时过长：先做 10% 时间分层抽样，再做全量补充
- 若反编码参数（`rain_max`, log/linear）不确定：并行输出两套统计，避免误判
- 若发现跨集合泄漏：优先修复切分配置，再继续模型对比实验

---

## 8. 下一步（待你确认后执行）

确认后我将按本计划进入实施阶段，先开发并提交以下审计脚本：

1. `scripts/data_audit_inventory.py`
2. `scripts/data_audit_match_quality.py`
3. `scripts/data_audit_distribution.py`
4. `scripts/data_audit_split_drift.py`

并在 `docs/CLOUD_TRAINING_WORKFLOW.md` 增加“数据审计流程”一节，确保团队可复用。
