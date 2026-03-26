# 可视化脚本改进计划

**日期**: 2026-03-16
**目标**: 修复 visualize.py 的 5 个已知设计缺陷
**优先级**: P2（非阻塞，辅助工具优化）

---

## 改进范围总览

| # | 问题 | 优先级 | 改动范围 | 预计工作量 |
|---|------|--------|---------|-----------|
| 1 | 模型未真正通用化 | P3-低 | 中等（需设计 config.json） | 2-3h |
| 2 | Colorbar 实现可以更好 | P1-高 | 小（仅改 plot 逻辑） | 30min |
| 3 | 集成入口还没单测到 | P1-高 | 小（补 3 个测试） | 1h |
| 4 | 内存拼接可扩展性 | P2-中 | 中等（改累计逻辑） | 1h |
| 5 | 样本图只展示最新帧 | P3-低 | 大（需重新设计布局） | 2-3h |

**建议本轮改进**: #2 + #3 + #4（共 2.5h，高性价比）
**暂缓**: #1 + #5（需求不紧急，设计成本高）

---

## 改进 #1: 模型通用化（暂缓）

### 当前问题
`scripts/visualize.py:101` 硬编码：
```python
from src.model_convlstm import ConvLSTMModel as StubModel
```

### 改进方案
**Option A: 基于 config.json 的动态加载**
1. 在 `train.py` 保存 checkpoint 时，额外保存 `runs/{run_name}/config.json`：
   ```json
   {
     "model_type": "ConvLSTMModel",
     "model_module": "src.model_convlstm",
     "model_kwargs": {"hidden_dim": 64, "num_layers": 2}
   }
   ```
2. `_load_model()` 读取 config.json，用 `importlib` 动态加载：
   ```python
   import importlib
   cfg = json.loads((run_dir / "config.json").read_text())
   module = importlib.import_module(cfg["model_module"])
   ModelClass = getattr(module, cfg["model_type"])
   model = ModelClass(**cfg["model_kwargs"])
   ```

**Option B: 从 checkpoint 元数据读取**
- 在 `torch.save()` 时保存 `{"model_class": "ConvLSTMModel", ...}`
- 加载时解析元数据

### 改动文件
- `src/train.py` — 保存 config.json
- `scripts/visualize.py` — 改 `_load_model()`
- `tests/test_visualize.py` — 补测试

### 暂缓理由
1. **需求不紧急**：当前只有 ConvLSTM，未来加新模型时再改不迟
2. **设计成本高**：需确定 config.json schema，影响 train.py
3. **测试复杂度**：需 mock 多种模型类型

**建议**: 等实现第 2 个模型（如 SimVP）时再统一设计

---

## 改进 #2: 共享 Colorbar（本轮实施）✅

### 当前问题
每行各加 colorbar 会让图拥挤（虽然当前代码还没加 colorbar）

### 改进方案
在 `plot_samples_from_tensors()` 末尾，用 `fig.colorbar()` 添加全图共享 colorbar：
```python
# 在 fig.tight_layout() 之前
fig.colorbar(im_rain, ax=axes, label="Rain Intensity",
             orientation="horizontal", pad=0.05, fraction=0.02)
fig.colorbar(im_diff, ax=axes, label="Pred - True",
             orientation="horizontal", pad=0.08, fraction=0.02)
```

**明确规则**：
- 只给 **rain** 和 **diff** 两类图各放一个共享 colorbar
- **不给 radar / pwv 加 colorbar**（灰度图，无需额外标注）
- 如果 `constrained_layout` 和 colorbar 冲突，优先保证图不重叠

### 改动文件
- `scripts/visualize.py` — `plot_samples_from_tensors()` 末尾加 2 行
- `tests/test_visualize.py` — 无需改（PNG 生成测试已覆盖）

### 预期效果
- 图底部出现 2 条横向 colorbar（rain / diff）
- 节省空间，更美观

---

## 改进 #3: 补集成入口测试（本轮实施）✅

### 当前问题
未测试：
- `main(samples)` — CLI `--mode samples`
- `main(threshold)` — CLI `--mode threshold`
- `_load_model()` 失败路径

### 改进方案
在 `tests/test_visualize.py` 补 3 个测试，**使用 mock 策略保持轻量单元测试**：

**Test 1: main(samples)**
```python
def test_main_samples_mode(tmp_path, monkeypatch):
    """CLI --mode samples invokes plot_samples."""
    # Mock 策略：
    # 1. monkeypatch _load_model() 返回假模型（lambda x: torch.rand(...)）
    # 2. monkeypatch RainDataset / _filter_dataset_by_split() 返回极小 fake dataset
    # 3. monkeypatch sys.argv = ["visualize.py", "--run-name", str(run_dir), "--mode", "samples"]
    # 4. 调用 main()
    # 5. 断言 vis_last.png 存在
```

**Test 2: main(threshold)**
```python
def test_main_threshold_mode(tmp_path, monkeypatch):
    """CLI --mode threshold invokes plot_threshold."""
    # Mock 策略：类似 test_main_samples_mode
    # monkeypatch _load_model() + dataset，避免依赖真实数据
```

**Test 3: _load_model() 失败路径**
```python
def test_load_model_missing_checkpoint(tmp_path):
    """_load_model raises FileNotFoundError if checkpoint missing."""
    from scripts.visualize import _load_model
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _load_model(run_dir, epoch=999, device=torch.device("cpu"))
```

### 改动文件
- `tests/test_visualize.py` — 新增 3 个测试函数

### Mock 策略明确
- **避免依赖真实 checkpoint 和真实数据目录**
- 用 `monkeypatch` mock `_load_model()` 和 `RainDataset`
- 保持测试是轻量单元测试，而不是 slow integration test

---

## 改进 #4: 流式累计 TP/FP/FN（本轮实施）✅

### 当前问题
`plot_threshold_from_tensors()` 在 L280-281 把所有预测拼到内存：
```python
pred_cat = torch.cat(all_preds, dim=0)
target_cat = torch.cat(all_targets, dim=0)
```

### 改进方案
改为流式累计原始 TP/FP/FN，避免拼接大张量：

```python
def plot_threshold_from_tensors(
    run_dir: Path,
    all_preds: list[torch.Tensor],
    all_targets: list[torch.Tensor],
) -> None:
    from src.metrics import THRESH_STRONG, THRESH_WEAK

    thresholds = np.arange(0, 1.01, 0.02)
    n_thresh = len(thresholds)

    # 用并行数组存累计量，避免浮点数作为 dict key
    tp_accum = np.zeros(n_thresh)
    fp_accum = np.zeros(n_thresh)
    fn_accum = np.zeros(n_thresh)

    # 流式累计
    for pred, target in zip(all_preds, all_targets):
        for i, th in enumerate(thresholds):
            pred_bin = (pred > th).float()
            target_bin = (target > th).float()
            tp_accum[i] += (pred_bin * target_bin).sum().item()
            fp_accum[i] += (pred_bin * (1 - target_bin)).sum().item()
            fn_accum[i] += ((1 - pred_bin) * target_bin).sum().item()

    # 计算 CSI/POD/FAR，保持现有 NaN 语义
    csi_vals, pod_vals, far_vals = [], [], []
    for i in range(n_thresh):
        tp, fp, fn = tp_accum[i], fp_accum[i], fn_accum[i]
        denom_pod = tp + fn
        denom_far = tp + fp
        denom_csi = tp + fp + fn

        # 保持和 src.metrics.compute_csi_pod_far 一致的 NaN 语义：
        # - CSI/POD: 目标无雨时返回 NaN (denom_pod == 0)
        # - FAR: 模型完全不预测雨时返回 NaN (denom_far == 0)
        csi = tp / denom_csi if denom_pod > 0 else float("nan")
        pod = tp / denom_pod if denom_pod > 0 else float("nan")
        far = fp / denom_far if denom_far > 0 else float("nan")

        csi_vals.append(csi)
        pod_vals.append(pod)
        far_vals.append(far)

    # 后续绘图逻辑不变
    ...
```

### 改动文件
- `scripts/visualize.py` — 重写 `plot_threshold_from_tensors()` 前半部分
- `tests/test_visualize.py` — 现有测试应该仍能通过（输出不变）

### 关键设计决策
1. **用并行 numpy 数组存累计量**，不用浮点数作为 dict key
2. **保持现有 NaN 语义**，不用 `+1e-8` 硬抹掉 NaN
3. **与 `src.metrics.compute_csi_pod_far` 保持一致**的指标定义

### 优点
- 内存占用从 O(N×H×W) 降到 O(1)
- 支持任意大小的 val/test split
- 支持未来三步输出（只需在外层加 step 循环）
- 保持指标语义不变

---

## 改进 #5: 多帧 RADAR 展示（暂缓）

### 当前问题
样本图只展示 RADAR(t)，无法反映 10 帧时序动态

### 改进方案
**Option A: 三帧代表**
- 在当前 RADAR 列左侧加 3 个小窗：t-9 / t-5 / t
- 布局：`[t-9][t-5][t] [PWV] [Pred] [True] [Diff]`

**Option B: 胶片式序列**
- 第一行：10 个小 RADAR 帧（t-9…t）
- 第二行：PWV / Pred / True / Diff

### 改动文件
- `scripts/visualize.py` — 重新设计 `plot_samples_from_tensors()` 布局
- 需调整 `figsize` 和 subplot grid

### 暂缓理由
1. **设计成本高**：需重新规划布局，可能影响可读性
2. **需求不紧急**：当前展示已满足基本调试需求
3. **论文展示时再优化**：可根据实际需求定制

**建议**: 等正式实验结果出来，需要做论文图时再设计

---

## 实施计划

### 本轮改进（2026-03-16）
**范围**: #2 + #3 + #4
**预计时间**: 2.5h
**步骤**:
1. 改进 #2: 共享 colorbar（30min）
2. 改进 #4: 流式累计（1h）
3. 改进 #3: 补测试（1h）
4. 运行全量测试确认无回归

### 下一轮（待定）
**范围**: #1 + #5
**触发条件**:
- #1: 实现第 2 个模型时
- #5: 需要论文图时

---

## 验收标准

### 改进 #2
- [ ] `runs/*/vis/*.png` 底部出现 2 条 colorbar
- [ ] 现有测试仍通过

### 改进 #3
- [ ] 新增 3 个测试，全部通过
- [ ] 测试覆盖率提升（可用 `pytest --cov` 验证）

### 改进 #4
- [ ] `plot_threshold_from_tensors()` 不再调用 `torch.cat()`
- [ ] 输出的 `threshold_curve.png` 与改进前视觉一致
- [ ] 现有测试仍通过

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Colorbar 位置不美观 | 中 | 低 | 可调整 `pad`/`fraction` 参数 |
| 集成测试需要真实 checkpoint | 高 | 中 | 创建 minimal fake checkpoint 或标记为 slow test |
| 流式累计数值精度差异 | 低 | 低 | 用相对误差断言（`np.allclose`） |

---

## 总结

**推荐方案**: 本轮实施 #2 + #3 + #4，暂缓 #1 + #5

**理由**:
- 高性价比：2.5h 工作量，解决 3 个实际问题
- 低风险：改动局部，不影响核心训练流程
- 可扩展性：#4 为未来三步输出铺路
- 稳定性：#3 提升测试覆盖率

**暂缓项不影响当前工作**，可在需求明确时再优化。
