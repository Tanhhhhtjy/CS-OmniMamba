# 训练完整性改进计划（修订版）

**日期**: 2026-03-16
**版本**: v2（根据 review 修正）
**目标**: 补充训练脚本缺失的核心功能，确保正式训练后有完整的分析材料

---

## 问题分析

当前 `train.py` 存在以下关键缺陷：

| 问题 | 影响 | 修正后优先级 |
|------|------|------------|
| **缺少 best checkpoint** | 只有 last.pt，可能不是最优模型 | **P0** |
| **训练日志不完整** | metrics.json 只有 weak 指标，缺 strong/mae_rain | **P0** |
| **Baseline 结果未保存** | Persistence/Zero 只打印，难以对比 | **P0** |
| **缺少训练元数据** | 无 config.json，难以复现实验 | **P0** |
| **缺少 eval-only 模式** | 无法单独评估已有 checkpoint | **P0** |
| 缺少 resume 能力 | 训练中断无法继续 | P1 |
| 无自动可视化 | 需手动运行 visualize.py | P1 |
| test set 评估能力 | 当前只能评估 val | P2 |

---

## 优先级重新定义

### P0：阻塞正式训练分析（必须实现）
1. 保存 best checkpoint
2. 扩展训练日志字段（weak + strong + mae_rain）
3. 保存 baseline 结果到文件
4. 保存训练元数据（config.json）
5. 提供 eval-only 评估能力

### P1：提升可观测性和鲁棒性
6. 自动周期性可视化（每 N 个 epoch）
7. Resume 训练能力

### P2：可选优化
8. Test set 评估（仅在显式开关下，**不默认执行**）
9. 早停机制
10. TensorBoard 集成

---

## P0 改进详细方案

### 改进 #1: 保存 best checkpoint（P0）

**当前问题**：
- 只保存 `last.pt` 和周期性的 `epoch_NNN.pt`
- 训练结束后可能拿到的不是最优模型

**改进方案**：
- 在训练循环中跟踪 `best_val_csi_weak`
- 当 `val_csi_weak` 提升时保存 `best.pt`
- 在训练日志中记录 best epoch

**实现位置**：`src/train.py` 主循环

**伪代码**：
```python
best_csi = -float('inf')
best_epoch = 0

for epoch in range(1, args.epochs + 1):
    # ... training ...
    val_metrics = eval_epoch(...)
    csi = val_metrics['csi_weak']
    
    if csi > best_csi:
        best_csi = csi
        best_epoch = epoch
        torch.save(model.state_dict(), run_dir / "best.pt")
        print(f"New best: epoch {epoch}, CSI={csi:.4f}")
```

---

### 改进 #2: 扩展训练日志字段（P0）

**当前问题**：
- `metrics.json` 只记录 `val_csi_weak`, `val_far_weak`, `val_mse`
- 缺少 `pod_weak`, `csi_strong`, `pod_strong`, `far_strong`, `mae_rain`

**改进方案**：
一次性补全所有指标，避免二次返工

**实现位置**：`src/train.py:140-141`

**改动**：
```python
# Before
entry = {"epoch": epoch, "train_loss": train_loss,
         "val_csi_weak": csi, "val_far_weak": far, "val_mse": mse}

# After
m = val_metrics  # MetricsAccumulator.compute() 返回的完整字典
entry = {
    "epoch": epoch,
    "train_loss": train_loss,
    "val_csi_weak": m['csi_weak'],
    "val_pod_weak": m['pod_weak'],
    "val_far_weak": m['far_weak'],
    "val_csi_strong": m['csi_strong'],
    "val_pod_strong": m['pod_strong'],
    "val_far_strong": m['far_strong'],
    "val_mse": m['mse'],
    "val_mae_rain": m['mae_rain']
}
```

---

### 改进 #3: 保存 baseline 结果（P0）

**当前问题**：
- Persistence/Zero 结果只打印到终端
- 没有保存到文件

**改进方案**：
- 保存到 `runs/{run_name}/baselines.json`
- 格式支持多 split（val/test）

**输出格式**：
```json
{
  "val": {
    "persistence": {
      "csi_weak": 0.9276,
      "pod_weak": 0.95,
      "far_weak": 0.02,
      "csi_strong": 0.85,
      "pod_strong": 0.88,
      "far_strong": 0.03,
      "mse": 0.0001,
      "mae_rain": 0.005
    },
    "zero": {
      "csi_weak": 0.0,
      "pod_weak": 0.0,
      "far_weak": null,
      "csi_strong": 0.0,
      "pod_strong": 0.0,
      "far_strong": null,
      "mse": 0.05,
      "mae_rain": 0.15
    }
  }
}
```

**实现位置**：`src/train.py:173-177` 之后

---

### 改进 #4: 保存训练元数据（P0）

**当前问题**：
- 无 `config.json`，无法复现实验
- 看到 `runs/xxx/` 不知道用了什么参数

**改进方案**：
- 训练开始时保存 `runs/{run_name}/config.json`

**输出格式**：
```json
{
  "run_name": "convlstm_e1",
  "epochs": 100,
  "batch_size": 8,
  "lr": 0.001,
  "device": "cuda",
  "ckpt_every": 5,
  "model": "ConvLSTMModel",
  "loss": "weighted_mse_loss",
  "train_samples": 20950,
  "val_samples": 3589,
  "timestamp": "2026-03-18T10:00:00"
}
```

**实现位置**：`src/train.py` main() 开头，datasets 构建完成后

---

### 改进 #5: eval-only 模式（P0）

**当前问题**：
- 无法单独评估已有 checkpoint
- 要出 test 结果必须重跑整个训练

**改进方案**：
新增 CLI 参数：
- `--eval-only`：跳过训练，只评估
- `--ckpt`：指定要加载的 checkpoint 路径
- `--split`：指定评估的 split（val / test，默认 val）

**用法示例**：
```bash
# 评估 best checkpoint 在 val 上的性能
python src/train.py --eval-only --ckpt runs/convlstm_e1/best.pt --split val

# 最终报告：评估 test set（显式触发，不默认）
python src/train.py --eval-only --ckpt runs/convlstm_e1/best.pt --split test
```

**输出**：
- 打印完整指标（weak + strong + mae_rain）
- 保存到 `runs/{run_name}/eval_{split}.json`

**实现位置**：`src/train.py` main() 入口处，检测到 `--eval-only` 后走独立分支

---

## P1 改进详细方案

### 改进 #6: 自动周期性可视化（P1）

**当前问题**：
- 训练过程中无法观察预测质量
- 需要手动运行 `scripts/visualize.py`

**改进方案**：
- 每 `--vis-every` 个 epoch（默认与 `--ckpt-every` 相同）自动生成样本图
- 从 val_loader 取第一个 batch 的前 2 个样本
- 保存到 `runs/{run_name}/vis/epoch_{NNN}.png`
- 失败不阻断训练（try/except 包裹）

**关键约束**：
- `@torch.no_grad()`
- 只取 2 个样本，不遍历整个 val set
- 复用 `scripts/visualize.py` 中的 `plot_samples_from_tensors()`
- 输出文件名与现有约定对齐：`{run_dir}/vis/vis_epoch_{NNN}.png`（不是 `epoch_{NNN}.png`）
- `scripts/visualize.py` 无需改动，命名已符合

**新增 CLI 参数**：
- `--vis-every`：默认等于 `--ckpt-every`，设为 0 则禁用

---

### 改进 #7: Resume 训练能力（P1）

**当前问题**：
- 训练中断后无法继续
- 当前 checkpoint 只保存裸 `model.state_dict()`，恢复后优化器动量丢失，best 状态无法延续

**改进方案**：
升级 checkpoint 为结构化 payload，同时新增 `--resume` CLI 参数

**Checkpoint 格式升级**：
```python
# 新格式（best.pt / last.pt / epoch_NNN.pt 统一使用）
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "best_csi": best_csi,
    "best_epoch": best_epoch,
}

# 旧格式兼容策略：加载时检测 key
ckpt = torch.load(path)
if isinstance(ckpt, dict) and "model" in ckpt:
    model.load_state_dict(ckpt["model"])          # 新格式
else:
    model.load_state_dict(ckpt)                   # 旧格式（裸 state_dict）
```

**Resume 实现细节**：
- 加载 model + optimizer state_dict
- 从 checkpoint 的 `epoch` 字段获取已完成 epoch，从下一个开始
- 从 checkpoint 的 `best_csi` / `best_epoch` 恢复 best 状态
- 保持 run_dir 不变，继续追加 metrics.json

---

## P0 补充：JSON NaN 规范化

**问题**：
- `json.dumps({"a": float("nan")})` 产出 `{"a": NaN}`，不是合法 JSON
- 现有 `metrics.py` 在无雨/无预测时返回 `float("nan")`
- 所有落盘 JSON 都有此风险

**修复方案**：
在 `src/train.py` 中添加一个小工具函数，所有 JSON 写出前统一调用：

```python
def _sanitize_for_json(obj):
    """Recursively replace float NaN/Inf with None for valid JSON output."""
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj
```

适用范围：`metrics.json` / `baselines.json` / `eval_{split}.json` / `config.json`

---

## P0 补充：eval-only 的 run_dir 归属

**问题**：
- `run_dir` 完全来自 `--run-name`（train.py:96-103）
- eval-only 示例只传 `--ckpt`，run_dir 来源不明确

**决策：从 `ckpt.parent` 推导 run_dir**

理由：
- 用户体验更简单，不需要记住 run_name
- checkpoint 本身就在 run_dir 下，路径关系天然成立

**用法示例**（修正后）：
```bash
# run_dir 自动推导为 runs/convlstm_e1/
python src/train.py --eval-only --ckpt runs/convlstm_e1/best.pt --split val

# test set 评估（显式触发）
python src/train.py --eval-only --ckpt runs/convlstm_e1/best.pt --split test
```

**实现细节**：
- `--eval-only` 时，`run_dir = Path(args.ckpt).parent`
- `--run-name` 在 eval-only 模式下忽略（或打印警告）
- 输出到 `run_dir / f"eval_{split}.json"`

---

## P2 改进详细方案

### 改进 #8: Test set 评估（P2，显式开关）

**重要约束**：
- **不默认执行**，必须显式触发
- 通过 `--eval-only --split test` 触发（见改进 #5）
- 目的：保持 test set 的独立性，避免变成调参反馈

**输出**：
- `runs/{run_name}/eval_test.json`
- 格式与 `eval_val.json` 相同，NaN 规范化为 null

---

## 测试计划

### 必须有测试（P0/P1 改动）

| 测试项 | 文件 | 测试内容 |
|--------|------|---------|
| best.pt 保存逻辑 | `tests/test_train.py` | 模拟 3 epoch，验证 best.pt 在 CSI 提升时更新，不提升时不覆盖 |
| metrics.json 新字段 | `tests/test_train.py` | 验证所有 8 个字段都写入，NaN 序列化为 null |
| baselines.json 格式 | `tests/test_train.py` | 验证 `{"val": {"persistence": {...}, "zero": {...}}}` 结构，null 而非 NaN |
| config.json 写出 | `tests/test_train.py` | 验证训练开始时写出，包含 run_name/epochs/lr/model/timestamp |
| eval-only val | `tests/test_train.py` | monkeypatch 数据集，验证跳过训练，run_dir 从 ckpt.parent 推导 |
| eval-only test | `tests/test_train.py` | 验证 --split test 写出 eval_test.json |
| eval-only 缺 --ckpt | `tests/test_train.py` | 验证 CLI 校验：--eval-only 没有 --ckpt 时报错退出 |
| checkpoint 新格式 | `tests/test_train.py` | 验证保存的 best.pt 包含 model/optimizer/epoch/best_csi |
| checkpoint 旧格式兼容 | `tests/test_train.py` | 验证加载裸 state_dict 的旧格式 checkpoint 不报错 |
| resume 恢复状态 | `tests/test_train.py` | 验证 --resume 后 epoch 从正确位置继续，best_csi 正确恢复 |
| 自动可视化不阻断训练 | `tests/test_train.py` | monkeypatch plot_samples_from_tensors 抛异常，验证训练继续完成 |
| 自动可视化文件命名 | `tests/test_train.py` | 验证输出为 `vis/vis_epoch_{NNN}.png` |

### 测试策略
- 所有测试使用 `tmp_path` fixture，不依赖真实数据目录
- monkeypatch `RainDataset` 返回极小 fake dataset（10 个样本）
- monkeypatch `_filter_dataset_by_split` 直接返回 fake dataset
- resume 测试：先保存一个结构化 checkpoint，再调用 main() 验证续训行为
- 不依赖真实 checkpoint 文件（除 resume 测试外，均用 tmp_path 生成）

---

## 改动范围汇总（修正后）

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `src/train.py` | 修改 | best.pt、日志字段、config.json、eval-only、resume、自动可视化、NaN 规范化、结构化 checkpoint |
| `tests/test_train.py` | 新增 | 12 个测试函数（文件当前不存在） |
| `scripts/visualize.py` | 不改动 | 命名约定已符合（`vis_epoch_{NNN}.png`） |

---

## 执行顺序

1. 改进 #4（config.json）— 最简单，先做
2. NaN 规范化工具函数 — 后续所有 JSON 写出都依赖它
3. 改进 #1（best.pt + 结构化 checkpoint）— P0 核心
4. 改进 #2（日志字段）— P0 核心
5. 改进 #3（baselines.json）— P0 核心
6. 改进 #5（eval-only，run_dir 从 ckpt.parent 推导）— P0 核心
7. 补测试（12 个函数）
8. 改进 #6（自动可视化）— P1
9. 改进 #7（resume + 旧格式兼容）— P1
