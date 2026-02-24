# 云端训练工作流（本地仅开发）

本项目采用：**本地写代码 + 提交 Git + 云端服务器训练**。

为避免资源占用和环境不一致，默认约定如下：

- 本地机器：仅做代码开发、静态检查、单元测试
- 云端服务器：执行训练与长时评估

---

## 1. 本地允许执行

- `python -m pytest tests/ -v`
- 小规模数据检查脚本（不触发完整训练）
- 文档与配置修改

## 2. 本地禁止执行

- `python train.py`
- 任何完整数据集训练命令
- 长时 GPU 占用任务

---

## 3. 推荐提交流程

1. 在本地完成改动
2. 运行测试：`python -m pytest tests/ -v`
3. 提交并推送：

```bash
git add .
git commit -m "<变更说明>"
git push
```

4. 在云端拉取最新代码并训练：

```bash
git pull
pip install -r requirements.txt
python train.py --confirm-train --data-root ./data --results-dir ./results
```

---

## 4. 云端训练最小记录要求

每次训练建议记录：

- 提交哈希（commit id）
- 关键超参数（`lr`、`batch_size`、`weight_decay`、`lr_scheduler_T0`）
- 开始时间/结束时间
- `best_epoch`、`best_val_loss`、`test ETS@0.04`

---

## 5. 与 `plan.md` 的配合

执行实验时，以 `plan.md` 中阶段七/七-A 为准：

- 先跑 Baseline
- 再按顺序做低风险参数改动（A-1/A-2/A-3）
- 每次只改一个变量组，便于归因

---

## 6. 数据审计流程（先审计再训练）

建议先执行数据质量审计，再启动长时训练：

```bash
mkdir -p results/data_audit/{figures,logs}

python -m scripts.data_audit_inventory \
	--data-root ./data \
	--output-dir ./results/data_audit \
	> ./results/data_audit/logs/inventory.log 2>&1

python -m scripts.data_audit_match_quality \
	--data-root ./data \
	--output-dir ./results/data_audit \
	> ./results/data_audit/logs/match_quality.log 2>&1

python -m scripts.data_audit_distribution \
	--data-root ./data \
	--output-dir ./results/data_audit \
	> ./results/data_audit/logs/distribution.log 2>&1

python -m scripts.data_audit_split_drift \
	--data-root ./data \
	--output-dir ./results/data_audit \
	--distribution-json ./results/data_audit/rain_bins_by_split.json \
	> ./results/data_audit/logs/split_drift.log 2>&1
```

关键输出：

- `results/data_audit/inventory.json`
- `results/data_audit/match_quality.json`
- `results/data_audit/rain_bins_by_split.json`
- `results/data_audit/split_drift_metrics.json`
- `results/data_audit/figures/*.png`
