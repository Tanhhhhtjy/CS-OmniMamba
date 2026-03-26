# 实验迭代日志

> 记录每轮实验的配置、指标、曲线观察和结论。
> 更新规则：每次服务器跑完一组实验后，在此追加对应章节，并更新"当前最优"快照。

---

## Baseline 参考

| 指标 | Persistence | Zero |
|------|-------------|------|
| csi_weak | 0.9276 | 0.0000 |
| pod_weak | 0.9619 | 0.0000 |
| far_weak | 0.0370 | null |
| csi_strong | 0.8563 | 0.0000 |
| mse | 0.0000292 | 0.001903 |

---

## E1 — Baseline（weighted_mse + Adam）

**日期**: 2026-03-19
**run_name**: `convlstm_e1`
**配置**:
- loss: weighted_mse（RAIN_WEIGHT=10）
- optimizer: Adam, lr=1e-3
- scheduler: none
- epochs: 100, batch_size: 16

**指标（best checkpoint，eval_val.json）**:

| csi_weak | pod_weak | far_weak | csi_strong | mse |
|----------|----------|----------|------------|-----|
| 0.3502 | 0.9119 | 0.6375 | 0.0774 | 0.001512 |

**训练曲线**: metrics.json 丢失，仅保留 best checkpoint 评估结果。

**结论**:
- 高 POD（0.91）+ 高 FAR（0.64）= 模型"撒网式"预测雨
- 根本原因：RAIN_WEIGHT=10 的非对称 loss 让模型宁可误报不能漏报
- 需要引入惩罚误报的机制（FACL）和更好的优化器

---

## E2a — 仅 FACL loss

**日期**: 2026-03-19
**run_name**: `E2a_facl_only`
**配置**:
- loss: facl+mse（lambda=0.5，等权）
- optimizer: Adam, lr=1e-3
- scheduler: none
- epochs: 100, batch_size: 16

**指标（best checkpoint，epoch 6）**:

| csi_weak | pod_weak | far_weak | csi_strong | mse |
|----------|----------|----------|------------|-----|
| 0.4759 | 0.7499 | 0.4342 | 0.0269 | 0.001414 |

**训练曲线观察**:
- epoch 1-6：CSI 0.35→0.48，FAR 0.63→0.43，快速收敛
- epoch 6 后：train_loss 持续下降（0.317→0.235），val_csi 震荡在 0.43-0.47，**过拟合**
- epoch 100：CSI=0.436，POD=0.637，FAR=0.420

**结论**:
- FACL 有效降低 FAR（0.638→0.434，-0.20）✅
- 代价：POD 0.912→0.750（-0.16），CSI_strong 0.077→0.027（强降水能力下降）❌
- best_epoch=6 说明 lambda=0.5 权重过大，FACL 项主导梯度，过早收敛后过拟合
- **需要调小 lambda（建议 0.1）**

---

## E2b — 仅 AdamW + CosineAnnealingLR

**日期**: 2026-03-19
**run_name**: `E2b_adamw_cosine`
**配置**:
- loss: weighted_mse（RAIN_WEIGHT=10）
- optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- scheduler: CosineAnnealingLR, T_max=epochs
- epochs: 100, batch_size: 16

**指标（best checkpoint，epoch 86）**:

| csi_weak | pod_weak | far_weak | csi_strong | mse |
|----------|----------|----------|------------|-----|
| 0.4300 | 0.8193 | 0.5250 | 0.0470 | 0.001446 |

**训练曲线观察**:
- epoch 1-4：POD=1.0，FAR=0.913，CSI=0.087，模型全预测为雨（局部最优）
- epoch 5：突然跳出，CSI=0.219
- epoch 6-86：震荡上升，best=0.430
- epoch 100：train_loss=0.000167（极低！），val_csi=0.410，**严重过拟合**

**结论**:
- AdamW+Cosine 单独改善有限：FAR 0.638→0.525（-0.11），CSI +0.08
- train_loss→0.000167 是严重过拟合信号，Cosine 把 lr 降到接近 0 后模型记住训练集
- epoch 1-4 卡在"全预测雨"局部最优，说明 weighted_mse 单独不足以跳出
- **T_max 应设为 epochs//2，或配合 FACL 使用**

---

## E2c — 全套（FACL + AdamW + Cosine）

**日期**: 2026-03-19
**run_name**: `E2c_full`
**配置**:
- loss: facl+mse（lambda=0.5，等权）
- optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- scheduler: CosineAnnealingLR, T_max=epochs
- epochs: 100, batch_size: 16

**指标（best checkpoint，epoch 24）**:

| csi_weak | pod_weak | far_weak | csi_strong | mse |
|----------|----------|----------|------------|-----|
| 0.4738 | 0.7248 | 0.4223 | 0.0575 | 0.001436 |

**训练曲线观察**:
- epoch 1：CSI=0.298，epoch 2 短暂退化（CSI=0.095），epoch 3 快速跳出（CSI=0.422）
- FACL 帮助跳出"全预测雨"局部最优（E2b 卡了 4 epoch，E2c 第 3 epoch 就跳出）✅
- epoch 24：best CSI=0.474，之后震荡下降
- epoch 100：train_loss=0.235，val_csi=0.455，过拟合程度比 E2b 轻

**结论**:
- 当前三组实验中综合最优：FAR 最低（0.422），CSI_strong 最高（0.058）
- FACL 与 AdamW+Cosine 有协同效应：跳出局部最优更快，训练更稳定
- best_epoch=24 仍偏早，lambda=0.5 仍然过大
- **下一步：调小 lambda 到 0.1，预期 POD 回升、best_epoch 推后**

---

## 实验对比矩阵

| 实验 | best_epoch | csi_weak | pod_weak | far_weak | csi_strong | 关键变量 |
|------|-----------|----------|----------|----------|------------|---------|
| E1 | ~90 | 0.350 | 0.912 | 0.638 | 0.077 | baseline |
| E2a | 6 | 0.476 | 0.750 | 0.434 | 0.027 | +FACL |
| E2b | 86 | 0.430 | 0.819 | 0.525 | 0.047 | +AdamW+Cosine |
| E2c | 24 | 0.474 | 0.725 | 0.422 | 0.058 | +FACL+AdamW+Cosine |
| Persistence | — | 0.928 | 0.962 | 0.037 | 0.856 | 上界 |

**E1→E2x 增益**:
- FACL 单独：CSI +0.126，FAR -0.204，POD -0.162
- AdamW+Cosine 单独：CSI +0.080，FAR -0.113，POD -0.093
- 全套：CSI +0.124，FAR -0.216，POD -0.187

---

## 当前最优

**最优配置**: E2c（facl+mse + AdamW + CosineAnnealingLR）
**最优指标**: csi_weak=0.474，far_weak=0.422，pod_weak=0.725
**主要问题**: lambda_facl=0.5 过大，POD 损失过多，best_epoch 偏早

---

## E3 — FACL lambda=0.1 + AdamW + Cosine

**日期**: 2026-03-21
**run_name**: `E3_facl_lambda01`
**配置**:
- loss: facl+mse，**lambda_facl=0.1**
- optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- scheduler: CosineAnnealingLR, T_max=100
- epochs: 100, **batch_size: 64**（注意：比 E1-E2c 大，之前均为 16）

> ⚠️ metrics.json 开头有两个 epoch=1 条目，说明训练中断后 resume 过一次。

**指标（best checkpoint，epoch 41）**:

| csi_weak | pod_weak | far_weak | csi_strong | mse |
|----------|----------|----------|------------|-----|
| 0.4432 | 0.8195 | 0.5089 | 0.1064 | 0.001439 |

**训练曲线观察**:
- epoch 1-4：POD=1.0，FAR=0.91，CSI=0.087，卡在「全预测为雨」局部最优（同 E2b）
- epoch 5：突然跳出（CSI 0.087→0.380），比 E2b（epoch 5 跳出）更慢，比 E2c（epoch 3）慢——lambda=0.1 弱化了 FACL 的跳出能力
- epoch 10：CSI=0.421（第一个高峰）
- epoch 41：best CSI=0.4432，此后进入 0.41–0.44 宽幅震荡平台
- epoch 41–100：train_loss 仅从 0.0289→0.0262（降幅极小），val_csi 无明显趋势
- epoch 100：CSI=0.435，train_loss=0.0262，**无明显过拟合**（train/val loss 差距小）

**观察**:
- POD 回升至 0.82（vs E2a/E2c 的 0.75/0.72）
- CSI_strong 0.106，明显高于所有 E2 实验（vs E2c 0.058）
- FAR=0.509，高于 E2c（0.422）
- best_epoch=41（vs E2c 的 24），峰值前训练更稳定
- epoch 41 后 train_loss 和 val_csi 几乎停止改善（train_loss 0.0289→0.0262，降幅极小）

**待验证假设**（E3 与 E2c 同时改了 lambda 和 batch_size，且有 resume，因果无法单独归因）:
- 假设 A：CosineAnnealingLR 把 lr 降至接近 0 导致后期停止优化（平台化）
- 假设 B：batch_size 从 16→64 减少有效优化步数，也可能贡献平台化
- 假设 C：lambda=0.1 弱化了 FACL 对误报的惩罚力度，导致 FAR 退步
- 三者在当前实验中混杂，需要 E4 对照实验分离

> ⚠️ 本轮训练发生过 resume，曲线细节归因需谨慎。

---

## 实验对比矩阵（更新至 E3）

| 实验 | best_epoch | csi_weak | pod_weak | far_weak | csi_strong | batch_size | 关键变量 |
|------|-----------|----------|----------|----------|------------|-----------|---------|
| E1 | ~90 | 0.350 | 0.912 | 0.638 | 0.077 | 16 | baseline |
| E2a | 6 | 0.476 | 0.750 | 0.434 | 0.027 | 16 | +FACL(λ=0.5) |
| E2b | 86 | 0.430 | 0.819 | 0.525 | 0.047 | 16 | +AdamW+Cosine |
| E2c | 24 | 0.474 | 0.725 | 0.422 | 0.058 | 16 | +FACL(λ=0.5)+AdamW+Cosine |
| E3 | 41 | 0.443 | 0.820 | 0.509 | 0.106 | 64 | +FACL(λ=0.1)+AdamW+Cosine |
| Persistence | — | 0.928 | 0.962 | 0.037 | 0.856 | — | 上界 |

**E1→E3 增益（E3 best，注意 batch_size 不同，不可直接对比）**:
- CSI +0.093，FAR -0.129，POD -0.092，CSI_strong +0.029

---

## 当前候选最优（Pareto 权衡）

不同目标维度下的最优候选不同，主指标优先级尚未最终确定：

| 优化目标 | 最优候选 | 指标 |
|---------|---------|------|
| **csi_weak 最高** | E2c | 0.474 |
| **far_weak 最低** | E2c | 0.422 |
| **csi_strong 最高** | E3 | 0.106 |
| **pod_weak 最高** | E1 | 0.912（但 FAR 最差）|
| **训练稳定性** | E2b/E3 | best_epoch 靠后 |

**主指标优先级（暂定）**: `csi_weak` > `far_weak` > `csi_strong`
- 按此优先级：**E2c 当前最优**（csi_weak=0.474，far_weak=0.422）
- E3 在 csi_strong 上有优势，若后续转向强降水优先，E3 方向更值得继续

---

## 待跑实验

### E4 — 拆分 E3 混杂因素（三组对照）

**目的**：分离 batch_size 和 scheduler 类型对 E3 表现的各自贡献，同时寻找更好的 recipe。

**E4a — 基准对照（仅控制 batch_size）**:
- loss: facl+mse，lambda_facl=0.1
- optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- scheduler: CosineAnnealingLR（与 E3 相同）
- epochs: 100, **batch_size: 16**
- 目的：验证 E3 的平台化/FAR 退步中有多少来自 batch_size=64

**E4b — WarmRestart**:
- loss: facl+mse，lambda_facl=0.1
- optimizer: AdamW, lr=1e-3
- scheduler: CosineAnnealingWarmRestarts, T_0=20, T_mult=2
- epochs: 100, batch_size: 16
- 目的：验证周期性重启 lr 是否打破平台（需先实现 `--scheduler cosine_restart`）

**E4c — 无 scheduler 对照**:
- loss: facl+mse，lambda_facl=0.1
- optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- scheduler: none
- epochs: 100, batch_size: 16
- 目的：验证 Cosine 是否真的有害

**决策逻辑**:
- E4a > E3（csi_weak）→ 证实 batch_size=64 是 E3 瓶颈之一
- E4b > E4a → warm restart 相对 plain cosine 有效
- E4c > E4a → plain cosine 有害，应弃用
- E4b 或 E4c > E2c → lambda=0.1 + batch_size=16 可以超越 lambda=0.5 的 E2c
