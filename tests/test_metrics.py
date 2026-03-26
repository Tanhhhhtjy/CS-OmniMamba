# tests/test_metrics.py
import torch
import pytest
from src.metrics import compute_csi_pod_far, compute_mse_mae, MetricsAccumulator

# ── CSI / POD / FAR ───────────────────────────────────────────────────────────

def test_perfect_prediction_csi_one():
    t = torch.rand(2, 1, 66, 70)
    m = compute_csi_pod_far(t, t, threshold=0.02)
    assert m["csi"] == pytest.approx(1.0, abs=1e-5)
    assert m["pod"] == pytest.approx(1.0, abs=1e-5)
    assert m["far"] == pytest.approx(0.0, abs=1e-5)

def test_all_zero_prediction_csi_zero():
    target = torch.rand(2, 1, 66, 70).clamp(0.05, 1.0)  # all rain
    pred   = torch.zeros_like(target)
    m = compute_csi_pod_far(pred, target, threshold=0.02)
    assert m["csi"] == pytest.approx(0.0, abs=1e-5)
    assert m["pod"] == pytest.approx(0.0, abs=1e-5)

def test_all_rain_false_alarm():
    target = torch.zeros(2, 1, 66, 70)       # no rain
    pred   = torch.ones(2, 1, 66, 70)        # predict all rain
    m = compute_csi_pod_far(pred, target, threshold=0.02)
    assert m["far"] == pytest.approx(1.0, abs=1e-5)

def test_no_rain_in_target_returns_nan_csi():
    """When there is genuinely no rain, CSI is undefined; return NaN."""
    target = torch.zeros(1, 1, 66, 70)
    pred   = torch.zeros(1, 1, 66, 70)
    m = compute_csi_pod_far(pred, target, threshold=0.02)
    assert torch.isnan(torch.tensor(m["csi"]))

# ── MSE / MAE ─────────────────────────────────────────────────────────────────

def test_mse_mae_perfect():
    t = torch.rand(2, 1, 66, 70)
    m = compute_mse_mae(t, t)
    assert m["mse"]      == pytest.approx(0.0, abs=1e-6)
    assert m["mae_rain"] == pytest.approx(0.0, abs=1e-6)

def test_mse_mae_all_norain_gives_nan_mae_rain():
    """MAE rain is undefined (no rain pixels) → NaN."""
    target = torch.zeros(2, 1, 66, 70)
    pred   = torch.rand(2, 1, 66, 70)
    m = compute_mse_mae(pred, target)
    assert torch.isnan(torch.tensor(m["mae_rain"]))

# ── MetricsAccumulator ────────────────────────────────────────────────────────

def test_accumulator_aggregates_tp_fp_fn():
    """
    MetricsAccumulator must aggregate raw TP/FP/FN across batches,
    NOT average per-batch CSI values.
    """
    acc = MetricsAccumulator()
    # Batch 1: all rain, perfect prediction
    t1 = torch.ones(1, 1, 66, 70) * 0.5
    acc.update(t1, t1)
    # Batch 2: also all rain, perfect prediction
    t2 = torch.ones(1, 1, 66, 70) * 0.5
    acc.update(t2, t2)
    result = acc.compute()
    assert result["csi_weak"] == pytest.approx(1.0, abs=1e-4)
    assert result["mse"]      == pytest.approx(0.0, abs=1e-5)

def test_accumulator_cross_batch_csi():
    """
    CSI computed from aggregated counts must differ from average of per-batch CSIs.
    Batch 1 has only FP (model predicts rain, no target rain) → per-batch CSI = NaN.
    Batch 2 has only TP.
    Aggregated: tp=16, fp=16, fn=0 → CSI = 16/(16+16+0) = 0.5
    """
    acc = MetricsAccumulator()
    # Batch 1: predict all rain, target all no-rain → all FP, CSI=NaN per-batch
    pred_fp  = torch.ones(1, 1, 4, 4) * 0.5
    tgt_none = torch.zeros(1, 1, 4, 4)
    acc.update(pred_fp, tgt_none)
    # Batch 2: perfect prediction on all-rain target → all TP, CSI=1 per-batch
    t_rain = torch.ones(1, 1, 4, 4) * 0.5
    acc.update(t_rain, t_rain)
    result = acc.compute()
    # Aggregated: tp=16, fp=16, fn=0 → CSI = 16/(16+16+0) = 0.5
    assert result["csi_weak"] == pytest.approx(0.5, abs=1e-4)
