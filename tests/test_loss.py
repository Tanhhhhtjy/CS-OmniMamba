# tests/test_loss.py
import torch
import pytest
from src.loss import weighted_mse_loss
from src.config import RAIN_EPS, RAIN_WEIGHT

def test_zero_loss_for_perfect_prediction():
    t = torch.rand(2, 1, 66, 70)
    assert weighted_mse_loss(t, t).item() == pytest.approx(0.0, abs=1e-6)

def test_rain_pixels_weighted_higher():
    """A unit error on a rain pixel should produce more loss than on a no-rain pixel."""
    # no-rain target: 0.0; rain target: 1.0 (above EPS)
    target_rain   = torch.ones(1, 1, 1, 1)
    target_norain = torch.zeros(1, 1, 1, 1)
    pred_off      = torch.zeros(1, 1, 1, 1)

    loss_rain   = weighted_mse_loss(pred_off, target_rain)
    loss_norain = weighted_mse_loss(pred_off, target_norain)
    # rain: weight=RAIN_WEIGHT, no-rain: weight=1
    assert loss_rain.item() > loss_norain.item()

def test_no_rain_frame_has_finite_loss():
    target = torch.zeros(2, 1, 66, 70)  # all no-rain
    pred   = torch.full_like(target, 0.5)
    loss   = weighted_mse_loss(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

def test_loss_weight_ratio():
    """With one rain pixel vs one no-rain pixel, ratio should equal RAIN_WEIGHT."""
    pred_rain   = torch.zeros(1, 1, 1, 1)
    target_rain = torch.ones(1, 1, 1, 1)   # rain: target > EPS, error = 1
    pred_no     = torch.ones(1, 1, 1, 1)
    target_no   = torch.zeros(1, 1, 1, 1)  # no-rain, error = 1

    loss_rain = weighted_mse_loss(pred_rain, target_rain).item()
    loss_no   = weighted_mse_loss(pred_no,   target_no).item()
    assert loss_rain == pytest.approx(loss_no * RAIN_WEIGHT, rel=1e-4)


# ── FACL tests ────────────────────────────────────────────────────────────────
from src.loss import facl_loss

def test_facl_zero_loss_perfect_prediction():
    """完美预测时 FACL loss 应接近 0。"""
    t = torch.rand(2, 1, 66, 70)
    loss = facl_loss(t, t)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)

def test_facl_positive_for_wrong_prediction():
    """预测全零、target 全一时 loss 应 > 0。"""
    pred   = torch.zeros(2, 1, 66, 70)
    target = torch.ones(2, 1, 66, 70)
    assert facl_loss(pred, target).item() > 0.0

def test_facl_finite():
    """随机输入不得产生 NaN 或 Inf。"""
    pred   = torch.rand(2, 1, 66, 70)
    target = torch.rand(2, 1, 66, 70)
    loss = facl_loss(pred, target)
    assert torch.isfinite(loss), f"facl_loss returned non-finite: {loss.item()}"

def test_facl_shape_invariant():
    """不同 batch size 和空间尺寸下均应返回标量。"""
    for shape in [(1, 1, 66, 70), (4, 1, 66, 70), (2, 1, 32, 32)]:
        pred   = torch.rand(*shape)
        target = torch.rand(*shape)
        loss = facl_loss(pred, target)
        assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"

def test_facl_less_loss_for_closer_prediction():
    """更接近 target 的预测应产生更低的 FACL loss。"""
    torch.manual_seed(42)  # fixed seed for determinism
    target = torch.rand(2, 1, 66, 70)
    pred_close = target + 0.01 * torch.randn_like(target)
    pred_far   = torch.rand_like(target)
    loss_close = facl_loss(pred_close.clamp(0, 1), target)
    loss_far   = facl_loss(pred_far, target)
    assert loss_close.item() < loss_far.item()

def test_facl_scale_comparable_to_mse():
    """归一化后 facl_loss 与 weighted_mse_loss 量级可比（0.001x ~ 10x MSE 范围内）。
    这确保 FACL 不会因量纲差异压制 MSE 或被 MSE 压制。
    """
    from src.loss import weighted_mse_loss
    torch.manual_seed(0)
    pred   = torch.rand(4, 1, 66, 70)
    target = torch.rand(4, 1, 66, 70)
    mse = weighted_mse_loss(pred, target).item()
    fl  = facl_loss(pred, target).item()
    ratio = fl / (mse + 1e-8)
    assert 0.1 < ratio < 5.0, (
        f"FACL/MSE ratio = {ratio:.3f} is outside [0.1, 5] — "
        f"amplitude normalization may be missing (FACL={fl:.4f}, MSE={mse:.4f})"
    )

def test_facl_gradients_finite_and_nonzero():
    """FACL 反向传播时梯度应有限且非零（确保训练接线正确）。"""
    pred   = torch.rand(2, 1, 66, 70, requires_grad=True)
    target = torch.rand(2, 1, 66, 70)
    loss = facl_loss(pred, target)
    loss.backward()
    assert pred.grad is not None, "No gradient computed"
    assert torch.isfinite(pred.grad).all(), "Gradient contains NaN or Inf"
    assert pred.grad.abs().sum().item() > 0.0, "Gradient is all zeros"
