"""Weighted MSE loss for precipitation nowcasting."""
import torch
import torch.nn as nn
from src.config import RAIN_EPS, RAIN_WEIGHT


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pixel-wise MSE with elevated weight for rain pixels.

    Args:
        pred:   float32 tensor, normalised [0, 1], any shape.
        target: same shape as pred.

    Returns:
        Scalar loss tensor.
    """
    rain_mask = (target > RAIN_EPS).float()
    weight = 1.0 + (RAIN_WEIGHT - 1.0) * rain_mask   # 1 for no-rain, RAIN_WEIGHT for rain
    return (weight * (pred - target) ** 2).mean()


def _amplitude_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Normalised L1 distance between FFT amplitude spectra.

    pred / target: float32, shape [B, C, H, W], values in [0, 1].

    The raw FFT amplitude scales with H*W (DC component ~ sum of pixel values).
    We divide by H*W so the result is comparable in magnitude to pixel-space MSE.
    Without this normalisation, amplitude loss is ~140x larger than MSE loss
    for typical rain-like inputs (measured on [B=8, H=66, W=70] tensors).
    """
    H, W = pred.shape[-2], pred.shape[-1]
    pred_fft   = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    pred_amp   = pred_fft.abs()
    target_amp = target_fft.abs()
    return (pred_amp - target_amp).abs().mean() / (H * W)


def _correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    1 - Pearson correlation between pred and target (averaged over batch).

    Each sample [C, H, W] is flattened to a vector; correlation is computed
    per sample then averaged. Returns values in [0, 2]; 0 = perfect correlation.

    Degenerate case: if pred or target is spatially constant (all same value),
    the norm is 0 and correlation is undefined. clamp(min=1e-8) in the
    denominator causes r → 0, giving loss → 1.0 (treated as uncorrelated).
    """
    B = pred.shape[0]
    p = pred.view(B, -1)    # [B, N]
    t = target.view(B, -1)  # [B, N]

    p_mean = p.mean(dim=1, keepdim=True)
    t_mean = t.mean(dim=1, keepdim=True)
    p_c = p - p_mean
    t_c = t - t_mean

    num   = (p_c * t_c).sum(dim=1)
    denom = (p_c.norm(dim=1) * t_c.norm(dim=1)).clamp(min=1e-8)
    r     = num / denom           # [B], in [-1, 1]
    return (1.0 - r).mean()       # loss in [0, 2]; 0 when perfectly correlated


def facl_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_amp: float = 1.0,
    lambda_corr: float = 0.5,
) -> torch.Tensor:
    """
    Fourier Amplitude and Correlation Loss (FACL).

    Combines:
    - Amplitude loss: normalised L1 on FFT amplitude spectra (divided by H*W
      to match pixel-space MSE magnitude; suppresses over-smoothing / blurring).
    - Correlation loss: 1 - Pearson correlation (encourages spatial structure
      alignment, robust to global brightness shift).

    Scale calibration (measured on [B=8, H=66, W=70] rain-like tensors):
      weighted_mse  ≈ 0.018
      amp_loss/H/W  ≈ 0.00056  (lambda_amp=1.0 → contributes ~0.03x MSE)
      corr_loss     ≈ 0.043    (lambda_corr=0.5 → contributes ~1.2x MSE)
    Total FACL with defaults is ~1.2x weighted_mse — suitable for direct addition.

    Typical usage in train.py:
        total_loss = weighted_mse_loss(pred, target) + facl_loss(pred, target)

    Reference: Yan et al., "Fourier Amplitude and Correlation Loss: Beyond
    Using L2 Loss for Skillful Precipitation Nowcasting", NeurIPS 2024.

    Args:
        pred:        float32, normalised [0, 1], shape [B, C, H, W].
        target:      same shape as pred.
        lambda_amp:  weight for normalised amplitude loss (default 1.0).
        lambda_corr: weight for correlation loss (default 0.5).

    Returns:
        Scalar loss tensor.
    """
    amp_l  = _amplitude_loss(pred, target)
    corr_l = _correlation_loss(pred, target)
    return lambda_amp * amp_l + lambda_corr * corr_l
