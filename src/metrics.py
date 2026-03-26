"""
Evaluation metrics for precipitation nowcasting.
All inputs are normalised float32 tensors in [0, 1].
"""
from __future__ import annotations
import math
from typing import Dict

import torch
from src.config import RAIN_EPS

# ── Threshold definitions ─────────────────────────────────────────────────────
THRESH_WEAK   = 2.0  / 255.0   # original pixel < 253
THRESH_STRONG = 55.0 / 255.0   # original pixel < 200


def compute_csi_pod_far(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute CSI, POD, FAR for a given threshold.

    Returns dict with keys: 'csi', 'pod', 'far'.
    Returns NaN for CSI/POD when there are no rain pixels in target.
    """
    pred_bin   = (pred   > threshold).float()
    target_bin = (target > threshold).float()

    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()

    denom_csi = tp + fp + fn
    denom_pod = tp + fn
    denom_far = tp + fp

    # CSI and POD are undefined when there is no rain in target (denom_pod == 0)
    csi = tp / denom_csi if denom_pod > 0 else float("nan")
    pod = tp / denom_pod if denom_pod > 0 else float("nan")
    # FAR is undefined when the model predicts no rain at all (denom_far == 0)
    far = fp / denom_far if denom_far > 0 else float("nan")

    return {"csi": csi, "pod": pod, "far": far}


def compute_mse_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Compute full-image MSE and rain-only MAE."""
    mse = ((pred - target) ** 2).mean().item()

    rain_mask = (target > RAIN_EPS)
    if rain_mask.any():
        mae_rain = (pred[rain_mask] - target[rain_mask]).abs().mean().item()
    else:
        mae_rain = float("nan")

    return {"mse": mse, "mae_rain": mae_rain}


class MetricsAccumulator:
    """
    Accumulate CSI/POD/FAR/MSE/MAE over a full evaluation epoch by collecting
    raw TP/FP/FN counts and pixel-level squared errors, then computing final
    metrics once at the end.
    """

    def __init__(self) -> None:
        # TP/FP/FN counts for two thresholds
        self._tp_w = self._fp_w = self._fn_w = 0.0
        self._tp_s = self._fp_s = self._fn_s = 0.0
        # MSE accumulation (sum of squared errors + pixel count)
        self._sq_sum: float = 0.0
        self._n_pixels: int = 0
        # MAE-rain accumulation (sum of abs errors on rain pixels + count)
        self._abs_rain_sum: float = 0.0
        self._n_rain_pixels: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate raw counts from one batch (or one sample)."""
        for thresh, prefix in [(THRESH_WEAK, "w"), (THRESH_STRONG, "s")]:
            pred_bin   = (pred   > thresh).float()
            target_bin = (target > thresh).float()
            tp = (pred_bin * target_bin).sum().item()
            fp = (pred_bin * (1 - target_bin)).sum().item()
            fn = ((1 - pred_bin) * target_bin).sum().item()
            if prefix == "w":
                self._tp_w += tp; self._fp_w += fp; self._fn_w += fn
            else:
                self._tp_s += tp; self._fp_s += fp; self._fn_s += fn

        # MSE: sum of pixel-wise squared errors
        self._sq_sum   += ((pred - target) ** 2).sum().item()
        self._n_pixels += pred.numel()

        # MAE-rain: errors only on rain pixels
        rain_mask = target > RAIN_EPS
        n_rain = rain_mask.sum().item()
        if n_rain > 0:
            self._abs_rain_sum   += (pred[rain_mask] - target[rain_mask]).abs().sum().item()
            self._n_rain_pixels  += int(n_rain)

    @staticmethod
    def _csi_pod_far(tp: float, fp: float, fn: float) -> Dict[str, float]:
        denom_pod = tp + fn
        denom_csi = tp + fp + fn
        denom_far = tp + fp
        return {
            "csi": tp / denom_csi if denom_pod > 0 else float("nan"),
            "pod": tp / denom_pod if denom_pod > 0 else float("nan"),
            "far": fp / denom_far if denom_far > 0 else float("nan"),
        }

    def compute(self) -> Dict[str, float]:
        w = self._csi_pod_far(self._tp_w, self._fp_w, self._fn_w)
        s = self._csi_pod_far(self._tp_s, self._fp_s, self._fn_s)
        mse      = self._sq_sum / self._n_pixels if self._n_pixels > 0 else float("nan")
        mae_rain = (self._abs_rain_sum / self._n_rain_pixels
                    if self._n_rain_pixels > 0 else float("nan"))
        return {
            "csi_weak":   w["csi"], "pod_weak":   w["pod"], "far_weak":   w["far"],
            "csi_strong": s["csi"], "pod_strong": s["pod"], "far_strong": s["far"],
            "mse": mse,
            "mae_rain": mae_rain,
        }

    def reset(self) -> None:
        self.__init__()
