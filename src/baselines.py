"""Persistence and Zero baselines for evaluation."""
from __future__ import annotations
from typing import Tuple
import torch


class PersistenceBaseline:
    """Predict RAIN(t+1) = RAIN(t). Requires RAIN at current timestep."""

    @staticmethod
    def predict(rain_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rain_t: normalised RAIN at time t, shape [B, 1, H, W].
        Returns:
            Prediction for t+1 (identical copy).
        """
        return rain_t.clone()


class ZeroBaseline:
    """Predict all-no-rain everywhere."""

    @staticmethod
    def predict(shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Args:
            shape: desired output shape, e.g. (B, 1, H, W).
        Returns:
            Zero tensor (no rain predicted).
        """
        return torch.zeros(shape, dtype=torch.float32)
