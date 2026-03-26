"""Pixel normalisation utilities shared by all modalities."""
import numpy as np

def normalize(pixel: np.ndarray) -> np.ndarray:
    """
    Convert uint8 array to float32 in [0, 1].
    Inverted encoding: 255 (no signal) → 0.0, 0 (max signal) → 1.0.
    """
    return (255.0 - pixel.astype(np.float32)) / 255.0

def denormalize(arr: np.ndarray) -> np.ndarray:
    """
    Convert normalised float32 back to uint8.
    Inverse of normalize(); clips to [0, 255] before casting.
    """
    pixel = 255.0 - arr * 255.0
    return np.clip(np.round(pixel), 0, 255).astype(np.uint8)
