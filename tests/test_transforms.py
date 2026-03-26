# tests/test_transforms.py
import numpy as np
import pytest
from src.transforms import normalize, denormalize

def test_normalize_no_rain():
    """Pixel 255 (no signal) should map to 0.0."""
    arr = np.array([[255, 255]], dtype=np.uint8)
    result = normalize(arr)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.0)

def test_normalize_max_signal():
    """Pixel 0 (max signal) should map to 1.0."""
    arr = np.array([[0, 0]], dtype=np.uint8)
    result = normalize(arr)
    np.testing.assert_allclose(result, 1.0)

def test_normalize_midpoint():
    arr = np.array([[127]], dtype=np.uint8)
    result = normalize(arr)
    np.testing.assert_allclose(result, (255 - 127) / 255.0, rtol=1e-5)

def test_normalize_preserves_shape():
    arr = np.zeros((66, 70), dtype=np.uint8)
    assert normalize(arr).shape == (66, 70)

def test_denormalize_roundtrip():
    """denormalize(normalize(x)) should recover original uint8 within ±1."""
    arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
    recovered = denormalize(normalize(arr))
    assert recovered.dtype == np.uint8
    np.testing.assert_array_equal(recovered, arr)
