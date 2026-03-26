# tests/test_baselines.py
import torch
from src.baselines import PersistenceBaseline, ZeroBaseline

def test_persistence_copies_last_rain_frame():
    # batch: last RAIN frame at t is known; predict RAIN(t+1) = RAIN(t)
    rain_t = torch.rand(4, 1, 66, 70)
    pred   = PersistenceBaseline.predict(rain_t)
    assert pred.shape == rain_t.shape
    assert torch.allclose(pred, rain_t)

def test_zero_baseline_all_zeros():
    shape = (4, 1, 66, 70)
    pred = ZeroBaseline.predict(shape)
    assert pred.shape == shape
    assert pred.sum().item() == 0.0
    assert pred.dtype == torch.float32
