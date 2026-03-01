from omnimamba.train_loop import _is_significant_improvement, _update_ema


def test_update_ema_initializes_with_first_value():
    assert _update_ema(None, 0.5, 0.2) == 0.5


def test_update_ema_applies_weighted_average():
    # ema = alpha * current + (1-alpha) * prev
    ema = _update_ema(0.5, 0.4, 0.2)
    assert abs(ema - 0.48) < 1e-9


def test_significant_improvement_respects_min_delta():
    best = 0.5
    # not enough improvement: 0.00005 < min_delta
    assert not _is_significant_improvement(0.49995, best, 1e-4)
    # enough improvement: 0.0002 > min_delta
    assert _is_significant_improvement(0.4998, best, 1e-4)
