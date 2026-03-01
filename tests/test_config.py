from omnimamba.config import TrainingConfig


def test_default_split_ranges():
    cfg = TrainingConfig()
    assert cfg.train_start < cfg.train_end
    assert cfg.val_start < cfg.val_end
    assert cfg.test_start < cfg.test_end
    assert 0.0 < cfg.val_ema_alpha <= 1.0
    assert cfg.early_stop_min_delta >= 0.0
