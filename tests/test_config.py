from omnimamba.config import TrainingConfig


def test_default_split_ranges():
    cfg = TrainingConfig()
    assert cfg.train_start < cfg.train_end
    assert cfg.val_start < cfg.val_end
    assert cfg.test_start < cfg.test_end
