import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from omnimamba.config import TrainingConfig
from omnimamba.losses import SpectralStructuralWeightedLoss
from omnimamba.train_loop import train_epoch, validate_epoch


class _DummyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img1 = torch.randn(1, 66, 70)
        img2 = torch.randn(1, 66, 70)
        targets = torch.randn(3, 66, 70)
        return img1, img2, targets


def test_train_loop_smoke():
    pytest.importorskip("mamba_ssm")
    from omnimamba.model import CrossAttentionMamba

    cfg = TrainingConfig(epochs=1, batch_size=1)
    model = CrossAttentionMamba(
        img_size=66,
        img_size_w=70,
        patch_size=2,
        stride=2,
        dim=64,
        depth=1,
        d_state=16,
        num_classes=3,
    )
    loader = DataLoader(_DummyDataset(), batch_size=1)
    device = torch.device("cpu")

    criterion = SpectralStructuralWeightedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss, _ = train_epoch(model, loader, device, criterion, optimizer)
    assert torch.isfinite(torch.tensor(loss))

    val_loss, metrics = validate_epoch(model, loader, device)
    assert val_loss >= 0
    assert "mae" in metrics
