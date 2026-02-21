import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from omnimamba.config import TrainingConfig
from omnimamba.losses import SpectralStructuralWeightedLoss
from omnimamba.train_loop import train_epoch, validate_epoch

T_SEQ = 4  # short sequence for test speed


class _DummyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img1 = torch.randn(1, 66, 70)              # PWV [1, H, W]
        radar_seq = torch.randn(T_SEQ, 1, 66, 70)  # [T, 1, H, W]
        targets = torch.randn(3, 66, 70)
        return img1, radar_seq, targets


def test_train_loop_smoke():
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
        radar_seq_len=T_SEQ,
    )
    loader = DataLoader(_DummyDataset(), batch_size=1)
    device = torch.device("cpu")

    criterion = SpectralStructuralWeightedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss, _ = train_epoch(model, loader, device, criterion, optimizer)
    assert torch.isfinite(torch.tensor(loss))

    val_loss, metrics = validate_epoch(model, loader, device)
    assert torch.isfinite(torch.tensor(val_loss))
    assert "mae" in metrics
    assert "csi" in metrics
    assert "ets" in metrics
