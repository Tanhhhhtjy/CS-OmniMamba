import pytest
import torch


def test_model_forward_shape():
    from omnimamba.model import CrossAttentionMamba

    model = CrossAttentionMamba(
        img_size=66,
        img_size_w=70,
        patch_size=2,
        stride=2,
        dim=64,
        depth=1,
        d_state=16,
        num_classes=3,
        radar_seq_len=4,
    )
    pwv = torch.randn(1, 1, 66, 70)
    radar_seq = torch.randn(1, 4, 1, 66, 70)   # [B, T, 1, H, W]
    out = model(pwv, radar_seq)
    assert out.shape == (1, 3, 66, 70)
    # output must be in [0, 1] due to sigmoid
    assert out.min().item() >= -1e-6
    assert out.max().item() <= 1.0 + 1e-6
