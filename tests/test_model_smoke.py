import torch

from omnimamba.model import CrossAttentionMamba


def test_model_forward_shape():
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
    x1 = torch.randn(1, 1, 66, 70)
    x2 = torch.randn(1, 1, 66, 70)
    out = model(x1, x2)
    assert out.shape == (1, 3, 66, 70)
