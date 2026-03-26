# tests/test_model_convlstm.py
"""Tests for ConvLSTMModel."""
import torch
import pytest
from src.config import T, H, W

B = 2   # batch size for all tests


class TestConvLSTMCell:
    def test_output_shape(self):
        """ConvLSTMCell 输出 h 和 c 的 shape 应与输入一致。"""
        from src.model_convlstm import ConvLSTMCell
        cell = ConvLSTMCell(in_channels=1, hidden_channels=16, kernel_size=3)
        x = torch.zeros(B, 1, H, W)
        h = torch.zeros(B, 16, H, W)
        c = torch.zeros(B, 16, H, W)
        h_new, c_new = cell(x, h, c)
        assert h_new.shape == (B, 16, H, W)
        assert c_new.shape == (B, 16, H, W)

    def test_cell_not_all_zero(self):
        """非零输入应产生非零隐状态（验证门控激活正常）。"""
        from src.model_convlstm import ConvLSTMCell
        cell = ConvLSTMCell(in_channels=1, hidden_channels=16, kernel_size=3)
        x = torch.ones(B, 1, H, W)
        h = torch.zeros(B, 16, H, W)
        c = torch.zeros(B, 16, H, W)
        h_new, c_new = cell(x, h, c)
        assert h_new.abs().sum() > 0


class TestConvLSTMModel:
    def test_forward_output_shape(self):
        """主模型 forward 输出 shape 应为 [B, 1, H, W]。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel()
        radar = torch.zeros(B, T, 1, H, W)
        pwv   = torch.zeros(B, 1, 1, H, W)
        out = model(radar, pwv)
        assert out.shape == (B, 1, H, W), f"Expected ({B},1,{H},{W}), got {out.shape}"

    def test_output_range(self):
        """输出值应在 [0, 1]（Sigmoid 激活）。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel()
        radar = torch.rand(B, T, 1, H, W)
        pwv   = torch.rand(B, 1, 1, H, W)
        out = model(radar, pwv)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_gradient_flows(self):
        """反向传播应能更新所有参数（无梯度断裂）。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel()
        radar = torch.rand(B, T, 1, H, W)
        pwv   = torch.rand(B, 1, 1, H, W)
        target = torch.rand(B, 1, H, W)
        out = model(radar, pwv)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN grad for {name}"

    def test_deterministic_eval(self):
        """eval 模式下两次 forward 结果完全一致（无 Dropout）。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel()
        model.eval()
        torch.manual_seed(42)
        radar = torch.rand(B, T, 1, H, W)
        pwv   = torch.rand(B, 1, 1, H, W)
        with torch.no_grad():
            out1 = model(radar, pwv)
            out2 = model(radar, pwv)
        assert torch.allclose(out1, out2)

    def test_custom_hidden_dim(self):
        """支持自定义 hidden_dim 参数。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel(hidden_dim=32, num_layers=1)
        radar = torch.zeros(B, T, 1, H, W)
        pwv   = torch.zeros(B, 1, 1, H, W)
        out = model(radar, pwv)
        assert out.shape == (B, 1, H, W)

    def test_legacy_t_kwarg(self):
        """ConvLSTMModel(t=10) 应能正常构造（向后兼容 StubModel(t=T)）。"""
        from src.model_convlstm import ConvLSTMModel
        model = ConvLSTMModel(t=10)
        radar = torch.zeros(B, T, 1, H, W)
        pwv   = torch.zeros(B, 1, 1, H, W)
        out = model(radar, pwv)
        assert out.shape == (B, 1, H, W)
