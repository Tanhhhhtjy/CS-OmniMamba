"""
ConvLSTM-based precipitation nowcasting model.

Interface (identical to StubModel):
    forward(radar[B,T,1,H,W], pwv[B,1,1,H,W]) -> Tensor[B,1,H,W]

Architecture:
    1. Concatenate PWV as the (T+1)-th "pseudo-frame" → [B, T+1, 1, H, W]
    2. Multi-layer ConvLSTM processes frames sequentially
    3. Final hidden state h: [B, hidden_dim, H, W]
    4. Output head: Conv2d(hidden_dim→1) + Sigmoid → [B, 1, H, W]
"""
from __future__ import annotations

import torch
import torch.nn as nn
# T, H, W not imported — forward infers spatial dims from input tensor shape


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al., 2015)."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        # Gates: i, f, g, o — fused into a single 4× conv for efficiency
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
        )

    def forward(
        self,
        x: torch.Tensor,          # [B, in_channels, H, W]
        h: torch.Tensor,          # [B, hidden_channels, H, W]
        c: torch.Tensor,          # [B, hidden_channels, H, W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)         # [B, in+hidden, H, W]
        gates = self.conv(combined)                  # [B, 4*hidden, H, W]
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMModel(nn.Module):
    """
    Multi-layer ConvLSTM for Seq→1 precipitation nowcasting.

    Args:
        hidden_dim:  Number of feature channels in each ConvLSTM hidden state.
        num_layers:  Number of stacked ConvLSTM layers.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, t: int | None = None, **kwargs) -> None:
        # t is accepted but ignored (legacy StubModel(t=T) compatibility)
        if kwargs:
            raise TypeError(f"ConvLSTMModel got unexpected keyword arguments: {list(kwargs.keys())}")
        super().__init__()
        self.num_layers = num_layers

        # Build stacked ConvLSTM cells
        # Layer 0: input channels = 1 (grayscale)
        # Layer k>0: input channels = hidden_dim (from previous layer's h)
        cells = []
        for i in range(num_layers):
            in_ch = 1 if i == 0 else hidden_dim
            cells.append(ConvLSTMCell(in_ch, hidden_dim, kernel_size=3))
        self.cells = nn.ModuleList(cells)

        # Output projection: hidden → 1 channel rain map
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        radar: torch.Tensor,   # [B, T, 1, H, W]
        pwv:   torch.Tensor,   # [B, 1, 1, H, W]
    ) -> torch.Tensor:         # [B, 1, H, W]
        assert radar.ndim == 5 and radar.shape[2] == 1, \
            f"radar must be [B,T,1,H,W], got {tuple(radar.shape)}"
        assert pwv.ndim == 5 and pwv.shape[1] == 1 and pwv.shape[2] == 1, \
            f"pwv must be [B,1,1,H,W], got {tuple(pwv.shape)}"
        assert radar.shape[0] == pwv.shape[0] and radar.shape[3:] == pwv.shape[3:], \
            "radar and pwv must share batch size and spatial dims"

        B, _, _, H, W = radar.shape

        # Concatenate PWV as the (T+1)-th frame → [B, T+1, 1, H, W]
        seq = torch.cat([radar, pwv], dim=1)          # [B, T+1, 1, H, W]
        T_seq = seq.shape[1]

        # Initialise hidden states for all layers (new_zeros tracks device AND dtype)
        h = [radar.new_zeros(B, cell.hidden_channels, H, W) for cell in self.cells]
        c = [radar.new_zeros(B, cell.hidden_channels, H, W) for cell in self.cells]

        # Step through the sequence
        for t in range(T_seq):
            x_t = seq[:, t]           # [B, 1, H, W]
            for layer_idx, cell in enumerate(self.cells):
                inp = x_t if layer_idx == 0 else h[layer_idx - 1]
                h[layer_idx], c[layer_idx] = cell(inp, h[layer_idx], c[layer_idx])

        # Use last layer's final hidden state as feature map
        out = self.output_conv(h[-1])   # [B, 1, H, W]
        return torch.sigmoid(out)
