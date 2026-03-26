"""
Minimal stub model: RADAR time steps + PWV flattened into channel dimension,
then processed by a 2D convolutional network.
NOT intended as the final model. Replace by implementing the same forward() signature.

Interface contract:
    Input:
        radar : Tensor [B, T, 1, H, W]
        pwv   : Tensor [B, 1, 1, H, W]
    Output:
        Tensor [B, 1, H, W]  — predicted RAIN(t+1), normalised [0, 1]
"""
import torch
import torch.nn as nn
from src.config import T, H, W


class StubModel(nn.Module):
    def __init__(self, t: int = T) -> None:
        super().__init__()
        in_ch = t + 1   # T RADAR frames + 1 PWV frame (all flattened to channel dim)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, radar: torch.Tensor, pwv: torch.Tensor) -> torch.Tensor:
        # radar: [B, T, 1, H, W] → squeeze channel → [B, T, H, W]
        # pwv:   [B, 1, 1, H, W] → squeeze channel → [B, 1, H, W]
        radar_2d = radar.squeeze(2)
        pwv_2d   = pwv.squeeze(2)
        x = torch.cat([radar_2d, pwv_2d], dim=1)  # [B, T+1, H, W]
        return self.net(x)   # [B, 1, H, W]
