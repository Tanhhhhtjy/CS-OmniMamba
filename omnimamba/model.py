from __future__ import annotations

import torch
import torch.nn as nn


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_fc[2].bias, -2.0)
        self.last_gate_mean = None

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        context, _ = self.attn(self.norm_q(x_q), self.norm_kv(x_kv), self.norm_kv(x_kv))
        concat = torch.cat([x_q, context], dim=-1)
        gate = self.gate_fc(concat)
        if self.training:
            try:
                self.last_gate_mean = gate.mean().detach().cpu().item()
            except Exception:
                self.last_gate_mean = None
        output = x_q + gate * (context - x_q)
        return output


class OmniBiMambaBlockPseudo(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = nn.SiLU()

        self.ssm_h = nn.GRU(
            input_size=self.d_inner,
            hidden_size=d_state,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ssm_proj_h = nn.Linear(d_state * 2, self.d_inner)

        self.ssm_v = nn.GRU(
            input_size=self.d_inner,
            hidden_size=d_state,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ssm_proj_v = nn.Linear(d_state * 2, self.d_inner)

        self.fusion_linear = nn.Linear(self.d_inner * 2, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = x_branch.permute(0, 2, 1)
        x_branch = self.conv1d(x_branch)[:, :, : x.shape[1]]
        x_branch = self.activation(x_branch)
        x_branch = x_branch.permute(0, 2, 1)

        ssm_out_h, _ = self.ssm_h(x_branch)
        x_feat_h = self.ssm_proj_h(ssm_out_h)

        bsz, length, channels = x_branch.shape
        x_v_in = (
            x_branch.view(bsz, height, width, channels)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(bsz, length, channels)
        )
        ssm_out_v, _ = self.ssm_v(x_v_in)
        x_feat_v = self.ssm_proj_v(ssm_out_v)
        x_feat_v = (
            x_feat_v.view(bsz, width, height, channels)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(bsz, length, channels)
        )

        x_combined = torch.cat([x_feat_h, x_feat_v], dim=-1)
        x_ssm_total = self.fusion_linear(x_combined)

        z_branch = self.activation(z_branch)
        x_out = x_ssm_total * z_branch

        out = self.out_proj(x_out)
        return residual + self.dropout(out)


class PrecipitationEnhancementCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = max(1, out_channels // 4)
        total_branch_channels = branch_channels * 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 7, padding=3),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(total_branch_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concat = torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )
        return self.fusion(concat) + self.residual(x)


class CrossAttentionMamba(nn.Module):
    def __init__(
        self,
        img_size: int = 66,
        img_size_w: int = 70,
        patch_size: int = 2,
        stride: int = 2,
        in_chans: int = 1,
        num_classes: int = 3,
        dim: int = 128,
        depth: int = 4,
        d_state: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.stride = stride
        self.dim = dim
        self.img_size = img_size
        self.img_size_w = img_size_w

        self.num_patches_h = (img_size - patch_size) // stride + 1
        self.num_patches_w = (img_size_w - patch_size) // stride + 1
        num_patches = self.num_patches_h * self.num_patches_w

        def build_patch_embed():
            return nn.Sequential(
                nn.Conv2d(in_chans, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dim // 2),
                nn.GELU(),
                nn.Conv2d(dim // 2, dim, kernel_size=patch_size, stride=stride),
            )

        self.patch_embed1 = build_patch_embed()
        self.patch_embed2 = build_patch_embed()

        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, dim))

        self.blocks1 = nn.ModuleList(
            [
                OmniBiMambaBlockPseudo(dim=dim, d_state=d_state, expand=2, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.blocks2 = nn.ModuleList(
            [
                OmniBiMambaBlockPseudo(dim=dim, d_state=d_state, expand=2, dropout=dropout)
                for _ in range(depth)
            ]
        )

        self.cross_attn = GatedCrossAttentionBlock(dim, num_heads=8, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

        self.head = nn.Linear(dim, patch_size * patch_size * num_classes)
        self.cnn_enhancement = PrecipitationEnhancementCNN(num_classes, num_classes)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        x1 = self.patch_embed1(img1).flatten(2).transpose(1, 2)
        x2 = self.patch_embed2(img2).flatten(2).transpose(1, 2)

        x1 = x1 + self.pos_embed1
        x2 = x2 + self.pos_embed2

        current_h, current_w = self.num_patches_h, self.num_patches_w

        for blk in self.blocks1:
            x1 = blk(x1, current_h, current_w)
        for blk in self.blocks2:
            x2 = blk(x2, current_h, current_w)

        x_fuse = self.cross_attn(x1, x2)
        x_fuse = self.norm(x_fuse)
        x_fuse = self.head(x_fuse)

        bsz = x_fuse.size(0)
        x_fuse = x_fuse.view(
            bsz,
            self.num_patches_h,
            self.num_patches_w,
            self.patch_size,
            self.patch_size,
            self.num_classes,
        )
        x_fuse = x_fuse.permute(0, 5, 1, 3, 2, 4).contiguous()
        x_fuse = x_fuse.view(
            bsz,
            self.num_classes,
            self.num_patches_h * self.patch_size,
            self.num_patches_w * self.patch_size,
        )

        x_raw = torch.nn.functional.interpolate(
            x_fuse, size=(self.img_size, self.img_size_w), mode="bilinear", align_corners=False
        )
        x_final = self.cnn_enhancement(x_raw)
        return x_final
