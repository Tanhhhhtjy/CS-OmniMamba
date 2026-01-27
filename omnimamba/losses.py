import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralStructuralWeightedLoss(nn.Module):
    def __init__(
        self,
        w_mae: float = 1.0,
        w_fft: float = 0.1,
        w_ssim: float = 0.2,
        heavy_rain_boost: float = 6.0,
    ):
        super().__init__()
        self.w_mae = w_mae
        self.w_fft = w_fft
        self.w_ssim = w_ssim
        self.heavy_rain_boost = heavy_rain_boost

    def continuous_weight_l1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        weights = 1.0 + (target * self.heavy_rain_boost)
        return (weights * diff).mean()

    def fft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        loss_fft = torch.mean(torch.abs(pred_fft - target_fft))
        return loss_fft

    def ssim_loss_simple(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu1 = F.avg_pool2d(pred, 3, 1, 1)
        mu2 = F.avg_pool2d(target, 3, 1, 1)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1 - ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_mae = self.continuous_weight_l1(pred, target)
        loss_fft = self.fft_loss(pred, target)
        bsz, timesteps, height, width = pred.shape
        pred_flat = pred.view(-1, 1, height, width)
        target_flat = target.view(-1, 1, height, width)
        loss_ssim = self.ssim_loss_simple(pred_flat, target_flat)

        total_loss = self.w_mae * loss_mae + self.w_fft * loss_fft + self.w_ssim * loss_ssim
        return total_loss
