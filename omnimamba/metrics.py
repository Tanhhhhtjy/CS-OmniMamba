import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def mape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((pred - target) / (target + 1e-8))) * 100


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse_val = torch.mean((pred - target) ** 2)
    if mse_val > 0:
        return 20 * torch.log10(1.0 / torch.sqrt(mse_val))
    return torch.tensor(100.0, device=pred.device)


def ssim_simple(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu1 = torch.mean(pred)
    mu2 = torch.mean(target)
    sigma1_sq = torch.var(pred)
    sigma2_sq = torch.var(target)
    sigma12 = torch.mean((pred - mu1) * (target - mu2))
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
