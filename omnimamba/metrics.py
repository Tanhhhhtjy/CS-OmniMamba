import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


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


def csi(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Critical Success Index (Threat Score).

    pred / target are assumed to be in [0, 1] (normalised pixel values).
    Threshold should be set relative to the same normalisation.
    CSI = TP / (TP + FP + FN),  range [0, 1], higher is better.
    """
    pred_bin = (pred >= threshold).float()
    tgt_bin = (target >= threshold).float()
    tp = (pred_bin * tgt_bin).sum()
    fp = (pred_bin * (1 - tgt_bin)).sum()
    fn = ((1 - pred_bin) * tgt_bin).sum()
    denom = tp + fp + fn
    return tp / (denom + 1e-8)


def ets(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Equitable Threat Score (Gilbert Skill Score).

    ETS = (TP - TP_rand) / (TP + FP + FN - TP_rand)
    where TP_rand = (TP+FP)*(TP+FN) / total
    Range [-1/3, 1], higher is better, 0 = no skill.
    """
    pred_bin = (pred >= threshold).float()
    tgt_bin = (target >= threshold).float()
    total = pred_bin.numel()
    tp = (pred_bin * tgt_bin).sum()
    fp = (pred_bin * (1 - tgt_bin)).sum()
    fn = ((1 - pred_bin) * tgt_bin).sum()
    tp_rand = (tp + fp) * (tp + fn) / (total + 1e-8)
    denom = tp + fp + fn - tp_rand
    return (tp - tp_rand) / (denom + 1e-8)
