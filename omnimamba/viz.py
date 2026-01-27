from typing import Iterable, Optional

import matplotlib.pyplot as plt
import torch


def plot_losses(train: Iterable[float], val: Iterable[float], save_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(list(train), label="Train Loss")
    plt.plot(list(val), label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()


def plot_gate_history(history: Iterable[Optional[float]], save_dir: str) -> None:
    processed = []
    last = 0.5
    for h in history:
        if h is not None:
            last = h
        processed.append(last)

    plt.figure(figsize=(8, 4))
    plt.plot(processed)
    plt.title("Gate Value Evolution")
    plt.savefig(f"{save_dir}/gate_curve.png")
    plt.close()


def show_results(model, dataloader, device, epoch: int, save_dir: str, num_samples: int = 3) -> None:
    model.eval()
    try:
        img1, img2, targets = next(iter(dataloader))
    except StopIteration:
        return

    img1, img2 = img1.to(device), img2.to(device)
    with torch.no_grad():
        preds = model(img1, img2)

    img1 = img1.cpu()
    img2 = img2.cpu()
    preds = preds.cpu().clamp(0, 1)
    targets = targets.cpu()

    fig, axes = plt.subplots(num_samples, 8, figsize=(20, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    titles = [
        "Input PWV",
        "Input Radar",
        "Pred +1h",
        "Target +1h",
        "Pred +2h",
        "Target +2h",
        "Pred +3h",
        "Target +3h",
    ]

    for i in range(min(num_samples, img1.size(0))):
        axes[i, 0].imshow(img1[i].squeeze(), cmap="gray")
        axes[i, 1].imshow(img2[i].squeeze(), cmap="jet")

        for t in range(3):
            axes[i, 2 + t * 2].imshow(preds[i, t], cmap="jet", vmin=0, vmax=1)
            axes[i, 3 + t * 2].imshow(targets[i, t], cmap="jet", vmin=0, vmax=1)

        if i == 0:
            for col in range(8):
                axes[i, col].set_title(titles[col])

        for col in range(8):
            axes[i, col].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/vis_epoch_{epoch + 1}.png")
    plt.close()
