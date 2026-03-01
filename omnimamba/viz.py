from collections.abc import Iterable, Sequence
from typing import Any, cast
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch


def plot_losses(train: Iterable[float], val: Iterable[float], save_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(list(train), label="Train Loss")
    plt.plot(list(val), label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()


def plot_gate_history(history: Iterable[float | None], save_dir: str) -> None:
    processed = []
    last = 0.5
    for h in history:
        if h is not None:
            last = h
        processed.append(last)

    plt.figure(figsize=(8, 4))
    plt.plot(processed)
    plt.title("Gate Value Evolution")
    plt.savefig(os.path.join(save_dir, "gate_curve.png"))
    plt.close()


_VIZ_TIMESTAMPS = (
    "2023-08-04-06-00-00",
    "2023-08-05-00-00-00",
    "2023-08-06-11-00-00",
)


def _normalise_timestamp_name(name: str) -> str:
    base = os.path.basename(name)
    if base.endswith(".png"):
        base = base[:-4]
    return base


def _resolve_viz_indices(dataset: Any, timestamps: Sequence[str]) -> list[int]:
    if dataset is None:
        return []
    pwv_paths = getattr(dataset, "pwv_paths", None)
    if pwv_paths is None:
        return []
    requested = [_normalise_timestamp_name(t) for t in timestamps]
    path_to_index = {}
    for idx, path in enumerate(pwv_paths):
        path_to_index[_normalise_timestamp_name(path)] = idx
    indices = []
    for t in requested:
        if t in path_to_index:
            indices.append(path_to_index[t])
    return indices


def _get_pwv_cmap(discrete: bool = True) -> LinearSegmentedColormap:
    colors_rgb = [
        (0, 0, 139),
        (0, 50, 200),
        (0, 120, 255),
        (0, 180, 255),
        (0, 220, 200),
        (50, 255, 150),
        (150, 255, 50),
        (255, 255, 0),
        (255, 180, 0),
        (255, 100, 0),
        (220, 0, 0),
    ]
    colors_norm = [[c / 255.0 for c in rgb] for rgb in colors_rgb]
    n_colors = 11 if discrete else 512
    return LinearSegmentedColormap.from_list("custom_pwv", colors_norm, N=n_colors)


def _get_radar_cmap(discrete: bool = True) -> LinearSegmentedColormap:
    colors_rgb = [
        (255, 50, 10),
        (255, 100, 20),
        (255, 160, 40),
        (255, 220, 60),
        (200, 255, 100),
        (140, 255, 180),
        (100, 255, 220),
        (80, 220, 255),
        (60, 180, 240),
        (40, 120, 200),
        (25, 60, 140),
        (15, 30, 80),
        (5, 10, 30),
    ]
    colors_norm = [[c / 255.0 for c in rgb] for rgb in colors_rgb]
    n_colors = 13 if discrete else 512
    return LinearSegmentedColormap.from_list("custom_radar", colors_norm, N=n_colors)


def _get_rain_cmap(discrete: bool = True) -> LinearSegmentedColormap:
    colors_rgb = [
        (97, 40, 31),
        (250, 1, 246),
        (0, 0, 254),
        (101, 183, 252),
        (61, 185, 63),
        (166, 242, 142),
        (255, 255, 255),
    ]
    colors_norm = [[c / 255.0 for c in rgb] for rgb in colors_rgb]
    n_colors = 7 if discrete else 256
    return LinearSegmentedColormap.from_list("custom_rain", colors_norm, N=n_colors)


def show_results(
    model, dataloader, device, epoch: int, save_dir: str, num_samples: int = 3
) -> None:
    model.eval()

    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        try:
            img1, radar_seq, targets = next(iter(dataloader))
        except StopIteration:
            return
    else:
        dataset_any = cast(Any, dataset)
        indices = _resolve_viz_indices(dataset_any, _VIZ_TIMESTAMPS)
        if indices:
            img1_list, radar_list, target_list = [], [], []
            for idx in indices[:num_samples]:
                sample = cast(Any, dataset_any)[idx]
                img1_list.append(sample[0])
                radar_list.append(sample[1])
                target_list.append(sample[2])
            img1 = torch.stack(img1_list, dim=0)
            radar_seq = torch.stack(radar_list, dim=0)
            targets = torch.stack(target_list, dim=0)
        else:
            try:
                img1, radar_seq, targets = next(iter(dataloader))
            except StopIteration:
                return

    img1 = img1.to(device)
    radar_seq = radar_seq.to(device)
    with torch.no_grad():
        preds = model(img1, radar_seq)

    img1 = img1.cpu()
    # Use last radar frame for display
    if radar_seq.dim() == 5:
        img2_display = radar_seq[:, -1].cpu()  # [B, 1, H, W]
    else:
        img2_display = radar_seq.cpu()
    preds = preds.cpu().clamp(0, 1)
    targets = targets.cpu()

    pwv_cmap = _get_pwv_cmap(discrete=True)
    radar_cmap = _get_radar_cmap(discrete=True)
    rain_cmap = _get_rain_cmap(discrete=True)

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
        axes[i, 0].imshow(
            img1[i].squeeze(), cmap=pwv_cmap, vmin=0, vmax=1, interpolation="bicubic"
        )
        axes[i, 1].imshow(
            img2_display[i].squeeze(),
            cmap=radar_cmap,
            vmin=0,
            vmax=1,
            interpolation="bicubic",
        )

        for t in range(3):
            pred_show = preds[i, t].clone()
            pred_show[pred_show > 0.95] = 1.0
            axes[i, 2 + t * 2].imshow(
                pred_show, cmap=rain_cmap, vmin=0, vmax=1, interpolation="bicubic"
            )
            axes[i, 3 + t * 2].imshow(
                targets[i, t], cmap=rain_cmap, vmin=0, vmax=1, interpolation="bicubic"
            )

        if i == 0:
            for col in range(8):
                axes[i, col].set_title(titles[col])

        for col in range(8):
            axes[i, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_epoch_{epoch + 1}.png"))
    plt.close()
