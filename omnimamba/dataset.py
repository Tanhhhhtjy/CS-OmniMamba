import random as _random
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import TrainingConfig
from .data_match import SampleRecord


class TripleChannelDataset(Dataset):
    """Dataset yielding (pwv, radar_seq, targets).

    radar_seq shape: [T, 1, H, W]  where T = radar_seq_len

    When ``augment=True`` (training mode), random horizontal and vertical
    flips are applied **consistently** across all modalities so that spatial
    correspondence between PWV, radar and target is preserved.
    """

    def __init__(
        self,
        pwv_paths: Sequence[str],
        radar_seq_paths: Sequence[Sequence[str]],
        target_paths_1h: Sequence[str],
        target_paths_2h: Sequence[str],
        target_paths_3h: Sequence[str],
        transform=None,
        augment: bool = False,
    ):
        self.pwv_paths = list(pwv_paths)
        self.radar_seq_paths = list(radar_seq_paths)
        self.targets = list(zip(target_paths_1h, target_paths_2h, target_paths_3h))
        self.transform = transform
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pwv_paths)

    @staticmethod
    def _apply_flips(tensor: torch.Tensor, hflip: bool, vflip: bool) -> torch.Tensor:
        """Flip a (..., H, W) tensor in-place consistently."""
        if hflip:
            tensor = torch.flip(tensor, [-1])
        if vflip:
            tensor = torch.flip(tensor, [-2])
        return tensor

    def __getitem__(self, idx: int):
        pwv = Image.open(self.pwv_paths[idx]).convert("L")
        t1, t2, t3 = self.targets[idx]
        target_1h = Image.open(t1).convert("L")
        target_2h = Image.open(t2).convert("L")
        target_3h = Image.open(t3).convert("L")

        # Load each radar frame in sequence
        radar_frames = [
            Image.open(p).convert("L") for p in self.radar_seq_paths[idx]
        ]

        if self.transform:
            pwv = self.transform(pwv)
            target_1h = self.transform(target_1h)
            target_2h = self.transform(target_2h)
            target_3h = self.transform(target_3h)
            radar_frames = [self.transform(f) for f in radar_frames]

        # radar_seq: [T, 1, H, W]
        radar_seq = torch.stack(radar_frames, dim=0)

        targets = torch.stack(
            [target_1h.squeeze(), target_2h.squeeze(), target_3h.squeeze()], dim=0
        )

        # Synchronized augmentation: same flip for every modality
        if self.augment:
            hflip = _random.random() < 0.5
            vflip = _random.random() < 0.5
            if hflip or vflip:
                pwv = self._apply_flips(pwv, hflip, vflip)
                targets = self._apply_flips(targets, hflip, vflip)
                radar_seq = self._apply_flips(radar_seq, hflip, vflip)

        return pwv, radar_seq, targets


def build_transforms(cfg: TrainingConfig):
    return transforms.Compose(
        [
            transforms.Resize((cfg.img_size, cfg.img_size_w)),
            transforms.ToTensor(),
        ]
    )


def _records_to_paths(records: Sequence[SampleRecord]) -> Tuple:
    pwv, radar_seqs, t1, t2, t3 = [], [], [], [], []
    for r in records:
        pwv.append(r.pwv_path)
        radar_seqs.append(r.radar_seq_paths)
        t1.append(r.target_1h_path)
        t2.append(r.target_2h_path)
        t3.append(r.target_3h_path)
    return pwv, radar_seqs, t1, t2, t3


def build_loaders(
    train_records: Sequence[SampleRecord],
    val_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
    cfg: TrainingConfig,
):
    tf = build_transforms(cfg)
    tr_paths = _records_to_paths(train_records)
    val_paths = _records_to_paths(val_records)
    test_paths = _records_to_paths(test_records)

    # Training set gets random H/V flip augmentation; val/test stay deterministic
    train_ds = TripleChannelDataset(*tr_paths, transform=tf, augment=True)
    val_ds = TripleChannelDataset(*val_paths, transform=tf, augment=False)
    test_ds = TripleChannelDataset(*test_paths, transform=tf, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.num_workers > 0,
    )

    return train_loader, val_loader, test_loader
