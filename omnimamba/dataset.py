from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import TrainingConfig
from .data_match import SampleRecord


class TripleChannelDataset(Dataset):
    def __init__(
        self,
        folder1_paths: Sequence[str],
        folder2_paths: Sequence[str],
        target_paths_1h: Sequence[str],
        target_paths_2h: Sequence[str],
        target_paths_3h: Sequence[str],
        transform=None,
    ):
        self.folder1_paths = list(folder1_paths)
        self.folder2_paths = list(folder2_paths)
        self.targets = list(zip(target_paths_1h, target_paths_2h, target_paths_3h))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.folder1_paths)

    def __getitem__(self, idx: int):
        img1 = Image.open(self.folder1_paths[idx]).convert("L")
        img2 = Image.open(self.folder2_paths[idx]).convert("L")
        t1, t2, t3 = self.targets[idx]
        target_1h = Image.open(t1).convert("L")
        target_2h = Image.open(t2).convert("L")
        target_3h = Image.open(t3).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            target_1h = self.transform(target_1h)
            target_2h = self.transform(target_2h)
            target_3h = self.transform(target_3h)

        targets = torch.stack(
            [target_1h.squeeze(), target_2h.squeeze(), target_3h.squeeze()], dim=0
        )
        return img1, img2, targets


def build_transforms(cfg: TrainingConfig):
    return transforms.Compose(
        [
            transforms.Resize((cfg.img_size, cfg.img_size_w)),
            transforms.ToTensor(),
        ]
    )


def _records_to_paths(records: Sequence[SampleRecord]) -> Tuple[List[str], ...]:
    folder1, folder2, t1, t2, t3 = [], [], [], [], []
    for r in records:
        folder1.append(r.pwv_path)
        folder2.append(r.radar_path)
        t1.append(r.target_1h_path)
        t2.append(r.target_2h_path)
        t3.append(r.target_3h_path)
    return folder1, folder2, t1, t2, t3


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

    train_ds = TripleChannelDataset(*tr_paths, transform=tf)
    val_ds = TripleChannelDataset(*val_paths, transform=tf)
    test_ds = TripleChannelDataset(*test_paths, transform=tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader
