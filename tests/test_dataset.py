import tempfile
from pathlib import Path

import torch
from PIL import Image

from omnimamba.config import TrainingConfig
from omnimamba.dataset import TripleChannelDataset, build_transforms


def _write_png(path: Path) -> None:
    img = Image.new("L", (1, 1), color=128)
    img.save(path)


def test_dataset_shapes():
    cfg = TrainingConfig()
    tf = build_transforms(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        pwv = base / "pwv.png"
        radar = base / "radar.png"
        t1 = base / "t1.png"
        t2 = base / "t2.png"
        t3 = base / "t3.png"

        for p in (pwv, radar, t1, t2, t3):
            _write_png(p)

        ds = TripleChannelDataset(
            [str(pwv)],
            [str(radar)],
            [str(t1)],
            [str(t2)],
            [str(t3)],
            transform=tf,
        )

        img1, img2, targets = ds[0]
        assert img1.shape == (1, cfg.img_size, cfg.img_size_w)
        assert img2.shape == (1, cfg.img_size, cfg.img_size_w)
        assert targets.shape == (3, cfg.img_size, cfg.img_size_w)
        assert isinstance(img1, torch.Tensor)
