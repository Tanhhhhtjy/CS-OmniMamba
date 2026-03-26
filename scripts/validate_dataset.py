"""Quick dataset validation script."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import RainDataset
from src.config import RADAR_PREP_DIR, PWV_DIR, RAIN_DIR

ds = RainDataset(RADAR_PREP_DIR, PWV_DIR, RAIN_DIR)
print(f"Total samples: {len(ds)}")
sample = ds[0]
for k, v in sample.items():
    print(f"  {k}: {v.shape}")
