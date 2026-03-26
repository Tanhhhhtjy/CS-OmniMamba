"""Entry point for offline RADAR preprocessing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import preprocess_all
preprocess_all()
