"""Project paths. Licensed raw data lives OUTSIDE the repo (DATA_DIR)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SKIN_DATA_DIR", Path.home() / "data" / "skin-cancer"))
ISIC2024 = DATA_DIR / "isic2024"
REPORTS = ROOT / "reports"
