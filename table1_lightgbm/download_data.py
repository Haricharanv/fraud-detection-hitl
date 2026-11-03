"""
download_data.py -- IEEE-CIS Fraud Detection Dataset Downloader
Samsung Capstone Project - Table 1 (Team LightGBM)
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COMPETITION = "ieee-fraud-detection"
ZIP_NAME = f"{COMPETITION}.zip"


def check_kaggle_credentials():
    """Return True if Kaggle API credentials are discoverable."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.is_file()


def print_setup_instructions():
    """Print setup instructions for Kaggle API."""
    print("""
+------------------------------------------------------------------+
|  Kaggle API credentials NOT found.                                |
|                                                                    |
|  To set up:                                                        |
|  1. Go to https://www.kaggle.com/settings -> API -> Create Token   |
|  2. Download kaggle.json                                           |
|  3. Place it at:                                                   |
|       Windows : %USERPROFILE%\\.kaggle\\kaggle.json                  |
|       Linux   : ~/.kaggle/kaggle.json                              |
|  4. Or set env vars KAGGLE_USERNAME and KAGGLE_KEY                 |
|                                                                    |
|  Also accept the competition rules at:                             |
|  https://www.kaggle.com/competitions/ieee-fraud-detection/rules    |
+------------------------------------------------------------------+
""")


def download_dataset():
    """Download the IEEE-CIS Fraud Detection dataset via Kaggle CLI."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] 'kaggle' package not installed. Run:  pip install kaggle")
        sys.exit(1)

    if not check_kaggle_credentials():
        print_setup_instructions()
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading '{COMPETITION}' dataset to {DATA_DIR} ...")
    api.competition_download_files(COMPETITION, path=str(DATA_DIR), quiet=False)
    print("[INFO] Download complete.")

    return DATA_DIR / ZIP_NAME


def extract_dataset(zip_path):
    """Extract the downloaded zip into data/ieee-fraud-detection/."""
    extract_dir = DATA_DIR / COMPETITION
    if extract_dir.exists():
        print(f"[INFO] Removing existing directory: {extract_dir}")
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracting {zip_path.name} -> {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("[INFO] Extraction complete.")

    zip_path.unlink()
    print(f"[INFO] Removed zip file: {zip_path.name}")

    files = sorted(extract_dir.iterdir())
    print(f"\n[INFO] Extracted {len(files)} files:")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"       {f.name:40s}  {size_mb:>8.1f} MB")


def main():
    """Orchestrate download and extraction."""
    target_dir = DATA_DIR / COMPETITION
    expected_file = target_dir / "train_transaction.csv"

    if expected_file.is_file():
        print(f"[INFO] Dataset already exists at {target_dir}")
        print("[INFO] Delete the directory and re-run to force re-download.")
        return

    zip_path = DATA_DIR / ZIP_NAME
    if zip_path.is_file():
        print(f"[INFO] Zip already downloaded at {zip_path}, skipping download.")
    else:
        zip_path = download_dataset()

    extract_dataset(zip_path)
    print("\n[SUCCESS] Dataset ready for Phase 1.")


if __name__ == "__main__":
    main()
