"""Step 0: download the Titanic CSV from Google Drive if it is not already on disk.

Run:  python -m src.download_data
Use --force to re-download even if the file exists.

The Drive file must be shared as "Anyone with the link can view".
"""
import argparse
import sys

import pandas as pd

from src import config

DRIVE_FILE_ID = "14cE-w5ilE2zMPz4cPVaxQLkBPSTQJLfT"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
EXPECTED_COLUMNS = {"PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp",
                    "Parch", "Ticket", "Fare", "Cabin", "Embarked"}


def validate(path) -> None:
    """Make sure we really got the CSV (Drive sometimes returns an HTML error page instead)."""
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing or len(df) < 100:
        raise ValueError(f"Downloaded file does not look like the Titanic data (missing: {missing}).")
    print(f"Data OK: {len(df)} rows, {len(df.columns)} columns.")


def main(force: bool = False) -> None:
    if config.RAW_DATA.exists() and not force:
        print(f"{config.RAW_DATA.name} already present - skipping download.")
        validate(config.RAW_DATA)
        return

    import gdown  # imported here so the script still works offline when the file exists

    config.RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset from Google Drive...")
    result = gdown.download(DRIVE_URL, str(config.RAW_DATA), quiet=False)
    if result is None:
        sys.exit("Download failed. Check that the Drive file is shared with 'Anyone with the link', "
                 "or place titanic.csv in data/raw/ manually.")
    validate(config.RAW_DATA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args().force)
