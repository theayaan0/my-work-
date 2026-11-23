from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "train_schedule.csv"
OUTPUT_DIR = PROJECT_ROOT / "Outputs"
BACKUP_FILE = OUTPUT_DIR / "raw_dataset_backup.csv"


def main() -> None:
    """Load the raw dataset, print basic inspection details, and save a backup."""
    dataframe = pd.read_csv(INPUT_FILE)

    print("Dataset shape (rows, columns):")
    print(dataframe.shape)
    print("\nColumn names:")
    print(list(dataframe.columns))
    print("\nColumn data types:")
    print(dataframe.dtypes)
    print("\nFirst 5 rows:")
    print(dataframe.head())

    dataframe.to_csv(BACKUP_FILE, index=False)
    print(f"\nRaw dataset backup saved to: {BACKUP_FILE}")


if __name__ == "__main__":
    main()
