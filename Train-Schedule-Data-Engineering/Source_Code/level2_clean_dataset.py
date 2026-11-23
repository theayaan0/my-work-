from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "train_schedule.csv"
BACKUP_FILE = PROJECT_ROOT / "Outputs" / "raw_dataset_backup.csv"
OUTPUT_DIR = PROJECT_ROOT / "Outputs"
CLEANED_FILE = OUTPUT_DIR / "cleaned_dataset.csv"
CRITICAL_COLUMNS = [
    "SN",
    "Train_No",
    "Station_Code",
    "Station_Name",
    "Arrival_time",
    "Departure_Time",
    "Distance",
]
TIME_COLUMNS = ["Arrival_time", "Departure_Time"]


def choose_input_file() -> Path:
    """Use the backup file if it exists; otherwise fall back to the raw CSV."""
    return BACKUP_FILE if BACKUP_FILE.exists() else RAW_INPUT_FILE


def format_hhmmss(value: object) -> str:
    """Normalize time values to HH:MM:SS while preserving empty values."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    pieces = text.split(":")
    if len(pieces) == 3 and all(piece.strip().isdigit() for piece in pieces):
        hours, minutes, seconds = (int(piece) for piece in pieces)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    parsed = pd.to_timedelta(text, errors="coerce")
    if pd.isna(parsed):
        return text

    total_seconds = int(parsed.total_seconds()) % (24 * 3600)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    """Clean the dataset, standardize time strings, and save the result."""
    input_file = choose_input_file()
    dataframe = pd.read_csv(input_file)

    print("Missing values before cleaning:")
    print(dataframe.isnull().sum())

    object_columns = dataframe.select_dtypes(include=["object"]).columns.tolist()
    fillable_object_columns = [column for column in object_columns if column not in CRITICAL_COLUMNS]
    if fillable_object_columns:
        dataframe[fillable_object_columns] = dataframe[fillable_object_columns].fillna("")

    dataframe = dataframe.dropna(subset=CRITICAL_COLUMNS).copy()

    before_duplicates = len(dataframe)
    dataframe = dataframe.drop_duplicates().copy()
    removed_duplicates = before_duplicates - len(dataframe)
    print(f"\nRemoved fully duplicate rows: {removed_duplicates}")

    for column in TIME_COLUMNS:
        dataframe[column] = dataframe[column].replace({np.nan: ""}).map(format_hhmmss)

    print("\nMissing values after cleaning:")
    print(dataframe.isnull().sum())

    dataframe.to_csv(CLEANED_FILE, index=False)
    print(f"\nCleaned dataset saved to: {CLEANED_FILE}")
    print("Note: '00:00:00' values are preserved here and handled during transformation.")


if __name__ == "__main__":
    main()
