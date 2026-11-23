from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "Outputs"
CLEANED_FILE = OUTPUT_DIR / "cleaned_dataset.csv"
TRANSFORMED_FILE = OUTPUT_DIR / "transformed_dataset.csv"
TIME_COLUMNS = ["Arrival_time", "Departure_Time"]


def prepare_time_series(series: pd.Series) -> pd.Series:
    """Convert HH:MM:SS strings to timedeltas while treating 00:00:00 as a terminus marker."""
    text_series = series.fillna("").astype(str).str.strip()
    valid_text = text_series.where(text_series.ne(""), pd.NA)
    valid_text = valid_text.mask(valid_text.eq("00:00:00"), pd.NA)
    return pd.to_timedelta(valid_text, errors="coerce")


def calculate_journey_duration(group: pd.DataFrame) -> float:
    """Compute train journey duration in minutes from first departure to last arrival."""
    first_departure = group["Departure_timedelta"].dropna().iloc[0] if group["Departure_timedelta"].notna().any() else pd.NaT
    last_arrival = group["Arrival_timedelta"].dropna().iloc[-1] if group["Arrival_timedelta"].notna().any() else pd.NaT

    if pd.isna(first_departure) or pd.isna(last_arrival):
        return float("nan")

    if last_arrival < first_departure:
        last_arrival = last_arrival + pd.Timedelta(days=1)

    duration = (last_arrival - first_departure).total_seconds() / 60
    return float(duration)


def main() -> None:
    """Create transformed time columns and train-level journey durations."""
    dataframe = pd.read_csv(CLEANED_FILE, keep_default_na=False)
    dataframe = dataframe.sort_values(["Train_No", "SN"], ascending=[True, True]).reset_index(drop=True)

    for column in TIME_COLUMNS:
        column_text = dataframe[column].fillna("").astype(str).str.strip()
        dataframe[column] = column_text.where(column_text.eq(""), column_text.str.zfill(8))

    dataframe["Arrival_timedelta"] = prepare_time_series(dataframe["Arrival_time"])
    dataframe["Departure_timedelta"] = prepare_time_series(dataframe["Departure_Time"])

    journey_duration_by_train = dataframe.groupby("Train_No").apply(calculate_journey_duration)
    dataframe["Journey_Duration_min"] = dataframe["Train_No"].map(journey_duration_by_train)

    dataframe.to_csv(TRANSFORMED_FILE, index=False)
    print(f"Transformed dataset saved to: {TRANSFORMED_FILE}")
    print("\nTransformed dataset preview:")
    print(dataframe.head())


if __name__ == "__main__":
    main()
