from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "Outputs"
TRANSFORMED_FILE = OUTPUT_DIR / "transformed_dataset.csv"
STOPS_FILE = OUTPUT_DIR / "train_stops_table.csv"
DISTANCE_FILE = OUTPUT_DIR / "train_distance_table.csv"
CROSS_TABLE_FILE = OUTPUT_DIR / "cross_table_trains_stations.csv"


def print_table_summary(table_name: str, dataframe: pd.DataFrame) -> None:
    """Print the shape and preview of a generated table."""
    print(f"\n{table_name} shape:")
    print(dataframe.shape)
    print(f"{table_name} preview:")
    print(dataframe.head())


def main() -> None:
    """Build train-level aggregation tables and save them as CSV files."""
    dataframe = pd.read_csv(TRANSFORMED_FILE)

    train_stops_table = (
        dataframe.groupby("Train_No", as_index=False)["SN"]
        .nunique()
        .rename(columns={"SN": "Number_of_Stops"})
    )
    train_stops_table.to_csv(STOPS_FILE, index=False)

    train_distance_table = (
        dataframe.groupby("Train_No", as_index=False)["Distance"]
        .max()
        .rename(columns={"Distance": "Total_Distance"})
    )
    train_distance_table.to_csv(DISTANCE_FILE, index=False)

    cross_table = pd.pivot_table(
        dataframe,
        index="Train_No",
        columns="Station_Code",
        values="Distance",
        aggfunc="max",
        fill_value=0,
    )
    cross_table.to_csv(CROSS_TABLE_FILE)

    print_table_summary("Train stops table", train_stops_table)
    print_table_summary("Train distance table", train_distance_table)
    print_table_summary("Cross table trains-stations", cross_table.reset_index())

    print(f"\nSaved: {STOPS_FILE}")
    print(f"Saved: {DISTANCE_FILE}")
    print(f"Saved: {CROSS_TABLE_FILE}")


if __name__ == "__main__":
    main()
