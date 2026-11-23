from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "Outputs"
TRANSFORMED_FILE = OUTPUT_DIR / "transformed_dataset.csv"
STOPS_FILE = OUTPUT_DIR / "train_stops_table.csv"
DISTANCE_FILE = OUTPUT_DIR / "train_distance_table.csv"
RESULTS_FILE = OUTPUT_DIR / "linear_regression_results.csv"


def build_ml_dataframe() -> pd.DataFrame:
    """Merge train-level features and target into one ML-ready table."""
    transformed = pd.read_csv(TRANSFORMED_FILE)
    stops_table = pd.read_csv(STOPS_FILE)
    distance_table = pd.read_csv(DISTANCE_FILE)

    journey_duration = transformed.groupby("Train_No", as_index=False)["Journey_Duration_min"].first()
    ml_dataframe = stops_table.merge(distance_table, on="Train_No", how="inner")
    ml_dataframe = ml_dataframe.merge(journey_duration, on="Train_No", how="inner")
    ml_dataframe = ml_dataframe.dropna(subset=["Number_of_Stops", "Total_Distance", "Journey_Duration_min"]).copy()
    return ml_dataframe


def plot_top_stops(stops_table: pd.DataFrame) -> None:
    """Plot the top 15 trains by number of stops."""
    top_stops = stops_table.sort_values("Number_of_Stops", ascending=False).head(15)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_stops, x="Train_No", y="Number_of_Stops", palette="Blues_d")
    plt.title("Top 15 Trains by Number of Stops")
    plt.xlabel("Train_No")
    plt.ylabel("Stop Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_journey_distribution(ml_dataframe: pd.DataFrame) -> None:
    """Plot the distribution of journey duration in minutes."""
    plt.figure(figsize=(10, 6))
    sns.histplot(ml_dataframe["Journey_Duration_min"], bins=30, kde=True, color="teal")
    plt.title("Distribution of Journey Duration (minutes)")
    plt.xlabel("Journey_Duration_min")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Train a linear regression model and save predicted vs actual values."""
    sns.set_style("whitegrid")
    ml_dataframe = build_ml_dataframe()

    features = ml_dataframe[["Number_of_Stops", "Total_Distance"]]
    target = ml_dataframe["Journey_Duration_min"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    r2 = r2_score(y_test, predictions)
    print(f"Linear Regression R^2 score on test set: {r2:.4f}")

    results = ml_dataframe.loc[x_test.index, ["Train_No"]].copy()
    results = results.join(x_test)
    results["Actual_Journey_Duration_min"] = y_test.values
    results["Predicted_Journey_Duration_min"] = predictions
    results.to_csv(RESULTS_FILE, index=False)

    plot_top_stops(pd.read_csv(STOPS_FILE))
    plot_journey_distribution(ml_dataframe)

    print(f"Linear regression results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
