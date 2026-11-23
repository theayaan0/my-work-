from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "Outputs"
TRANSFORMED_FILE = OUTPUT_DIR / "transformed_dataset.csv"
STOPS_FILE = OUTPUT_DIR / "train_stops_table.csv"
DISTANCE_FILE = OUTPUT_DIR / "train_distance_table.csv"
RESULTS_FILE = OUTPUT_DIR / "decision_tree_results.csv"


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


def plot_feature_importances(model: DecisionTreeRegressor, feature_names: list[str]) -> None:
    """Plot the fitted feature importances for the decision tree model."""
    importance_frame = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_frame, x="Feature", y="Importance", palette="viridis")
    plt.title("Decision Tree Feature Importances")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Build the train-level machine learning dataset used for modeling.
    sns.set_style("whitegrid")
    ml_dataframe = build_ml_dataframe()

    # Split the selected features and target into training and testing sets.
    features = ml_dataframe[["Number_of_Stops", "Total_Distance"]]
    target = ml_dataframe["Journey_Duration_min"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Train the decision tree regressor with the requested depth and seed.
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(x_train, y_train)

    # Generate predictions and compute the requested evaluation metrics.
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    print(f"Decision Tree R^2 score on training set: {train_r2:.4f}")
    print(f"Decision Tree R^2 score on test set: {test_r2:.4f}")
    print(f"Decision Tree Mean Absolute Error on test set: {test_mae:.4f}")

    # Save actual and predicted values for the test split.
    results = ml_dataframe.loc[x_test.index, ["Train_No"]].copy()
    results = results.join(x_test)
    results["Actual_Journey_Duration_min"] = y_test.values
    results["Predicted_Journey_Duration_min"] = test_predictions
    results.to_csv(RESULTS_FILE, index=False)

    # Plot the model's feature importance scores.
    plot_feature_importances(model, ["Number_of_Stops", "Total_Distance"])

    print(f"Decision tree results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
