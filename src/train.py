from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from feature_engineering import (
    prepare_features,
    CATEGORICAL_FEATURES,
)


DATA_PATH = Path("data/train-test.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

EXPERIMENT_MODEL_PATH = MODEL_DIR / "experiment_model.cbm"
FINAL_MODEL_PATH = MODEL_DIR / "freight_rate_model.cbm"

METRICS_PATH = REPORT_DIR / "experiment_metrics.json"
OCTOBER_PREDICTIONS_PATH = REPORT_DIR / "october_predictions.csv"

RANDOM_SEED = 42

MODEL_PARAMS = {
    "iterations": 1500,
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "random_seed": RANDOM_SEED,
    "allow_writing_files": False,
}


def load_data() -> pd.DataFrame:
    """Load and validate the labeled training dataset."""
    df = pd.read_csv(DATA_PATH)

    required_columns = ["date", "posted_rate", "load_id"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Invalid dates detected.")
    if df["posted_rate"].isna().any():
        raise ValueError("Missing target values detected.")
    if not np.isfinite(df["posted_rate"]).all():
        raise ValueError("Non-finite target values detected.")
    if (df["posted_rate"] <= 0).any():
        raise ValueError("Non-positive target values detected.")

    return df.sort_values("date").reset_index(drop=True)


def chronological_split(df: pd.DataFrame):
    """
    Split into January-September training and October validation.
    """
    train_df = df[df["date"] < "2025-10-01"].copy()
    valid_df = df[(df["date"] >= "2025-10-01") & (df["date"] < "2025-11-01")].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Training or validation split is empty.")

    print("\nChronological split")
    print("-" * 40)
    print(f"Training:   {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"Validation: {valid_df['date'].min().date()} to {valid_df['date'].max().date()}")
    print("Training records:", len(train_df))
    print("Validation records:", len(valid_df))

    return train_df, valid_df


def prepare_datasets(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    """Apply feature engineering to both splits."""
    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)

    y_train = train_df["posted_rate"]
    y_valid = valid_df["posted_rate"]

    assert list(X_train.columns) == list(X_valid.columns)

    return X_train, X_valid, y_train, y_valid


def build_model(iterations: int = 1500) -> CatBoostRegressor:
    params = MODEL_PARAMS.copy()
    params["iterations"] = iterations
    return CatBoostRegressor(**params)


def evaluate_model(y_true: pd.Series, predictions: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R2."""
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def train_experiment():
    """
    Train on January-September, evaluate on October.
    Saves experiment_metrics.json with the selected iteration count.
    """
    print("\nLoading dataset...")
    df = load_data()

    train_df, valid_df = chronological_split(df)
    X_train, X_valid, y_train, y_valid = prepare_datasets(train_df, valid_df)

    print("\nFeatures:", X_train.shape[1])
    print("Categorical features:", CATEGORICAL_FEATURES)
    print("\nStarting training...")

    model = build_model()
    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        verbose=200,
    )

    predictions = model.predict(X_valid)
    metrics = evaluate_model(y_valid, predictions)

    best_iteration = model.get_best_iteration()
    if best_iteration < 0:
        best_iteration = model.tree_count_ - 1
    selected_iterations = best_iteration + 1

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(EXPERIMENT_MODEL_PATH))

    october_results = valid_df[["load_id", "date", "pickup", "delivery", "equipment", "distance", "weight"]].copy()
    october_results["actual_rate"] = y_valid.to_numpy()
    october_results["predicted_rate"] = predictions
    october_results["error"] = october_results["actual_rate"] - october_results["predicted_rate"]
    october_results["absolute_error"] = october_results["error"].abs()
    october_results["percentage_error"] = (october_results["absolute_error"] / october_results["actual_rate"]) * 100
    october_results.to_csv(OCTOBER_PREDICTIONS_PATH, index=False)

    metrics["best_iteration"] = int(best_iteration)
    metrics["selected_iterations"] = int(selected_iterations)

    high_value_mask = y_valid > 10000
    high_value_count = high_value_mask.sum()
    if high_value_count > 0:
        hv_mae = float(np.abs(y_valid[high_value_mask].to_numpy() - predictions[high_value_mask]).mean())
        metrics["high_value_count"] = int(high_value_count)
        metrics["high_value_mae"] = hv_mae
        print(f"\nHigh-value loads (>$10K): {high_value_count}")
        print(f"High-value MAE: ${hv_mae:.2f}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nOctober results")
    print("-" * 40)
    print(f"MAE:  ${metrics['mae']:.2f}")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print("Selected iterations:", selected_iterations)
    print("\nArtifacts saved.")

    return metrics


def train_final():
    """
    Retrain on all 48,000 labeled records using the iteration count
    from the experiment phase. Requires experiment_metrics.json to exist.
    """
    print("\nStarting final training...")

    df = load_data()
    X_full = prepare_features(df)
    y_full = df["posted_rate"]

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "reports/experiment_metrics.json not found. Run train_experiment() first."
        )

    with open(METRICS_PATH, "r") as f:
        experiment_metrics = json.load(f)

    selected_iterations = experiment_metrics["selected_iterations"]

    print("Training records:", len(X_full))
    print("Iterations:", selected_iterations)

    final_model = build_model(iterations=selected_iterations)
    final_model.fit(
        X_full,
        y_full,
        cat_features=CATEGORICAL_FEATURES,
        verbose=200,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(FINAL_MODEL_PATH))

    print("\nFinal model saved:", FINAL_MODEL_PATH)

    return final_model


if __name__ == "__main__":
    # Runs final training by default.
    # To rerun the October experiment, call train_experiment() here instead.
    train_final()
