"""
Compares three models across four chronological validation windows:
  Baseline 1: training median (constant predictor)
  Baseline 2: linear regression on distance only
  Candidate:  15-feature CatBoost
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import prepare_features, CATEGORICAL_FEATURES


DATA_PATH = "data/train-test.csv"

CATBOOST_PARAMS = {
    "iterations": 1500,
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "random_seed": 42,
    "allow_writing_files": False,
}

WINDOWS = [
    ("2025-07-01", "2025-07-01", "2025-08-01", "July (Jan-Jun train)"),
    ("2025-08-01", "2025-08-01", "2025-09-01", "August (Jan-Jul train)"),
    ("2025-09-01", "2025-09-01", "2025-10-01", "September (Jan-Aug train)"),
    ("2025-10-01", "2025-10-01", "2025-11-01", "October (Jan-Sep train)"),
]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs(y_true - y_pred) / y_true) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def run_window(df: pd.DataFrame, train_end: str, val_start: str, val_end: str):
    train_df = df[df["date"] < train_end].copy()
    valid_df = df[(df["date"] >= val_start) & (df["date"] < val_end)].copy()

    y_train = train_df["posted_rate"].to_numpy()
    y_valid = valid_df["posted_rate"].to_numpy()

    # Baseline 1: training median
    b1 = metrics(y_valid, np.full(len(y_valid), np.median(y_train)))

    # Baseline 2: linear regression on distance only
    lr = LinearRegression().fit(train_df[["distance"]].fillna(0).to_numpy(), y_train)
    lr_pred = np.maximum(lr.predict(valid_df[["distance"]].fillna(0).to_numpy()), 0.01)
    b2 = metrics(y_valid, lr_pred)

    # Candidate: 15-feature CatBoost
    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)

    model = CatBoostRegressor(**CATBOOST_PARAMS, verbose=0)
    model.fit(
        X_train, y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        use_best_model=True,
    )
    cb_pred = np.maximum(model.predict(X_valid), 0.01)
    cb = metrics(y_valid, cb_pred)

    return {"train_rows": len(train_df), "val_rows": len(valid_df), "b1": b1, "b2": b2, "cb": cb}


def print_window_results(label: str, result: dict):
    print(f"\n{'=' * 60}")
    print(f"Window: {label}")
    print(f"Train: {result['train_rows']:,}  Val: {result['val_rows']:,}")
    print(f"{'Model':<38} {'MAE':>8} {'RMSE':>8} {'R2':>7} {'MAPE':>7}")
    print("-" * 60)
    for name, m in [
        ("Baseline 1 (training median)", result["b1"]),
        ("Baseline 2 (distance linear)", result["b2"]),
        ("Candidate (CatBoost 15-feat)", result["cb"]),
    ]:
        print(f"{name:<38} ${m['MAE']:>7,.0f} ${m['RMSE']:>7,.0f} {m['R2']:>7.4f} {m['MAPE']:>6.2f}%")


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    all_rows = []

    for train_end, val_start, val_end, label in WINDOWS:
        print(f"\nRunning: {label}...")
        result = run_window(df, train_end, val_start, val_end)
        print_window_results(label, result)

        for model_name, m in [
            ("Baseline 1 - training median", result["b1"]),
            ("Baseline 2 - distance linear", result["b2"]),
            ("Candidate - CatBoost 15-feat", result["cb"]),
        ]:
            all_rows.append({
                "window": label, "model": model_name,
                "train_rows": result["train_rows"], "val_rows": result["val_rows"],
                **m,
            })

    report = pd.DataFrame(all_rows)
    report.to_csv("reports/baseline_comparison.csv", index=False)

    print(f"\n{'=' * 60}")
    print("AVERAGE ACROSS ALL WINDOWS")
    print("-" * 60)
    print(report.groupby("model")[["MAE", "RMSE", "R2", "MAPE"]].mean().to_string())
    print("\nSaved: reports/baseline_comparison.csv")


if __name__ == "__main__":
    main()
