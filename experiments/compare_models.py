"""
Compares three CatBoost variants on the October validation window:
  A: 15-feature baseline
  B: + market_index and quote_signal
  C: + log-transformed target
"""

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import prepare_features, CATEGORICAL_FEATURES


DATA_PATH = "data/train-test.csv"

MODEL_PARAMS = {
    "iterations": 1500,
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": 200,
}


def load_datasets():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train = df[df["date"] < "2025-10-01"].copy()
    valid = df[(df["date"] >= "2025-10-01") & (df["date"] < "2025-11-01")].copy()

    X_train = prepare_features(train)
    X_valid = prepare_features(valid)

    for feature in ["market_index", "quote_signal"]:
        X_train[feature] = train[feature].to_numpy()
        X_valid[feature] = valid[feature].to_numpy()

    return X_train, X_valid, train["posted_rate"], valid["posted_rate"]


def evaluate_predictions(name, y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)

    high_value_mask = y_true.to_numpy() >= 5000
    extreme_mask = y_true.to_numpy() >= 10000

    high_value_mae = (
        mean_absolute_error(y_true.to_numpy()[high_value_mask], predictions[high_value_mask])
        if high_value_mask.any() else np.nan
    )
    extreme_mae = (
        mean_absolute_error(y_true.to_numpy()[extreme_mask], predictions[extreme_mask])
        if extreme_mask.any() else np.nan
    )

    print(f"\n{name}")
    print("-" * 55)
    print(f"MAE:                  ${mae:,.2f}")
    print(f"RMSE:                 ${rmse:,.2f}")
    print(f"R2:                   {r2:.4f}")
    print(f"MAE >= $5K:           ${high_value_mae:,.2f}")
    print(f"MAE >= $10K:          ${extreme_mae:,.2f}")

    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
            "MAE_5K_plus": high_value_mae, "MAE_10K_plus": extreme_mae}


def train_and_evaluate(name, X_train, X_valid, y_train, y_valid, log_target=False):
    cat_features = [f for f in CATEGORICAL_FEATURES if f in X_train.columns]

    target_train = np.log1p(y_train) if log_target else y_train
    target_valid = np.log1p(y_valid) if log_target else y_valid

    model = CatBoostRegressor(**MODEL_PARAMS)
    model.fit(
        X_train, target_train,
        cat_features=cat_features,
        eval_set=(X_valid, target_valid),
        early_stopping_rounds=100,
        use_best_model=True,
    )

    predictions = model.predict(X_valid)
    if log_target:
        predictions = np.expm1(predictions)
    predictions = np.maximum(predictions, 0.01)

    print(f"Best iteration: {model.best_iteration_ + 1}")
    result = evaluate_predictions(name, y_valid, predictions)
    return model, predictions, result


def main():
    X_train, X_valid, y_train, y_valid = load_datasets()

    baseline_features = [c for c in X_train.columns if c not in ["market_index", "quote_signal"]]

    print("Training rows:", len(X_train))
    print("October validation rows:", len(X_valid))

    results = []

    _, _, result_a = train_and_evaluate(
        "A - Baseline 15 features",
        X_train[baseline_features].copy(),
        X_valid[baseline_features].copy(),
        y_train, y_valid,
    )
    results.append(result_a)

    _, _, result_b = train_and_evaluate(
        "B - Baseline + market_index + quote_signal",
        X_train.copy(), X_valid.copy(),
        y_train, y_valid,
    )
    results.append(result_b)

    _, _, result_c = train_and_evaluate(
        "C - Extended features + log target",
        X_train.copy(), X_valid.copy(),
        y_train, y_valid,
        log_target=True,
    )
    results.append(result_c)

    comparison = pd.DataFrame(results)
    comparison.to_csv("reports/model_comparison.csv", index=False)

    print("\nFINAL COMPARISON")
    print("=" * 75)
    print(comparison.to_string(index=False))
    print("\nSaved: reports/model_comparison.csv")


if __name__ == "__main__":
    main()
