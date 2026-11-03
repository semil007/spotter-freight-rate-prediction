"""
Tests five feature variants across four chronological windows
to see if route-level historical aggregates improve on the baseline.

Fallback hierarchy for unseen routes:
  1. Equipment + route historical median
  2. Route historical median
  3. Equipment historical median
  4. Global historical median
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
REPORT_PATH = "reports/route_features_experiment.csv"

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
    ("2025-07-01", "2025-07-01", "2025-08-01", "July"),
    ("2025-08-01", "2025-08-01", "2025-09-01", "August"),
    ("2025-09-01", "2025-09-01", "2025-10-01", "September"),
    ("2025-10-01", "2025-10-01", "2025-11-01", "October"),
]

ROUTE_COLS = ["route_hist_median", "route_hist_mean"]
EQUIP_ROUTE_COLS = ["equip_route_hist_median"]
HAVERSINE_COLS = ["haversine_distance", "distance_minus_haversine"]

EXPERIMENTS = [
    ("A - Baseline 15-feat",     dict(add_route=False, add_equip_route=False, add_haversine=False)),
    ("B - + route median/mean",  dict(add_route=True,  add_equip_route=False, add_haversine=False)),
    ("C - + equip-route median", dict(add_route=False, add_equip_route=True,  add_haversine=False)),
    ("D - + route + equip-route",dict(add_route=True,  add_equip_route=True,  add_haversine=False)),
    ("E - Best agg + haversine", dict(add_route=True,  add_equip_route=True,  add_haversine=True)),
]


def build_historical_aggregates(train_df: pd.DataFrame) -> dict:
    df = train_df.copy()
    df["route"] = df["pickup"] + "_" + df["delivery"]
    df["equip_route"] = df["equipment"] + "_" + df["route"]

    route_agg = (
        df.groupby("route")["posted_rate"]
        .agg(["median", "mean"])
        .rename(columns={"median": "route_hist_median", "mean": "route_hist_mean"})
    )
    equip_route_agg = (
        df.groupby("equip_route")["posted_rate"].median().rename("equip_route_hist_median")
    )
    equip_agg = df.groupby("equipment")["posted_rate"].median().rename("equip_hist_median")

    return {
        "global_median": float(df["posted_rate"].median()),
        "route_agg": route_agg,
        "equip_route_agg": equip_route_agg,
        "equip_agg": equip_agg,
    }


def apply_historical_aggregates(df: pd.DataFrame, aggs: dict) -> pd.DataFrame:
    result = df.copy()
    result["route"] = result["pickup"] + "_" + result["delivery"]
    result["equip_route"] = result["equipment"] + "_" + result["route"]

    gm = aggs["global_median"]
    ra = aggs["route_agg"]
    era = aggs["equip_route_agg"]
    ea = aggs["equip_agg"]

    result["route_hist_median"] = (
        result["route"].map(ra["route_hist_median"])
        .fillna(result["equipment"].map(ea))
        .fillna(gm)
    )
    result["route_hist_mean"] = (
        result["route"].map(ra["route_hist_mean"])
        .fillna(result["equipment"].map(ea))
        .fillna(gm)
    )
    result["equip_route_hist_median"] = (
        result["equip_route"].map(era)
        .fillna(result["route"].map(ra["route_hist_median"]))
        .fillna(result["equipment"].map(ea))
        .fillna(gm)
    )
    return result


def haversine_distance(df: pd.DataFrame) -> pd.Series:
    lat1, lon1 = np.radians(df["pickup_lat"]), np.radians(df["pickup_lon"])
    lat2, lon2 = np.radians(df["delivery_lat"]), np.radians(df["delivery_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_features(train_df, valid_df, add_route=False, add_equip_route=False, add_haversine=False):
    aggs = build_historical_aggregates(train_df)
    train_aug = apply_historical_aggregates(train_df, aggs)
    valid_aug = apply_historical_aggregates(valid_df, aggs)

    X_train = prepare_features(train_aug)
    X_valid = prepare_features(valid_aug)

    extra_cols = []
    if add_route:
        extra_cols += ROUTE_COLS
    if add_equip_route:
        extra_cols += EQUIP_ROUTE_COLS
    if add_haversine:
        for df_aug, X in [(train_aug, X_train), (valid_aug, X_valid)]:
            X["haversine_distance"] = haversine_distance(df_aug).values
            X["distance_minus_haversine"] = df_aug["distance"].values - X["haversine_distance"].values
        extra_cols += HAVERSINE_COLS

    for col in extra_cols:
        if col in ROUTE_COLS + EQUIP_ROUTE_COLS:
            X_train[col] = train_aug[col].values
            X_valid[col] = valid_aug[col].values

    return X_train, X_valid


def run_model(X_train, X_valid, y_train, y_valid) -> dict:
    cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    model = CatBoostRegressor(**CATBOOST_PARAMS, verbose=0)
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        use_best_model=True,
    )

    preds = np.maximum(model.predict(X_valid), 0.01)
    y = y_valid.to_numpy()

    hv_mask = y >= 10000
    hv_mae = float(np.abs(y[hv_mask] - preds[hv_mask]).mean()) if hv_mask.any() else np.nan

    return {
        "MAE": mean_absolute_error(y, preds),
        "RMSE": np.sqrt(mean_squared_error(y, preds)),
        "R2": r2_score(y, preds),
        "MAPE": float(np.mean(np.abs(y - preds) / y) * 100),
        "HV_MAE": hv_mae,
        "best_iter": model.best_iteration_ + 1,
        "n_features": X_train.shape[1],
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    all_rows = []

    for train_end, val_start, val_end, window_label in WINDOWS:
        train_df = df[df["date"] < train_end].copy()
        valid_df = df[(df["date"] >= val_start) & (df["date"] < val_end)].copy()

        y_train = train_df["posted_rate"]
        y_valid = valid_df["posted_rate"]

        print(f"\n{'=' * 65}")
        print(f"Window: {window_label}  train={len(train_df):,}  val={len(valid_df):,}")
        print(f"{'Experiment':<35} {'Feat':>4} {'Iter':>5} {'MAE':>8} {'RMSE':>8} {'R2':>7} {'MAPE':>7}")
        print("-" * 65)

        for exp_name, kwargs in EXPERIMENTS:
            X_train, X_valid = build_features(train_df, valid_df, **kwargs)
            result = run_model(X_train, X_valid, y_train, y_valid)

            print(
                f"{exp_name:<35} {result['n_features']:>4} {result['best_iter']:>5} "
                f"${result['MAE']:>7,.0f} ${result['RMSE']:>7,.0f} "
                f"{result['R2']:>7.4f} {result['MAPE']:>6.2f}%"
            )

            all_rows.append({"window": window_label, "experiment": exp_name, **result})

    report = pd.DataFrame(all_rows)
    report.to_csv(REPORT_PATH, index=False)

    print(f"\n{'=' * 65}")
    print("AVERAGE MAE ACROSS ALL WINDOWS")
    print("-" * 65)
    summary = report.groupby("experiment")["MAE"].mean().sort_values()
    for exp, mae in summary.items():
        print(f"  {exp:<35}  ${mae:,.2f}")

    best = summary.idxmin()
    improvement = (summary["A - Baseline 15-feat"] - summary[best]) / summary["A - Baseline 15-feat"] * 100
    print(f"\nBest: {best}")
    print(f"Improvement over baseline: {improvement:.1f}%")
    print(f"\nSaved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
