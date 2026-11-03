import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from feature_engineering import prepare_features

MODEL_PATH = "models/freight_rate_model.cbm"
VALIDATION_PATH = "data/validation.csv"
DECEMBER_PATH = "data/december-chart-inputs.csv"
VALIDATION_OUTPUT = "validation_predictions.csv"
DECEMBER_OUTPUT = "data/december-chart-inputs.csv"


def load_model() -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model


def predict_and_validate(
    model: CatBoostRegressor,
    df: pd.DataFrame,
    dataset_name: str,
) -> np.ndarray:
    X = prepare_features(df)

    expected_features = model.feature_names_
    missing = set(expected_features) - set(X.columns)
    if missing:
        raise ValueError(f"{dataset_name}: missing model features: {missing}")

    X = X[expected_features]
    predictions = np.asarray(model.predict(X), dtype=float)

    if np.any(np.isnan(predictions)) or not np.all(np.isfinite(predictions)):
        raise ValueError(f"{dataset_name}: predictions contain NaN or non-finite values.")
    if np.any(predictions <= 0):
        raise ValueError(f"{dataset_name}: predictions contain non-positive values.")

    return predictions


def main() -> None:
    print("Loading model...")
    model = load_model()
    print(f"Features: {len(model.feature_names_)}")

    print("\nValidation predictions...")
    validation_df = pd.read_csv(VALIDATION_PATH)
    print(f"Rows: {len(validation_df)}")

    val_predictions = predict_and_validate(model, validation_df, "validation")

    pd.DataFrame({
        "load_id": validation_df["load_id"],
        "predicted_rate": val_predictions,
    }).to_csv(VALIDATION_OUTPUT, index=False)

    print(f"Saved: {VALIDATION_OUTPUT}")
    print(f"Range: ${val_predictions.min():.2f} - ${val_predictions.max():.2f}")

    print("\nDecember predictions...")
    december_df = pd.read_csv(DECEMBER_PATH)
    print(f"Rows: {len(december_df)}")

    dec_predictions = predict_and_validate(model, december_df, "december")

    december_df["predicted_rate"] = dec_predictions
    december_df.to_csv(DECEMBER_OUTPUT, index=False)

    print(f"Saved: {DECEMBER_OUTPUT}")
    print(f"Range: ${dec_predictions.min():.2f} - ${dec_predictions.max():.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
