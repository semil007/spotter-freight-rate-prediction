import numpy as np
import pandas as pd


# Columns required in every input dataset (training, validation, December)
BASE_FEATURES = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
    "distance_bin",
]

MODEL_FEATURES = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "route",
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "day_sin",
    "day_cos",
    "weight_missing",
    "weight_per_mile",
    "distance_bin",
]


def create_route_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine pickup and delivery into a single route string.

    Example: Lexington + Fort Wayne -> Lexington_Fort Wayne
    """
    result = df.copy()

    required = ["pickup", "delivery"]

    if not all(col in result.columns for col in required):
        raise ValueError("Pickup and delivery columns are required.")

    result["route"] = (
        result["pickup"].astype("string")
        + "_"
        + result["delivery"].astype("string")
    )

    return result


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the shipment date.
    """
    result = df.copy()

    if "date" not in result.columns:
        raise ValueError("Date column is required.")

    dates = pd.to_datetime(result["date"], errors="coerce")

    if dates.isna().any():
        raise ValueError("Invalid or missing dates detected.")

    result["month"] = dates.dt.month
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_year"] = dates.dt.dayofyear

    result["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding so day 365 and day 1 are treated as adjacent
    result["day_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["day_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)

    return result


def create_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Straight-line distance between pickup and delivery using the Haversine formula.

    Not used in the shared model because the December dataset has no coordinates.
    """
    result = df.copy()

    required = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

    if not all(col in result.columns for col in required):
        raise ValueError("All four coordinate columns are required.")

    lat1 = np.radians(result["pickup_lat"])
    lon1 = np.radians(result["pickup_lon"])
    lat2 = np.radians(result["delivery_lat"])
    lon2 = np.radians(result["delivery_lon"])

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )

    result["haversine_distance"] = 3958.8 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return result


def create_shipment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive shipment features from weight and distance.
    """
    result = df.copy()

    required = ["weight", "distance"]

    if not all(col in result.columns for col in required):
        raise ValueError("Weight and distance columns are required.")

    # NaN weight stays NaN so the missing indicator works correctly
    result["weight_per_mile"] = result["weight"] / result["distance"].replace(0, np.nan)

    result["distance_bin"] = pd.cut(
        result["distance"],
        bins=[-np.inf, 250, 500, 1000, 1500, 2000, 3000, np.inf],
        labels=["very_short", "short", "medium", "medium_long", "long", "very_long", "extreme"],
    ).astype("string")

    return result


def create_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag missing numeric values so the model can distinguish absent from zero.
    """
    result = df.copy()

    if "weight" not in result.columns:
        raise ValueError("Weight column is required.")

    result["weight_missing"] = result["weight"].isna().astype(int)

    return result


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Returns a DataFrame with exactly the columns in MODEL_FEATURES,
    in that order. The same function is called during training and inference
    so the feature schema is always consistent.
    """
    missing_columns = [col for col in BASE_FEATURES if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")

    result = df.copy()

    result = create_route_feature(result)
    result = create_date_features(result)
    result = create_shipment_features(result)
    result = create_missing_indicators(result)

    # Cast categoricals to string; missing values get a sentinel
    for col in CATEGORICAL_FEATURES:
        result[col] = (
            result[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return result[MODEL_FEATURES].copy()
