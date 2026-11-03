
import pandas as pd
import numpy as np


def load_data():
    predictions = pd.read_csv(
        "reports/october_predictions.csv"
    )

    original = pd.read_csv(
        "data/train-test.csv"
    )

    original["date"] = pd.to_datetime(
        original["date"]
    )

    october = original[
        (original["date"] >= "2025-10-01")
        & (original["date"] < "2025-11-01")
    ].copy()

    # Match prediction results with original features
    df = predictions.merge(
        october[
            [
                "load_id",
                "market_index",
                "quote_signal",
            ]
        ],
        on="load_id",
        how="left",
        validate="one_to_one"
    )

    return df


def analyze_extreme_targets(df):
    print("\nEXTREME ACTUAL FREIGHT RATES")
    print("=" * 60)

    print(
        df.nlargest(20, "actual_rate")[
            [
                "pickup",
                "delivery",
                "distance",
                "equipment",
                "actual_rate",
                "predicted_rate",
                "absolute_error",
                "market_index",
                "quote_signal",
            ]
        ].to_string(index=False)
    )


def analyze_high_value_segments(df):
    print("\nHIGH-VALUE SHIPMENT ANALYSIS")
    print("=" * 60)

    bins = [0, 5000, 10000, 15000, 20000, np.inf]

    labels = [
        "Below 5K",
        "5K-10K",
        "10K-15K",
        "15K-20K",
        "Above 20K",
    ]

    df = df.copy()

    df["rate_segment"] = pd.cut(
        df["actual_rate"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    result = df.groupby(
        "rate_segment",
        observed=True
    ).agg(
        records=("actual_rate", "count"),
        actual_mean=("actual_rate", "mean"),
        predicted_mean=("predicted_rate", "mean"),
        MAE=("absolute_error", "mean"),
        mean_distance=("distance", "mean"),
        mean_market_index=("market_index", "mean"),
        mean_quote_signal=("quote_signal", "mean"),
    )

    print(result)


def analyze_feature_relationships(df):
    print("\nCORRELATION WITH ACTUAL RATE")
    print("=" * 60)

    columns = [
        "actual_rate",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        "predicted_rate",
        "absolute_error",
    ]

    print(
        df[columns].corr(
            method="spearman"
        )["actual_rate"].sort_values(
            ascending=False
        )
    )


def analyze_extreme_errors(df):
    print("\nLARGEST PREDICTION ERRORS")
    print("=" * 60)

    print(
        df.nlargest(20, "absolute_error")[
            [
                "pickup",
                "delivery",
                "distance",
                "equipment",
                "actual_rate",
                "predicted_rate",
                "market_index",
                "quote_signal",
                "absolute_error",
            ]
        ].to_string(index=False)
    )


def main():
    df = load_data()

    analyze_extreme_targets(df)
    analyze_high_value_segments(df)
    analyze_feature_relationships(df)
    analyze_extreme_errors(df)


if __name__ == "__main__":
    main()
