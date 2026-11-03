import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_predictions():
    return pd.read_csv("reports/october_predictions.csv")


def analyze_overall(df):
    print("\nOVERALL PERFORMANCE")
    print("-" * 40)
    print("MAE:", mean_absolute_error(df["actual_rate"], df["predicted_rate"]))
    print("RMSE:", np.sqrt(mean_squared_error(df["actual_rate"], df["predicted_rate"])))
    print("Mean percentage error:", df["percentage_error"].mean())


def analyze_equipment(df):
    print("\nEQUIPMENT BREAKDOWN")
    print("-" * 40)
    result = df.groupby("equipment").agg(
        records=("actual_rate", "count"),
        MAE=("absolute_error", "mean"),
        actual_mean=("actual_rate", "mean"),
        predicted_mean=("predicted_rate", "mean"),
    )
    print(result)


def analyze_distance(df):
    print("\nDISTANCE BREAKDOWN")
    print("-" * 40)
    result = df.groupby(
        pd.cut(df["distance"], bins=[0, 500, 1000, 1500, 2000, 3000, np.inf]),
        observed=True,
    ).agg(
        records=("actual_rate", "count"),
        MAE=("absolute_error", "mean"),
    )
    print(result)


def analyze_largest_errors(df):
    print("\nTOP 15 LARGEST ERRORS")
    print("-" * 40)
    columns = ["pickup", "delivery", "equipment", "distance", "actual_rate", "predicted_rate", "absolute_error"]
    print(df.nlargest(15, "absolute_error")[columns].to_string(index=False))


def main():
    df = load_predictions()
    analyze_overall(df)
    analyze_equipment(df)
    analyze_distance(df)
    analyze_largest_errors(df)


if __name__ == "__main__":
    main()
