import pandas as pd

from feature_engineering import (
    prepare_features,
    MODEL_FEATURES,
    CATEGORICAL_FEATURES,
)

train = pd.read_csv("data/train-test.csv")
validation = pd.read_csv("data/validation.csv")
december = pd.read_csv("data/december-chart-inputs.csv")

X_train = prepare_features(train)
X_validation = prepare_features(validation)
X_december = prepare_features(december)

print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("December:", X_december.shape)

# All datasets must have exactly the same feature schema
assert list(X_train.columns) == MODEL_FEATURES
assert list(X_validation.columns) == MODEL_FEATURES
assert list(X_december.columns) == MODEL_FEATURES

print("All feature pipelines passed!")