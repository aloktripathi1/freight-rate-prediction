from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Paths / configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "train-test.csv"
EXP_DIR = ROOT / "experiments" / "EXP-007"

EXP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

SPLIT_DATE = pd.Timestamp("2025-10-01")
VAL_END_DATE = pd.Timestamp("2025-11-01")

TARGET = "posted_rate"
RPM_TARGET = "posted_rate_per_mile"

ID_COL = "load_id"
DATE_COL = "date"
DISTANCE_COL = "distance"


# ============================================================
# Base features
# ============================================================

BASE_FEATURE_COLUMNS = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]


# ============================================================
# Create temporal features
# ============================================================

def add_temporal_features(df):
    df = df.copy()

    start_date = pd.Timestamp("2025-01-01")

    df["month"] = df[DATE_COL].dt.month
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["day_of_month"] = df[DATE_COL].dt.day
    df["day_of_year"] = df[DATE_COL].dt.dayofyear

    df["days_since_start"] = (
        df[DATE_COL] - start_date
    ).dt.days

    return df


# These are explicit calendar features.
TEMPORAL_FEATURE_COLUMNS = [
    "month",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "days_since_start",
]


# Date itself is NOT passed to the model.
MODEL_BASE_FEATURE_COLUMNS = [
    c
    for c in BASE_FEATURE_COLUMNS
    if c != DATE_COL
]

MODEL_FEATURE_COLUMNS = (
    MODEL_BASE_FEATURE_COLUMNS
    + TEMPORAL_FEATURE_COLUMNS
)


CATEGORICAL_COLUMNS = [
    "pickup",
    "delivery",
    "equipment",
]

NUMERICAL_COLUMNS = [
    c
    for c in MODEL_FEATURE_COLUMNS
    if c not in CATEGORICAL_COLUMNS
]


# ============================================================
# Load data
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL]
)

print(
    f"Full development dataset: {df.shape}"
)


# ============================================================
# Data-quality correction
# ============================================================

negative_weight_count = int(
    (df["weight"] < 0).sum()
)

print(
    f"\nNegative weights found: "
    f"{negative_weight_count}"
)

df.loc[
    df["weight"] < 0,
    "weight",
] = np.nan

print(
    "Negative weights after correction:",
    int((df["weight"] < 0).sum()),
)


# ============================================================
# Distance validation
# ============================================================

if (
    df[DISTANCE_COL] <= 0
).any():
    raise ValueError(
        "Distance must be strictly positive."
    )


# ============================================================
# Create temporal features
# ============================================================

df = add_temporal_features(df)

print("\nTemporal feature ranges:")

for column in TEMPORAL_FEATURE_COLUMNS:
    print(
        f"  {column}: "
        f"{df[column].min()} → "
        f"{df[column].max()}"
    )


# ============================================================
# Create RPM target
# ============================================================

df[RPM_TARGET] = (
    df[TARGET]
    / df[DISTANCE_COL]
)


# ============================================================
# Chronological split
# ============================================================

train_df = df[
    df[DATE_COL] < SPLIT_DATE
].copy()

val_df = df[
    (df[DATE_COL] >= SPLIT_DATE)
    & (df[DATE_COL] < VAL_END_DATE)
].copy()

print(
    f"\nTraining:   {train_df.shape} "
    f"({train_df[DATE_COL].min().date()} → "
    f"{train_df[DATE_COL].max().date()})"
)

print(
    f"Validation: {val_df.shape} "
    f"({val_df[DATE_COL].min().date()} → "
    f"{val_df[DATE_COL].max().date()})"
)


# ============================================================
# X / y
# ============================================================

X_train = train_df[
    MODEL_FEATURE_COLUMNS
]

y_train = train_df[
    RPM_TARGET
]

X_val = val_df[
    MODEL_FEATURE_COLUMNS
]

y_val = val_df[
    RPM_TARGET
]


# ============================================================
# Preprocessing
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            NUMERICAL_COLUMNS,
        ),
        (
            "cat",
            categorical_pipeline,
            CATEGORICAL_COLUMNS,
        ),
    ]
)


# ============================================================
# Model
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            model,
        ),
    ]
)


# ============================================================
# Training
# ============================================================

print("\nTraining EXP-007...")

start_time = time.time()

pipeline.fit(
    X_train,
    y_train,
)

training_time = (
    time.time() - start_time
)

print(
    f"Training time: "
    f"{training_time:.2f} sec"
)


# ============================================================
# Predict RPM
# ============================================================

train_rpm_pred = pipeline.predict(
    X_train
)

val_rpm_pred = pipeline.predict(
    X_val
)


# ============================================================
# Reconstruct posted rate
# ============================================================

train_pred = (
    train_rpm_pred
    * train_df[DISTANCE_COL].to_numpy()
)

val_pred = (
    val_rpm_pred
    * val_df[DISTANCE_COL].to_numpy()
)

train_actual = (
    train_df[TARGET].to_numpy()
)

val_actual = (
    val_df[TARGET].to_numpy()
)


# ============================================================
# Metrics
# ============================================================

def regression_metrics(y_true, y_pred):
    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


train_metrics = regression_metrics(
    train_actual,
    train_pred,
)

val_metrics = regression_metrics(
    val_actual,
    val_pred,
)


# ============================================================
# Generalization gap
# ============================================================

mae_gap = (
    val_metrics["mae"]
    - train_metrics["mae"]
)

rmse_gap = (
    val_metrics["rmse"]
    - train_metrics["rmse"]
)

r2_gap = (
    train_metrics["r2"]
    - val_metrics["r2"]
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print("\nTraining:")

print(
    f"  MAE :  "
    f"{train_metrics['mae']:.4f}"
)

print(
    f"  RMSE:  "
    f"{train_metrics['rmse']:.4f}"
)

print(
    f"  R²  :  "
    f"{train_metrics['r2']:.4f}"
)

print("\nInternal Validation:")

print(
    f"  MAE :  "
    f"{val_metrics['mae']:.4f}"
)

print(
    f"  RMSE:  "
    f"{val_metrics['rmse']:.4f}"
)

print(
    f"  R²  :  "
    f"{val_metrics['r2']:.4f}"
)

print("\nGeneralization gap:")

print(
    f"  MAE gap : "
    f"{mae_gap:.4f}"
)

print(
    f"  RMSE gap: "
    f"{rmse_gap:.4f}"
)

print(
    f"  R² gap  : "
    f"{r2_gap:.4f}"
)


# ============================================================
# Save model
# ============================================================

joblib.dump(
    pipeline,
    EXP_DIR / "model_pipeline.joblib",
)


# ============================================================
# Save validation predictions
# ============================================================

predictions = pd.DataFrame(
    {
        "load_id": val_df[ID_COL].values,

        "predicted_rate": val_pred,

        "actual_rate": val_actual,

        "predicted_rate_per_mile": (
            val_rpm_pred
        ),

        "actual_rate_per_mile": (
            val_actual
            / val_df[DISTANCE_COL].to_numpy()
        ),
    }
)

predictions.to_csv(
    EXP_DIR / "validation_predictions.csv",
    index=False,
)


# ============================================================
# Save metrics
# ============================================================

metrics = {
    "experiment": "EXP-007",

    "description": (
        "Rate-per-mile target with "
        "calendar/time features"
    ),

    "seed": SEED,

    "split_date": str(
        SPLIT_DATE.date()
    ),

    "validation_end_date": str(
        VAL_END_DATE.date()
    ),

    "train_rows": int(
        len(train_df)
    ),

    "validation_rows": int(
        len(val_df)
    ),

    "negative_weights_corrected": (
        negative_weight_count
    ),

    "training_time_seconds": (
        float(training_time)
    ),

    "temporal_features": (
        TEMPORAL_FEATURE_COLUMNS
    ),

    "train_metrics_posted_rate": (
        train_metrics
    ),

    "validation_metrics_posted_rate": (
        val_metrics
    ),

    "generalization_gap": {
        "mae": float(mae_gap),
        "rmse": float(rmse_gap),
        "r2": float(r2_gap),
    },
}


with open(
    EXP_DIR / "metrics.json",
    "w",
) as f:
    json.dump(
        metrics,
        f,
        indent=2,
    )


# ============================================================
# Save configuration
# ============================================================

config = {
    "experiment": "EXP-007",

    "original_target": TARGET,

    "model_target": {
        "name": RPM_TARGET,
        "formula": (
            "posted_rate / distance"
        ),
    },

    "prediction_reconstruction": (
        "predicted_rate_per_mile * distance"
    ),

    "date_usage": {
        "for_split": True,
        "raw_date_as_model_feature": False,
    },

    "temporal_features": (
        TEMPORAL_FEATURE_COLUMNS
    ),

    "id_column": ID_COL,
    "date_column": DATE_COL,

    "categorical_columns": (
        CATEGORICAL_COLUMNS
    ),

    "numerical_columns": (
        NUMERICAL_COLUMNS
    ),

    "preprocessing": {
        "numeric_imputation": "median",
        "categorical_imputation": (
            "most_frequent"
        ),
        "one_hot_handle_unknown": (
            "ignore"
        ),
        "negative_weight_handling": (
            "weight < 0 -> NaN"
        ),
    },

    "model": {
        "type": "RandomForestRegressor",
        "n_estimators": 100,
        "min_samples_leaf": 5,
        "random_state": SEED,
        "n_jobs": -1,
    },
}


with open(
    EXP_DIR / "config.json",
    "w",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )


print("\n" + "=" * 70)
print("EXP-007 complete.")
print("=" * 70)

print(
    f"Artifacts saved to: "
    f"{EXP_DIR}"
)
