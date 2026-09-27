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
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "train-test.csv"
EXP_DIR = ROOT / "experiments" / "EXP-008"

EXP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

SPLIT_DATE = pd.Timestamp("2025-10-01")
VAL_END_DATE = pd.Timestamp("2025-11-01")

TARGET = "posted_rate"
RPM_TARGET = "posted_rate_per_mile"
LOG_RPM_TARGET = "log_posted_rate_per_mile"

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
# Create explicit route feature
# ============================================================

def add_route_feature(df):
    df = df.copy()

    df["route"] = (
        df["pickup"].astype(str)
        + " -> "
        + df["delivery"].astype(str)
    )

    return df


df_route_feature = [
    "route",
]


# Date is used only for chronological splitting.
MODEL_FEATURE_COLUMNS = [
    c for c in BASE_FEATURE_COLUMNS
    if c != DATE_COL
] + df_route_feature


CATEGORICAL_COLUMNS = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]

NUMERICAL_COLUMNS = [
    c for c in MODEL_FEATURE_COLUMNS
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
# Validate distance
# ============================================================

if (
    df[DISTANCE_COL] <= 0
).any():
    raise ValueError(
        "Distance must be strictly positive."
    )


# ============================================================
# Create route
# ============================================================

df = add_route_feature(df)

print(
    f"\nUnique routes: "
    f"{df['route'].nunique():,}"
)


# ============================================================
# Create rate-per-mile target
# ============================================================

df[RPM_TARGET] = (
    df[TARGET]
    / df[DISTANCE_COL]
)

if (
    df[RPM_TARGET] <= 0
).any():
    raise ValueError(
        "Rate-per-mile must be positive."
    )


# ============================================================
# Log RPM target
# ============================================================

df[LOG_RPM_TARGET] = np.log(
    df[RPM_TARGET]
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
# Route coverage
# ============================================================

train_routes = set(
    train_df["route"].unique()
)

validation_known_routes = (
    val_df["route"]
    .isin(train_routes)
    .mean()
    * 100
)

print(
    f"\nValidation known-route coverage: "
    f"{validation_known_routes:.2f}%"
)


# ============================================================
# X / y
# ============================================================

X_train = train_df[
    MODEL_FEATURE_COLUMNS
]

y_train = train_df[
    LOG_RPM_TARGET
]

X_val = val_df[
    MODEL_FEATURE_COLUMNS
]

y_val = val_df[
    LOG_RPM_TARGET
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
# Train
# ============================================================

print("\nTraining EXP-008...")

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
# Predict log RPM
# ============================================================

train_log_rpm_pred = pipeline.predict(
    X_train
)

val_log_rpm_pred = pipeline.predict(
    X_val
)


# ============================================================
# Convert back to RPM
# ============================================================

train_rpm_pred = np.exp(
    train_log_rpm_pred
)

val_rpm_pred = np.exp(
    val_log_rpm_pred
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
    "experiment": "EXP-008",

    "description": (
        "Explicit pickup-delivery route "
        "categorical feature + log RPM target"
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

    "unique_routes": int(
        df["route"].nunique()
    ),

    "validation_known_route_percentage": (
        float(validation_known_routes)
    ),

    "negative_weights_corrected": (
        negative_weight_count
    ),

    "training_time_seconds": (
        float(training_time)
    ),

    "target": (
        "log(posted_rate / distance)"
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
    "experiment": "EXP-008",

    "original_target": TARGET,

    "model_target": (
        "log(posted_rate / distance)"
    ),

    "prediction_reconstruction": (
        "exp(predicted_log_rpm) * distance"
    ),

    "added_feature": "route",

    "route_formula": (
        "pickup + ' -> ' + delivery"
    ),

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
print("EXP-008 complete.")
print("=" * 70)

print(
    f"Artifacts saved to: "
    f"{EXP_DIR}"
)
