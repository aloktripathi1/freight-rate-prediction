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
EXP_DIR = ROOT / "experiments" / "EXP-004"

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
# Features
# ============================================================

FEATURE_COLUMNS = [
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

# Raw datetime is used for splitting only.
MODEL_FEATURE_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column != DATE_COL
]

CATEGORICAL_COLUMNS = [
    "pickup",
    "delivery",
    "equipment",
]

NUMERICAL_COLUMNS = [
    column
    for column in MODEL_FEATURE_COLUMNS
    if column not in CATEGORICAL_COLUMNS
]


# ============================================================
# Load data
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

print(f"Full development dataset: {df.shape}")


# ============================================================
# Data-quality correction
# Same as EXP-003
# ============================================================

negative_weight_count = int((df["weight"] < 0).sum())

print(f"\nNegative weights found: {negative_weight_count}")

# Negative freight weight is physically invalid.
# Convert to NaN so the existing median imputer handles it.
df.loc[df["weight"] < 0, "weight"] = np.nan

print(
    "Negative weights after correction:",
    int((df["weight"] < 0).sum()),
)


# ============================================================
# Validate distance
# ============================================================

if (df[DISTANCE_COL] <= 0).any():
    raise ValueError(
        "EXP-004 requires strictly positive distance values."
    )

print(
    f"\nDistance range: "
    f"{df[DISTANCE_COL].min():.2f} → "
    f"{df[DISTANCE_COL].max():.2f}"
)


# ============================================================
# Create new target
# ============================================================

df[RPM_TARGET] = df[TARGET] / df[DISTANCE_COL]

print("\nRate-per-mile target:")
print(df[RPM_TARGET].describe())


# ============================================================
# Chronological split
# ============================================================

train_df = df[df[DATE_COL] < SPLIT_DATE].copy()

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

X_train = train_df[MODEL_FEATURE_COLUMNS]
y_train = train_df[RPM_TARGET]

X_val = val_df[MODEL_FEATURE_COLUMNS]
y_val = val_df[RPM_TARGET]


# ============================================================
# Preprocessing
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
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
# Same regularization as EXP-003
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ============================================================
# Training
# ============================================================

print("\nTraining EXP-004...")

start_time = time.time()

pipeline.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"Training time: {training_time:.2f} sec")


# ============================================================
# Predict rate-per-mile
# ============================================================

train_rpm_pred = pipeline.predict(X_train)
val_rpm_pred = pipeline.predict(X_val)


# ============================================================
# Convert back to total posted rate
# ============================================================

train_pred = (
    train_rpm_pred
    * train_df[DISTANCE_COL].to_numpy()
)

val_pred = (
    val_rpm_pred
    * val_df[DISTANCE_COL].to_numpy()
)

train_actual = train_df[TARGET].to_numpy()
val_actual = val_df[TARGET].to_numpy()


# ============================================================
# Metrics
# Evaluate on original business target: posted_rate
# ============================================================

def regression_metrics(y_true, y_pred):
    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
        "r2": float(
            r2_score(y_true, y_pred)
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
# Results
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print("\nTraining:")
print(f"  MAE :  {train_metrics['mae']:.4f}")
print(f"  RMSE:  {train_metrics['rmse']:.4f}")
print(f"  R²  :  {train_metrics['r2']:.4f}")

print("\nInternal Validation:")
print(f"  MAE :  {val_metrics['mae']:.4f}")
print(f"  RMSE:  {val_metrics['rmse']:.4f}")
print(f"  R²  :  {val_metrics['r2']:.4f}")

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

print("\nGeneralization gap:")
print(f"  MAE gap : {mae_gap:.4f}")
print(f"  RMSE gap: {rmse_gap:.4f}")
print(f"  R² gap  : {r2_gap:.4f}")


# ============================================================
# Save model
# ============================================================

model_path = EXP_DIR / "model_pipeline.joblib"

joblib.dump(
    pipeline,
    model_path,
)


# ============================================================
# Save validation predictions
# ============================================================

predictions = pd.DataFrame(
    {
        "load_id": val_df[ID_COL].values,
        "predicted_rate": val_pred,
        "actual_rate": val_actual,
        "predicted_rate_per_mile": val_rpm_pred,
        "actual_rate_per_mile": y_val.values,
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
    "experiment": "EXP-004",
    "description": (
        "Predict rate per mile and reconstruct posted rate"
    ),
    "seed": SEED,
    "split_date": str(SPLIT_DATE.date()),
    "validation_end_date": str(VAL_END_DATE.date()),
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "negative_weights_corrected": negative_weight_count,
    "training_time_seconds": float(training_time),
    "train_metrics_posted_rate": train_metrics,
    "validation_metrics_posted_rate": val_metrics,
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
    json.dump(metrics, f, indent=2)


# ============================================================
# Save configuration
# ============================================================

config = {
    "experiment": "EXP-004",

    "original_target": TARGET,

    "model_target": {
        "name": RPM_TARGET,
        "formula": "posted_rate / distance",
    },

    "prediction_reconstruction": (
        "predicted_rate_per_mile * distance"
    ),

    "id_column": ID_COL,
    "date_column": DATE_COL,

    "feature_columns": FEATURE_COLUMNS,
    "model_feature_columns": MODEL_FEATURE_COLUMNS,

    "categorical_columns": CATEGORICAL_COLUMNS,
    "numerical_columns": NUMERICAL_COLUMNS,

    "preprocessing": {
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "one_hot_handle_unknown": "ignore",
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
    json.dump(config, f, indent=2)


print("\n" + "=" * 70)
print("EXP-004 complete.")
print("=" * 70)
print(f"Artifacts saved to: {EXP_DIR}")
