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
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "train-test.csv"
EXP_DIR = ROOT / "experiments" / "EXP-003"

EXP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

SEED = 42
SPLIT_DATE = pd.Timestamp("2025-10-01")

TARGET = "posted_rate"
ID_COL = "load_id"
DATE_COL = "date"

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

MODEL_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c != DATE_COL]

CATEGORICAL_COLUMNS = [
    "pickup",
    "delivery",
    "equipment",
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

df[DATE_COL] = pd.to_datetime(df[DATE_COL])

print(f"Full development dataset: {df.shape}")


# ============================================================
# EXP-002 data-quality correction
# ============================================================

negative_weight_count = (df["weight"] < 0).sum()

print(f"\nNegative weights found: {negative_weight_count}")

# Negative freight weight is physically invalid.
# Convert it to NaN so the existing median imputation handles it.
df.loc[df["weight"] < 0, "weight"] = np.nan

print(
    "Negative weights after correction:",
    (df["weight"] < 0).sum()
)


# ============================================================
# Chronological split
# ============================================================

train_df = df[df[DATE_COL] < SPLIT_DATE].copy()
val_df = df[
    (df[DATE_COL] >= SPLIT_DATE)
    & (df[DATE_COL] < pd.Timestamp("2025-11-01"))
].copy()

print(
    f"\nTraining:   {train_df.shape} "
    f"({train_df[DATE_COL].min().date()} → {train_df[DATE_COL].max().date()})"
)

print(
    f"Validation: {val_df.shape} "
    f"({val_df[DATE_COL].min().date()} → {val_df[DATE_COL].max().date()})"
)


X_train = train_df[MODEL_FEATURE_COLUMNS]
y_train = train_df[TARGET]

X_val = val_df[MODEL_FEATURE_COLUMNS]
y_val = val_df[TARGET]


# ============================================================
# Preprocessing
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
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
        ("num", numeric_pipeline, NUMERICAL_COLUMNS),
        ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
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
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ============================================================
# Train
# ============================================================

print("\nTraining EXP-003...")
start_time = time.time()

pipeline.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"Training time: {training_time:.2f} sec")


# ============================================================
# Evaluation
# ============================================================

train_pred = pipeline.predict(X_train)
val_pred = pipeline.predict(X_val)


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


train_metrics = regression_metrics(y_train, train_pred)
val_metrics = regression_metrics(y_val, val_pred)

print("\nTraining:")
print(f"  MAE :  {train_metrics['mae']:.4f}")
print(f"  RMSE:  {train_metrics['rmse']:.4f}")
print(f"  R²  :  {train_metrics['r2']:.4f}")

print("\nInternal Validation:")
print(f"  MAE :  {val_metrics['mae']:.4f}")
print(f"  RMSE:  {val_metrics['rmse']:.4f}")
print(f"  R²  :  {val_metrics['r2']:.4f}")

print("\nGeneralization gap:")
print(
    f"  MAE gap : "
    f"{val_metrics['mae'] - train_metrics['mae']:.4f}"
)
print(
    f"  RMSE gap: "
    f"{val_metrics['rmse'] - train_metrics['rmse']:.4f}"
)
print(
    f"  R² gap  : "
    f"{train_metrics['r2'] - val_metrics['r2']:.4f}"
)


# ============================================================
# Save model
# ============================================================

model_path = EXP_DIR / "model_pipeline.joblib"
joblib.dump(pipeline, model_path)


# ============================================================
# Save validation predictions
# ============================================================

predictions = pd.DataFrame(
    {
        "load_id": val_df[ID_COL].values,
        "predicted_rate": val_pred,
        "actual_rate": y_val.values,
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
    "experiment": "EXP-003",
    "description": "Negative weight correction + RF min_samples_leaf=5",
    "seed": SEED,
    "split_date": str(SPLIT_DATE.date()),
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "negative_weights_corrected": int(negative_weight_count),
    "training_time_seconds": float(training_time),
    "train_metrics": train_metrics,
    "validation_metrics": val_metrics,
    "generalization_gap": {
        "mae": float(
            val_metrics["mae"] - train_metrics["mae"]
        ),
        "rmse": float(
            val_metrics["rmse"] - train_metrics["rmse"]
        ),
        "r2": float(
            train_metrics["r2"] - val_metrics["r2"]
        ),
    },
}

with open(EXP_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


# ============================================================
# Save configuration
# ============================================================

config = {
    "experiment": "EXP-003",
    "target": TARGET,
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
        "negative_weight_handling": "weight < 0 -> NaN",
    },
    "model": {
        "type": "RandomForestRegressor",
        "n_estimators": 100,
        "random_state": SEED,
        "n_jobs": -1,
    },
}

with open(EXP_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)


print("\nEXP-003 complete.")
print(f"Artifacts saved to: {EXP_DIR}")
