from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "train-test.csv"
VALIDATION_PATH = ROOT / "data" / "validation.csv"

OUTPUT_DIR = ROOT / "final_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "model_pipeline.joblib"
CONFIG_PATH = OUTPUT_DIR / "config.json"
SUMMARY_PATH = OUTPUT_DIR / "prediction_summary.json"

SUBMISSION_PATH = ROOT / "predictions" / "validation_predictions.csv"


# ============================================================
# Configuration
# ============================================================

SEED = 42

TARGET = "posted_rate"
RPM_TARGET = "posted_rate_per_mile"
LOG_RPM_TARGET = "log_posted_rate_per_mile"

ID_COL = "load_id"
DATE_COL = "date"
DISTANCE_COL = "distance"

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
    "market_index",
    "quote_signal",
]

CATEGORICAL_COLUMNS = [
    "pickup",
    "delivery",
    "equipment",
]

NUMERICAL_COLUMNS = [
    c
    for c in FEATURE_COLUMNS
    if c not in CATEGORICAL_COLUMNS
]


# ============================================================
# Load data
# ============================================================

print("Loading training data...")

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

print(f"Training data:   {train_df.shape}")
print(f"Validation data: {validation_df.shape}")


# ============================================================
# Basic validation
# ============================================================

required_train_columns = set(
    FEATURE_COLUMNS + [ID_COL, TARGET]
)

required_validation_columns = set(
    FEATURE_COLUMNS + [ID_COL]
)

missing_train = required_train_columns - set(
    train_df.columns
)

missing_validation = required_validation_columns - set(
    validation_df.columns
)

if missing_train:
    raise ValueError(
        f"Missing training columns: {sorted(missing_train)}"
    )

if missing_validation:
    raise ValueError(
        f"Missing validation columns: {sorted(missing_validation)}"
    )


if train_df[ID_COL].duplicated().any():
    raise ValueError(
        "Duplicate load_id values found in training data."
    )

if validation_df[ID_COL].duplicated().any():
    raise ValueError(
        "Duplicate load_id values found in validation data."
    )


# ============================================================
# Parse date
# Date is NOT used as a model feature.
# ============================================================

train_df[DATE_COL] = pd.to_datetime(
    train_df[DATE_COL]
)

validation_df[DATE_COL] = pd.to_datetime(
    validation_df[DATE_COL]
)


print(
    f"\nTraining date range: "
    f"{train_df[DATE_COL].min().date()} → "
    f"{train_df[DATE_COL].max().date()}"
)

print(
    f"Validation date range: "
    f"{validation_df[DATE_COL].min().date()} → "
    f"{validation_df[DATE_COL].max().date()}"
)


# ============================================================
# Data-quality correction
# ============================================================

train_negative_weights = int(
    (train_df["weight"] < 0).sum()
)

validation_negative_weights = int(
    (validation_df["weight"] < 0).sum()
)

print(
    f"\nNegative training weights: "
    f"{train_negative_weights}"
)

print(
    f"Negative validation weights: "
    f"{validation_negative_weights}"
)

train_df.loc[
    train_df["weight"] < 0,
    "weight",
] = np.nan

validation_df.loc[
    validation_df["weight"] < 0,
    "weight",
] = np.nan


# ============================================================
# Distance validation
# ============================================================

if (train_df[DISTANCE_COL] <= 0).any():
    raise ValueError(
        "Training contains non-positive distance."
    )

if (validation_df[DISTANCE_COL] <= 0).any():
    raise ValueError(
        "Validation contains non-positive distance."
    )


# ============================================================
# Create final training target
# ============================================================

train_df[RPM_TARGET] = (
    train_df[TARGET]
    / train_df[DISTANCE_COL]
)

if (train_df[RPM_TARGET] <= 0).any():
    raise ValueError(
        "Training contains non-positive rate-per-mile."
    )

train_df[LOG_RPM_TARGET] = np.log(
    train_df[RPM_TARGET]
)

print("\nFinal log-RPM target:")
print(
    train_df[LOG_RPM_TARGET].describe()
)


# ============================================================
# Prepare X / y
# ============================================================

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[LOG_RPM_TARGET]

X_validation = validation_df[FEATURE_COLUMNS]


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
# Final model
# ============================================================

model = ExtraTreesRegressor(
    n_estimators=100,
    min_samples_leaf=20,
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
# Train on ALL labeled development data
# ============================================================

print("\nTraining final model on all labeled data...")

start_time = time.time()

pipeline.fit(
    X_train,
    y_train,
)

joblib.dump(
    pipeline,
    MODEL_PATH,
)

print(f"Model saved to: {MODEL_PATH}")

training_time = time.time() - start_time

print(
    f"Training time: "
    f"{training_time:.2f} sec"
)


# ============================================================
# Predict validation set
# ============================================================

print("\nPredicting validation.csv...")

predicted_log_rpm = pipeline.predict(
    X_validation
)

predicted_rpm = np.exp(
    predicted_log_rpm
)

predicted_rate = (
    predicted_rpm
    * validation_df[DISTANCE_COL].to_numpy()
)


# ============================================================
# Validate predictions
# ============================================================

if len(predicted_rate) != len(validation_df):
    raise ValueError(
        "Prediction count does not match validation rows."
    )

if not np.isfinite(predicted_rate).all():
    raise ValueError(
        "Predictions contain NaN or infinite values."
    )

if (predicted_rate <= 0).any():
    raise ValueError(
        "Predictions must be strictly positive."
    )


# ============================================================
# Build exact submission
# ============================================================

submission = pd.DataFrame(
    {
        "load_id": validation_df[ID_COL].values,
        "predicted_rate": predicted_rate,
    }
)


# Exact column order
submission = submission[
    [
        "load_id",
        "predicted_rate",
    ]
]


# ============================================================
# Final submission validation
# ============================================================

expected_rows = 12000

if len(submission) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} predictions, "
        f"got {len(submission)}."
    )

if submission["load_id"].isna().any():
    raise ValueError(
        "Submission contains missing load_id."
    )

if submission["predicted_rate"].isna().any():
    raise ValueError(
        "Submission contains missing predictions."
    )

if submission["load_id"].duplicated().any():
    raise ValueError(
        "Submission contains duplicate load_id values."
    )

if list(submission.columns) != [
    "load_id",
    "predicted_rate",
]:
    raise ValueError(
        "Submission columns are incorrect."
    )

if not np.isfinite(
    submission["predicted_rate"]
).all():
    raise ValueError(
        "Submission contains non-finite predictions."
    )

if (
    submission["predicted_rate"] <= 0
).any():
    raise ValueError(
        "Submission contains non-positive predictions."
    )


# ============================================================
# Save submission
# ============================================================

submission.to_csv(
    SUBMISSION_PATH,
    index=False,
)

print(
    f"\nSubmission saved to:\n"
    f"{SUBMISSION_PATH}"
)


# ============================================================
# Prediction summary
# ============================================================

summary = {
    "training_rows": int(len(train_df)),
    "validation_rows": int(len(validation_df)),
    "negative_training_weights_corrected": (
        train_negative_weights
    ),
    "negative_validation_weights_corrected": (
        validation_negative_weights
    ),
    "training_time_seconds": float(
        training_time
    ),
    "prediction_summary": {
        "min": float(
            submission["predicted_rate"].min()
        ),
        "median": float(
            submission["predicted_rate"].median()
        ),
        "mean": float(
            submission["predicted_rate"].mean()
        ),
        "max": float(
            submission["predicted_rate"].max()
        ),
    },
}


with open(
    SUMMARY_PATH,
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# Save model configuration
# ============================================================

config = {
    "model_selection_basis": (
        "EXP-012 temporal cross-validation"
    ),

    "target": (
        "log(posted_rate / distance)"
    ),

    "prediction_reconstruction": (
        "exp(predicted_log_rpm) * distance"
    ),

    "features": FEATURE_COLUMNS,

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
        "one_hot_handle_unknown": "ignore",
        "negative_weight_handling": (
            "weight < 0 -> NaN"
        ),
    },

    "model": {
        "type": "ExtraTreesRegressor",
        "n_estimators": 100,
        "min_samples_leaf": 20,
        "random_state": SEED,
        "n_jobs": -1,
    },

    "training_data": {
        "file": "data/train-test.csv",
        "rows": int(len(train_df)),
        "date_start": str(
            train_df[DATE_COL].min().date()
        ),
        "date_end": str(
            train_df[DATE_COL].max().date()
        ),
    },

    "prediction_data": {
        "file": "data/validation.csv",
        "rows": int(len(validation_df)),
        "date_start": str(
            validation_df[DATE_COL].min().date()
        ),
        "date_end": str(
            validation_df[DATE_COL].max().date()
        ),
    },
}


with open(
    CONFIG_PATH,
    "w",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRAINING COMPLETE")
print("=" * 70)

print(
    f"Training rows:      {len(train_df):,}"
)

print(
    f"Validation rows:    {len(validation_df):,}"
)

print(
    f"Prediction min:     "
    f"${submission['predicted_rate'].min():,.2f}"
)

print(
    f"Prediction median:  "
    f"${submission['predicted_rate'].median():,.2f}"
)

print(
    f"Prediction mean:    "
    f"${submission['predicted_rate'].mean():,.2f}"
)

print(
    f"Prediction max:     "
    f"${submission['predicted_rate'].max():,.2f}"
)

print("\nFiles:")
print(f"  Model:      {MODEL_PATH}")
print(f"  Config:     {CONFIG_PATH}")
print(f"  Summary:    {SUMMARY_PATH}")
print(f"  Submission: {SUBMISSION_PATH}")
