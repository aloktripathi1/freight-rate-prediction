from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "train-test.csv"
EXP_DIR = ROOT / "experiments" / "EXP-012"

EXP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

TARGET = "posted_rate"
RPM_TARGET = "posted_rate_per_mile"
LOG_RPM_TARGET = "log_posted_rate_per_mile"

DATE_COL = "date"
DISTANCE_COL = "distance"


# ============================================================
# Temporal folds
#
# Train period is [train_start, train_end)
# Validation period is [val_start, val_end)
# ============================================================

FOLDS = [
    {
        "name": "fold_1",
        "train_start": "2025-01-01",
        "train_end": "2025-07-01",
        "val_start": "2025-07-01",
        "val_end": "2025-09-01",
    },
    {
        "name": "fold_2",
        "train_start": "2025-01-01",
        "train_end": "2025-08-01",
        "val_start": "2025-08-01",
        "val_end": "2025-10-01",
    },
    {
        "name": "fold_3",
        "train_start": "2025-01-01",
        "train_end": "2025-10-01",
        "val_start": "2025-10-01",
        "val_end": "2025-11-01",
    },
]


# ============================================================
# Features
# Same as EXP-011
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
    "market_index",
    "quote_signal",
]

CATEGORICAL_COLUMNS = [
    "pickup",
    "delivery",
    "equipment",
]

NUMERICAL_COLUMNS = [
    c for c in FEATURE_COLUMNS
    if c not in CATEGORICAL_COLUMNS
]


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


# ============================================================
# Build pipeline
# ============================================================

def build_pipeline():
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

    model = ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=20,
        random_state=SEED,
        n_jobs=-1,
    )

    return Pipeline(
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
# Load dataset
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
# Data quality correction
# ============================================================

negative_weight_count = int(
    (df["weight"] < 0).sum()
)

print(
    f"Negative weights found: "
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
# Create RPM / log-RPM targets
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

df[LOG_RPM_TARGET] = np.log(
    df[RPM_TARGET]
)


# ============================================================
# Run temporal folds
# ============================================================

fold_results = []
all_predictions = []

overall_start = time.time()

for fold in FOLDS:

    fold_name = fold["name"]

    train_start = pd.Timestamp(
        fold["train_start"]
    )
    train_end = pd.Timestamp(
        fold["train_end"]
    )
    val_start = pd.Timestamp(
        fold["val_start"]
    )
    val_end = pd.Timestamp(
        fold["val_end"]
    )

    train_df = df[
        (df[DATE_COL] >= train_start)
        & (df[DATE_COL] < train_end)
    ].copy()

    val_df = df[
        (df[DATE_COL] >= val_start)
        & (df[DATE_COL] < val_end)
    ].copy()

    print("\n" + "=" * 80)
    print(f"{fold_name.upper()}")
    print("=" * 80)

    print(
        f"Training:   {len(train_df):,} rows "
        f"({train_df[DATE_COL].min().date()} → "
        f"{train_df[DATE_COL].max().date()})"
    )

    print(
        f"Validation: {len(val_df):,} rows "
        f"({val_df[DATE_COL].min().date()} → "
        f"{val_df[DATE_COL].max().date()})"
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LOG_RPM_TARGET]

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[LOG_RPM_TARGET]

    pipeline = build_pipeline()

    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # Predict log RPM
    # --------------------------------------------------------

    train_log_rpm_pred = pipeline.predict(
        X_train
    )

    val_log_rpm_pred = pipeline.predict(
        X_val
    )

    # --------------------------------------------------------
    # Back-transform
    # --------------------------------------------------------

    train_rpm_pred = np.exp(
        train_log_rpm_pred
    )

    val_rpm_pred = np.exp(
        val_log_rpm_pred
    )

    # --------------------------------------------------------
    # Reconstruct posted rate
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Metrics on original posted_rate
    # --------------------------------------------------------

    train_metrics = regression_metrics(
        train_actual,
        train_pred,
    )

    val_metrics = regression_metrics(
        val_actual,
        val_pred,
    )

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

    print(
        f"\nTraining time: "
        f"{training_time:.2f} sec"
    )

    print("\nTraining:")
    print(
        f"  MAE : "
        f"{train_metrics['mae']:.4f}"
    )
    print(
        f"  RMSE: "
        f"{train_metrics['rmse']:.4f}"
    )
    print(
        f"  R²  : "
        f"{train_metrics['r2']:.4f}"
    )

    print("\nValidation:")
    print(
        f"  MAE : "
        f"{val_metrics['mae']:.4f}"
    )
    print(
        f"  RMSE: "
        f"{val_metrics['rmse']:.4f}"
    )
    print(
        f"  R²  : "
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

    # --------------------------------------------------------
    # Save fold metrics
    # --------------------------------------------------------

    fold_results.append(
        {
            "fold": fold_name,
            "train_start": str(
                train_start.date()
            ),
            "train_end": str(
                train_end.date()
            ),
            "validation_start": str(
                val_start.date()
            ),
            "validation_end": str(
                val_end.date()
            ),
            "train_rows": int(
                len(train_df)
            ),
            "validation_rows": int(
                len(val_df)
            ),
            "training_time_seconds": float(
                training_time
            ),
            "train_mae": (
                train_metrics["mae"]
            ),
            "train_rmse": (
                train_metrics["rmse"]
            ),
            "train_r2": (
                train_metrics["r2"]
            ),
            "validation_mae": (
                val_metrics["mae"]
            ),
            "validation_rmse": (
                val_metrics["rmse"]
            ),
            "validation_r2": (
                val_metrics["r2"]
            ),
            "mae_gap": float(
                mae_gap
            ),
            "rmse_gap": float(
                rmse_gap
            ),
            "r2_gap": float(
                r2_gap
            ),
        }
    )

    # --------------------------------------------------------
    # Save fold predictions
    # --------------------------------------------------------

    fold_predictions = pd.DataFrame(
        {
            "fold": fold_name,
            "load_id": val_df["load_id"].values,
            "actual_rate": val_actual,
            "predicted_rate": val_pred,
            "actual_rate_per_mile": (
                val_actual
                / val_df[DISTANCE_COL].to_numpy()
            ),
            "predicted_rate_per_mile": (
                val_rpm_pred
            ),
        }
    )

    all_predictions.append(
        fold_predictions
    )


# ============================================================
# Aggregate results
# ============================================================

total_time = (
    time.time() - overall_start
)

results_df = pd.DataFrame(
    fold_results
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)


print("\n" + "=" * 80)
print("TEMPORAL CV SUMMARY")
print("=" * 80)

print(
    results_df[
        [
            "fold",
            "train_rows",
            "validation_rows",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
        ]
    ].to_string(index=False)
)


# ============================================================
# Aggregate statistics
# ============================================================

aggregate = {
    "mean_validation_mae": float(
        results_df["validation_mae"].mean()
    ),
    "std_validation_mae": float(
        results_df["validation_mae"].std(
            ddof=1
        )
    ),
    "mean_validation_rmse": float(
        results_df["validation_rmse"].mean()
    ),
    "std_validation_rmse": float(
        results_df["validation_rmse"].std(
            ddof=1
        )
    ),
    "mean_validation_r2": float(
        results_df["validation_r2"].mean()
    ),
    "std_validation_r2": float(
        results_df["validation_r2"].std(
            ddof=1
        )
    ),
    "mean_mae_gap": float(
        results_df["mae_gap"].mean()
    ),
    "total_training_time_seconds": float(
        total_time
    ),
}


# ============================================================
# Save artifacts
# ============================================================

results_df.to_csv(
    EXP_DIR / "fold_results.csv",
    index=False,
)

predictions_df.to_csv(
    EXP_DIR / "temporal_cv_predictions.csv",
    index=False,
)

summary = {
    "experiment": "EXP-012",
    "description": (
        "Expanding-window temporal cross-validation "
        "of ExtraTrees log-RPM candidate"
    ),
    "seed": SEED,
    "model": {
        "type": "ExtraTreesRegressor",
        "n_estimators": 100,
        "min_samples_leaf": 20,
        "random_state": SEED,
        "n_jobs": -1,
    },
    "target": (
        "log(posted_rate / distance)"
    ),
    "prediction_reconstruction": (
        "exp(predicted_log_rpm) * distance"
    ),
    "negative_weights_corrected": (
        negative_weight_count
    ),
    "folds": FOLDS,
    "aggregate": aggregate,
}


with open(
    EXP_DIR / "metrics.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print("\n" + "=" * 80)
print("EXP-012 COMPLETE")
print("=" * 80)

print(
    f"Mean validation MAE: "
    f"${aggregate['mean_validation_mae']:,.2f}"
)

print(
    f"Std validation MAE:  "
    f"${aggregate['std_validation_mae']:,.2f}"
)

print(
    f"Mean validation RMSE: "
    f"${aggregate['mean_validation_rmse']:,.2f}"
)

print(
    f"Mean validation R²:   "
    f"{aggregate['mean_validation_r2']:.4f}"
)

print(
    f"\nArtifacts saved to: "
    f"{EXP_DIR}"
)
