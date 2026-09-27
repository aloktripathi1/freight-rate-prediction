"""
EXP-001: Baseline freight-rate prediction model.

Purpose
-------
Establish a reproducible baseline before feature engineering or
hyperparameter tuning.

Validation strategy
-------------------
Chronological split:
    Train      : 2025-01-01 through 2025-09-30
    Validation : 2025-10-01 through 2025-10-31

The provided validation.csv (November-December) is NOT used here.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "train-test.csv"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "EXP-001"

MODEL_PATH = EXPERIMENT_DIR / "baseline_pipeline.joblib"
METRICS_PATH = EXPERIMENT_DIR / "metrics.json"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"
PREDICTIONS_PATH = EXPERIMENT_DIR / "validation_predictions.csv"


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def get_git_commit() -> str | None:
    """Return the current Git commit for experiment traceability."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Calculate root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the labeled development dataset."""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    if "posted_rate" not in df.columns:
        raise ValueError("Expected target column 'posted_rate' not found.")

    return df


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split development data chronologically.

    September and earlier -> training
    October                -> internal validation
    """
    train_mask = df["date"] < pd.Timestamp("2025-10-01")
    val_mask = (
        (df["date"] >= pd.Timestamp("2025-10-01"))
        & (df["date"] <= pd.Timestamp("2025-10-31"))
    )

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Chronological split produced an empty dataset.")

    return train_df, val_df


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Numerical:
        Median imputation.

    Categorical:
        Most-frequent imputation + one-hot encoding.

    No scaling is used because the baseline model is tree-based.
    """

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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP-001 — BASELINE MODEL")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    df = load_data()

    print(f"\nFull development dataset: {df.shape}")

    # -------------------------------------------------------------
    # Chronological split
    # -------------------------------------------------------------

    train_df, val_df = chronological_split(df)

    print(
        f"Training:   {train_df.shape} "
        f"({train_df['date'].min().date()} → "
        f"{train_df['date'].max().date()})"
    )

    print(
        f"Validation: {val_df.shape} "
        f"({val_df['date'].min().date()} → "
        f"{val_df['date'].max().date()})"
    )

    # -------------------------------------------------------------
    # Features / target
    # -------------------------------------------------------------

    target = "posted_rate"
    id_column = "load_id"

    feature_columns = [
        column
        for column in df.columns
        if column not in {target, id_column}
    ]

    # Raw date is used for chronological splitting only.
    # Temporal feature engineering will be evaluated separately.
    model_feature_columns = [
        column
        for column in feature_columns
        if column != "date"
    ]

    categorical_features = [
        "pickup",
        "delivery",
        "equipment",
    ]

    numeric_features = [
        column
        for column in model_feature_columns
        if column not in categorical_features
    ]

    X_train = train_df[model_feature_columns]
    y_train = train_df[target]

    X_val = val_df[model_feature_columns]
    y_val = val_df[target]

    # -------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    print("\nTraining model...")

    start_time = time.perf_counter()

    pipeline.fit(X_train, y_train)

    training_seconds = time.perf_counter() - start_time

    print(f"Training time: {training_seconds:.2f} seconds")

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------

    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = rmse(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = rmse(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nTraining:")
    print(f"  MAE :  {train_mae:.4f}")
    print(f"  RMSE:  {train_rmse:.4f}")
    print(f"  R²  :  {train_r2:.4f}")

    print("\nInternal Validation:")
    print(f"  MAE :  {val_mae:.4f}")
    print(f"  RMSE:  {val_rmse:.4f}")
    print(f"  R²  :  {val_r2:.4f}")

    print("\nGeneralization gap:")
    print(f"  MAE gap :  {val_mae - train_mae:.4f}")
    print(f"  RMSE gap:  {val_rmse - train_rmse:.4f}")
    print(f"  R² gap  :  {train_r2 - val_r2:.4f}")

    # -------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------

    predictions = pd.DataFrame(
        {
            "load_id": val_df[id_column].values,
            "actual_rate": y_val.values,
            "predicted_rate": y_val_pred,
        }
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    # -------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------

    metrics = {
        "experiment_id": "EXP-001",
        "model": "RandomForestRegressor",
        "random_seed": RANDOM_SEED,
        "train": {
            "rows": int(len(train_df)),
            "start_date": str(train_df["date"].min().date()),
            "end_date": str(train_df["date"].max().date()),
            "mae": float(train_mae),
            "rmse": float(train_rmse),
            "r2": float(train_r2),
        },
        "validation": {
            "rows": int(len(val_df)),
            "start_date": str(val_df["date"].min().date()),
            "end_date": str(val_df["date"].max().date()),
            "mae": float(val_mae),
            "rmse": float(val_rmse),
            "r2": float(val_r2),
        },
        "training_seconds": float(training_seconds),
        "features": model_feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "git_commit": get_git_commit(),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # -------------------------------------------------------------
    # Save configuration
    # -------------------------------------------------------------

    config = {
        "experiment_id": "EXP-001",
        "description": "Raw-feature Random Forest baseline",
        "random_seed": RANDOM_SEED,
        "validation_strategy": "chronological",
        "train_period": "2025-01-01 to 2025-09-30",
        "validation_period": "2025-10-01 to 2025-10-31",
        "date_used_for_split_only": True,
        "target": target,
        "excluded_columns": [id_column],
        "model_parameters": {
            "n_estimators": 100,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        },
        "preprocessing": {
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot",
            "handle_unknown": "ignore",
            "scaling": False,
        },
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    # -------------------------------------------------------------
    # Save complete pipeline
    # -------------------------------------------------------------

    import joblib

    joblib.dump(pipeline, MODEL_PATH)

    print("\nArtifacts saved:")
    print(f"  Model       : {MODEL_PATH}")
    print(f"  Metrics     : {METRICS_PATH}")
    print(f"  Config      : {CONFIG_PATH}")
    print(f"  Predictions : {PREDICTIONS_PATH}")

    print("\nEXP-001 complete.")


if __name__ == "__main__":
    main()