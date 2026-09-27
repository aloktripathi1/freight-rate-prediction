from pathlib import Path
import json
import time
from collections import defaultdict

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
EXP_DIR = ROOT / "experiments" / "EXP-006"

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
# Same as EXP-004
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

MODEL_BASE_FEATURE_COLUMNS = [
    c for c in BASE_FEATURE_COLUMNS
    if c != DATE_COL
]

CATEGORICAL_COLUMNS = [
    "pickup",
    "delivery",
    "equipment",
]

NUMERICAL_COLUMNS = [
    c for c in MODEL_BASE_FEATURE_COLUMNS
    if c not in CATEGORICAL_COLUMNS
]


# ============================================================
# Historical feature names
# ============================================================

HISTORICAL_NUMERICAL_COLUMNS = [
    "hist_global_rpm_mean",

    "hist_route_rpm_mean",
    "hist_route_rpm_count",

    "hist_pickup_rpm_mean",
    "hist_pickup_rpm_count",

    "hist_delivery_rpm_mean",
    "hist_delivery_rpm_count",

    "hist_equipment_rpm_mean",
    "hist_equipment_rpm_count",
]

MODEL_FEATURE_COLUMNS = (
    MODEL_BASE_FEATURE_COLUMNS
    + HISTORICAL_NUMERICAL_COLUMNS
)

ALL_NUMERICAL_COLUMNS = (
    NUMERICAL_COLUMNS
    + HISTORICAL_NUMERICAL_COLUMNS
)


# ============================================================
# Historical feature state
# ============================================================

def create_state():
    return {
        "global_sum": 0.0,
        "global_count": 0,

        "route_sum": defaultdict(float),
        "route_count": defaultdict(int),

        "pickup_sum": defaultdict(float),
        "pickup_count": defaultdict(int),

        "delivery_sum": defaultdict(float),
        "delivery_count": defaultdict(int),

        "equipment_sum": defaultdict(float),
        "equipment_count": defaultdict(int),
    }


def safe_mean(total, count):
    if count == 0:
        return np.nan
    return total / count


def add_historical_features_from_history(
    current_df,
    state,
):
    """
    Create historical pricing features using ONLY the state
    accumulated before the current rows.

    The state is not modified by this function.
    """

    out = current_df.copy()

    # Global historical RPM
    global_mean = safe_mean(
        state["global_sum"],
        state["global_count"],
    )

    out["hist_global_rpm_mean"] = global_mean

    # Route
    route_keys = list(
        zip(out["pickup"], out["delivery"])
    )

    out["hist_route_rpm_mean"] = [
        safe_mean(
            state["route_sum"][key],
            state["route_count"][key],
        )
        for key in route_keys
    ]

    out["hist_route_rpm_count"] = [
        state["route_count"][key]
        for key in route_keys
    ]

    # Pickup
    out["hist_pickup_rpm_mean"] = [
        safe_mean(
            state["pickup_sum"][key],
            state["pickup_count"][key],
        )
        for key in out["pickup"]
    ]

    out["hist_pickup_rpm_count"] = [
        state["pickup_count"][key]
        for key in out["pickup"]
    ]

    # Delivery
    out["hist_delivery_rpm_mean"] = [
        safe_mean(
            state["delivery_sum"][key],
            state["delivery_count"][key],
        )
        for key in out["delivery"]
    ]

    out["hist_delivery_rpm_count"] = [
        state["delivery_count"][key]
        for key in out["delivery"]
    ]

    # Equipment
    out["hist_equipment_rpm_mean"] = [
        safe_mean(
            state["equipment_sum"][key],
            state["equipment_count"][key],
        )
        for key in out["equipment"]
    ]

    out["hist_equipment_rpm_count"] = [
        state["equipment_count"][key]
        for key in out["equipment"]
    ]

    return out


def update_state_with_day(
    state,
    day_df,
):
    """
    Update historical state using all observations from one day.

    This happens AFTER generating that day's features so the current
    day's target can never leak into its own historical features.
    """

    rpm = day_df[RPM_TARGET].to_numpy()

    state["global_sum"] += float(rpm.sum())
    state["global_count"] += int(len(rpm))

    for pickup, delivery, equipment, value in zip(
        day_df["pickup"],
        day_df["delivery"],
        day_df["equipment"],
        rpm,
    ):
        route_key = (pickup, delivery)

        state["route_sum"][route_key] += float(value)
        state["route_count"][route_key] += 1

        state["pickup_sum"][pickup] += float(value)
        state["pickup_count"][pickup] += 1

        state["delivery_sum"][delivery] += float(value)
        state["delivery_count"][delivery] += 1

        state["equipment_sum"][equipment] += float(value)
        state["equipment_count"][equipment] += 1


def build_training_historical_features(train_df):
    """
    For each training date, generate historical features from
    strictly earlier dates, then update state with that date.
    """

    state = create_state()
    pieces = []

    unique_dates = sorted(
        train_df[DATE_COL].dropna().unique()
    )

    for current_date in unique_dates:

        day_mask = (
            train_df[DATE_COL] == current_date
        )

        day_df = train_df.loc[day_mask].copy()

        # Generate features BEFORE updating state.
        day_features = add_historical_features_from_history(
            day_df,
            state,
        )

        pieces.append(day_features)

        # Now current day's information becomes historical
        # for future dates.
        update_state_with_day(
            state,
            day_df,
        )

    result = pd.concat(
        pieces,
        axis=0,
    ).sort_index()

    return result, state


def build_validation_historical_features(
    val_df,
    trained_state,
):
    """
    Validation features are built from the frozen state containing
    ONLY January-September training history.

    Validation actual targets are never added to the state.
    """

    return add_historical_features_from_history(
        val_df,
        trained_state,
    )


# ============================================================
# Load data
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL]
)

print(
    f"Full development dataset: "
    f"{df.shape}"
)


# ============================================================
# Data-quality correction
# Same as EXP-004
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
# Build leakage-safe historical features
# ============================================================

print(
    "\nBuilding leakage-safe historical "
    "pricing features..."
)

history_start = time.time()

train_with_history, final_state = (
    build_training_historical_features(
        train_df
    )
)

val_with_history = (
    build_validation_historical_features(
        val_df,
        final_state,
    )
)

history_time = (
    time.time() - history_start
)

print(
    f"Historical feature generation time: "
    f"{history_time:.2f} sec"
)


# ============================================================
# Historical feature coverage
# ============================================================

print("\nHistorical feature coverage:")

for column in [
    "hist_route_rpm_mean",
    "hist_pickup_rpm_mean",
    "hist_delivery_rpm_mean",
    "hist_equipment_rpm_mean",
]:
    train_coverage = (
        train_with_history[column]
        .notna()
        .mean()
        * 100
    )

    val_coverage = (
        val_with_history[column]
        .notna()
        .mean()
        * 100
    )

    print(
        f"  {column}: "
        f"train={train_coverage:.2f}% | "
        f"validation={val_coverage:.2f}%"
    )


# ============================================================
# X / y
# ============================================================

X_train = train_with_history[
    MODEL_FEATURE_COLUMNS
]

y_train = train_with_history[
    RPM_TARGET
]

X_val = val_with_history[
    MODEL_FEATURE_COLUMNS
]

y_val = val_with_history[
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
            ALL_NUMERICAL_COLUMNS,
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
# Same as EXP-004
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

print("\nTraining EXP-006...")

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
# Predictions
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
    * train_with_history[
        DISTANCE_COL
    ].to_numpy()
)

val_pred = (
    val_rpm_pred
    * val_with_history[
        DISTANCE_COL
    ].to_numpy()
)

train_actual = (
    train_with_history[
        TARGET
    ].to_numpy()
)

val_actual = (
    val_with_history[
        TARGET
    ].to_numpy()
)


# ============================================================
# Metrics
# ============================================================

def regression_metrics(
    y_true,
    y_pred,
):
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

print("\nInternal Validation:")
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
        "load_id": val_with_history[
            ID_COL
        ].values,

        "predicted_rate": val_pred,

        "actual_rate": val_actual,

        "predicted_rate_per_mile": (
            val_rpm_pred
        ),

        "actual_rate_per_mile": (
            val_actual
            / val_with_history[
                DISTANCE_COL
            ].to_numpy()
        ),
    }
)

predictions.to_csv(
    EXP_DIR
    / "validation_predictions.csv",
    index=False,
)


# ============================================================
# Save metrics
# ============================================================

metrics = {
    "experiment": "EXP-006",

    "description": (
        "Historical leakage-safe pricing "
        "features + RF rate-per-mile target"
    ),

    "seed": SEED,

    "split_date": str(
        SPLIT_DATE.date()
    ),

    "validation_end_date": str(
        VAL_END_DATE.date()
    ),

    "train_rows": int(
        len(train_with_history)
    ),

    "validation_rows": int(
        len(val_with_history)
    ),

    "negative_weights_corrected": (
        negative_weight_count
    ),

    "historical_feature_generation_seconds": (
        float(history_time)
    ),

    "training_time_seconds": (
        float(training_time)
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
    "experiment": "EXP-006",

    "target": RPM_TARGET,

    "prediction_reconstruction": (
        "predicted_rate_per_mile * distance"
    ),

    "historical_features": {
        "route": [
            "hist_route_rpm_mean",
            "hist_route_rpm_count",
        ],
        "pickup": [
            "hist_pickup_rpm_mean",
            "hist_pickup_rpm_count",
        ],
        "delivery": [
            "hist_delivery_rpm_mean",
            "hist_delivery_rpm_count",
        ],
        "equipment": [
            "hist_equipment_rpm_mean",
            "hist_equipment_rpm_count",
        ],
        "global": [
            "hist_global_rpm_mean",
        ],
        "leakage_rule": (
            "Only observations from dates strictly "
            "before prediction date are used."
        ),
    },

    "id_column": ID_COL,
    "date_column": DATE_COL,

    "base_feature_columns": (
        BASE_FEATURE_COLUMNS
    ),

    "model_feature_columns": (
        MODEL_FEATURE_COLUMNS
    ),

    "categorical_columns": (
        CATEGORICAL_COLUMNS
    ),

    "numerical_columns": (
        ALL_NUMERICAL_COLUMNS
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
print("EXP-006 complete.")
print("=" * 70)

print(
    f"Artifacts saved to: "
    f"{EXP_DIR}"
)
