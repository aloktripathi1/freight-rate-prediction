from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "train-test.csv"
PRED_PATH = ROOT / "experiments" / "EXP-004" / "validation_predictions.csv"
OUT_DIR = ROOT / "experiments" / "EXP-004" / "error_analysis"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load data
# ============================================================

print("Loading validation data and predictions...")

df = pd.read_csv(DATA_PATH)
pred = pd.read_csv(PRED_PATH)

df["date"] = pd.to_datetime(df["date"])

# October is the internal validation set
val_df = df[
    (df["date"] >= "2025-10-01")
    & (df["date"] < "2025-11-01")
].copy()

print(f"Validation rows: {len(val_df):,}")
print(f"Prediction rows: {len(pred):,}")


# ============================================================
# Merge predictions with original features
# ============================================================

result = val_df.merge(
    pred[
        [
            "load_id",
            "predicted_rate",
            "actual_rate",
            "predicted_rate_per_mile",
            "actual_rate_per_mile",
        ]
    ],
    on="load_id",
    how="inner",
    validate="one_to_one",
)

if len(result) != len(val_df):
    raise ValueError(
        "Prediction merge failed: row count mismatch."
    )


# ============================================================
# Error calculations
# ============================================================

result["error"] = (
    result["predicted_rate"]
    - result["actual_rate"]
)

result["absolute_error"] = result["error"].abs()

result["squared_error"] = (
    result["error"] ** 2
)

# Percentage error can become unstable for very small rates,
# so we only use it for descriptive analysis.
result["absolute_percentage_error"] = (
    result["absolute_error"]
    / result["actual_rate"].abs()
    * 100
)


# ============================================================
# Overall metrics
# ============================================================

mae = mean_absolute_error(
    result["actual_rate"],
    result["predicted_rate"],
)

rmse = np.sqrt(
    mean_squared_error(
        result["actual_rate"],
        result["predicted_rate"],
    )
)

bias = result["error"].mean()

median_abs_error = result["absolute_error"].median()

p90_abs_error = result["absolute_error"].quantile(0.90)

p95_abs_error = result["absolute_error"].quantile(0.95)

p99_abs_error = result["absolute_error"].quantile(0.99)


print("\n" + "=" * 70)
print("OVERALL ERROR ANALYSIS")
print("=" * 70)

print(f"MAE:              ${mae:,.2f}")
print(f"RMSE:             ${rmse:,.2f}")
print(f"Mean signed error: ${bias:,.2f}")
print(f"Median abs error: ${median_abs_error:,.2f}")
print(f"P90 abs error:    ${p90_abs_error:,.2f}")
print(f"P95 abs error:    ${p95_abs_error:,.2f}")
print(f"P99 abs error:    ${p99_abs_error:,.2f}")


# ============================================================
# Helper for grouped error analysis
# ============================================================

def grouped_analysis(group_col):
    grouped = (
        result.groupby(group_col, dropna=False)
        .agg(
            rows=("load_id", "size"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda x: np.sqrt(x.mean())),
            mean_error=("error", "mean"),
            median_error=("error", "median"),
            actual_rate_mean=("actual_rate", "mean"),
            predicted_rate_mean=("predicted_rate", "mean"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    return grouped


# ============================================================
# Equipment analysis
# ============================================================

equipment = grouped_analysis("equipment")

equipment.to_csv(
    OUT_DIR / "by_equipment.csv",
    index=False,
)

print("\n" + "=" * 70)
print("ERROR BY EQUIPMENT")
print("=" * 70)

print(
    equipment.to_string(
        index=False,
        formatters={
            "mae": "${:,.2f}".format,
            "rmse": "${:,.2f}".format,
            "mean_error": "${:,.2f}".format,
            "median_error": "${:,.2f}".format,
            "actual_rate_mean": "${:,.2f}".format,
            "predicted_rate_mean": "${:,.2f}".format,
        },
    )
)


# ============================================================
# Distance bins
# ============================================================

result["distance_bin"] = pd.cut(
    result["distance"],
    bins=[
        0,
        250,
        500,
        750,
        1000,
        1500,
        2000,
        np.inf,
    ],
    labels=[
        "<=250",
        "251-500",
        "501-750",
        "751-1000",
        "1001-1500",
        "1501-2000",
        ">2000",
    ],
)

distance_analysis = grouped_analysis("distance_bin")

distance_analysis.to_csv(
    OUT_DIR / "by_distance.csv",
    index=False,
)

print("\n" + "=" * 70)
print("ERROR BY DISTANCE")
print("=" * 70)

print(
    distance_analysis.to_string(
        index=False,
        formatters={
            "mae": "${:,.2f}".format,
            "rmse": "${:,.2f}".format,
            "mean_error": "${:,.2f}".format,
            "median_error": "${:,.2f}".format,
        },
    )
)


# ============================================================
# Actual posted-rate bins
# ============================================================

result["actual_rate_bin"] = pd.cut(
    result["actual_rate"],
    bins=[
        0,
        1000,
        1500,
        2000,
        2500,
        3000,
        4000,
        6000,
        10000,
        np.inf,
    ],
    labels=[
        "<1000",
        "1000-1500",
        "1500-2000",
        "2000-2500",
        "2500-3000",
        "3000-4000",
        "4000-6000",
        "6000-10000",
        ">10000",
    ],
)

rate_analysis = grouped_analysis("actual_rate_bin")

rate_analysis.to_csv(
    OUT_DIR / "by_actual_rate.csv",
    index=False,
)

print("\n" + "=" * 70)
print("ERROR BY ACTUAL RATE")
print("=" * 70)

print(
    rate_analysis.to_string(
        index=False,
        formatters={
            "mae": "${:,.2f}".format,
            "rmse": "${:,.2f}".format,
            "mean_error": "${:,.2f}".format,
            "median_error": "${:,.2f}".format,
        },
    )
)


# ============================================================
# Weight bins
# ============================================================

result["weight_bin"] = pd.cut(
    result["weight"],
    bins=[
        -np.inf,
        10000,
        20000,
        30000,
        40000,
        50000,
        np.inf,
    ],
    labels=[
        "<=10k",
        "10k-20k",
        "20k-30k",
        "30k-40k",
        "40k-50k",
        ">50k",
    ],
)

weight_analysis = grouped_analysis("weight_bin")

weight_analysis.to_csv(
    OUT_DIR / "by_weight.csv",
    index=False,
)

print("\n" + "=" * 70)
print("ERROR BY WEIGHT")
print("=" * 70)

print(
    weight_analysis.to_string(
        index=False,
        formatters={
            "mae": "${:,.2f}".format,
            "rmse": "${:,.2f}".format,
            "mean_error": "${:,.2f}".format,
            "median_error": "${:,.2f}".format,
        },
    )
)


# ============================================================
# Route analysis
# ============================================================

result["route"] = (
    result["pickup"].astype(str)
    + " -> "
    + result["delivery"].astype(str)
)

route_analysis = (
    result.groupby("route")
    .agg(
        rows=("load_id", "size"),
        mae=("absolute_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(x.mean())),
        mean_error=("error", "mean"),
        actual_rate_mean=("actual_rate", "mean"),
    )
    .reset_index()
)

# Only consider routes with at least 10 observations
route_analysis = route_analysis[
    route_analysis["rows"] >= 10
].sort_values(
    "mae",
    ascending=False,
)

route_analysis.to_csv(
    OUT_DIR / "by_route.csv",
    index=False,
)

print("\n" + "=" * 70)
print("TOP 15 ROUTES BY MAE (MIN 10 LOADS)")
print("=" * 70)

print(
    route_analysis.head(15).to_string(
        index=False,
        formatters={
            "mae": "${:,.2f}".format,
            "rmse": "${:,.2f}".format,
            "mean_error": "${:,.2f}".format,
            "actual_rate_mean": "${:,.2f}".format,
        },
    )
)


# ============================================================
# Largest individual errors
# ============================================================

worst = result[
    [
        "load_id",
        "pickup",
        "delivery",
        "equipment",
        "distance",
        "weight",
        "actual_rate",
        "predicted_rate",
        "error",
        "absolute_error",
    ]
].sort_values(
    "absolute_error",
    ascending=False,
)

worst.head(100).to_csv(
    OUT_DIR / "worst_100_predictions.csv",
    index=False,
)


# ============================================================
# Error contribution
# ============================================================

result["error_squared_contribution"] = (
    result["squared_error"]
    / result["squared_error"].sum()
)

result["absolute_error_contribution"] = (
    result["absolute_error"]
    / result["absolute_error"].sum()
)

result.sort_values(
    "absolute_error",
    ascending=False,
).head(100).to_csv(
    OUT_DIR / "top_error_contributors.csv",
    index=False,
)


# ============================================================
# Save complete annotated validation set
# ============================================================

result.to_csv(
    OUT_DIR / "validation_error_analysis.csv",
    index=False,
)


# ============================================================
# Save summary JSON
# ============================================================

summary = {
    "rows": int(len(result)),
    "mae": float(mae),
    "rmse": float(rmse),
    "mean_signed_error": float(bias),
    "median_absolute_error": float(median_abs_error),
    "p90_absolute_error": float(p90_abs_error),
    "p95_absolute_error": float(p95_abs_error),
    "p99_absolute_error": float(p99_abs_error),
}

with open(
    OUT_DIR / "summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)


print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(f"Results saved to: {OUT_DIR}")
