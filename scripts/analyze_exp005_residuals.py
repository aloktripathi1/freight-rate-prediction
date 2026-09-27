from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "train-test.csv"
PRED = ROOT / "experiments" / "EXP-005" / "validation_predictions.csv"

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])

val = df[
    (df["date"] >= "2025-10-01") &
    (df["date"] < "2025-11-01")
].copy()

pred = pd.read_csv(PRED)

val = val.merge(
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
    validate="one_to_one",
)

val["rpm_residual"] = (
    val["actual_rate_per_mile"]
    - val["predicted_rate_per_mile"]
)

val["absolute_rate_error"] = (
    val["actual_rate"] - val["predicted_rate"]
).abs()

print("\n" + "=" * 80)
print("RPM RESIDUAL CORRELATION")
print("=" * 80)

numeric_cols = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "actual_rate_per_mile",
]

print(
    val[numeric_cols + ["rpm_residual"]]
    .corr()["rpm_residual"]
    .sort_values()
    .to_string()
)


def analyze_bins(column, bins=5):
    temp = val.copy()

    temp["bin"] = pd.qcut(
        temp[column],
        q=bins,
        duplicates="drop",
    )

    out = (
        temp.groupby("bin", observed=True)
        .agg(
            rows=("load_id", "size"),
            actual_rpm=("actual_rate_per_mile", "mean"),
            predicted_rpm=("predicted_rate_per_mile", "mean"),
            rpm_residual=("rpm_residual", "mean"),
            mae=("absolute_rate_error", "mean"),
            actual_rate=("actual_rate", "mean"),
        )
        .reset_index()
    )

    return out


for column in [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]:
    print("\n" + "=" * 80)
    print(f"ERROR BY {column.upper()} QUANTILE")
    print("=" * 80)

    print(
        analyze_bins(column).to_string(
            index=False,
            formatters={
                "actual_rpm": "{:.3f}".format,
                "predicted_rpm": "{:.3f}".format,
                "rpm_residual": "{:.3f}".format,
                "mae": "${:,.2f}".format,
                "actual_rate": "${:,.2f}".format,
            },
        )
    )


print("\n" + "=" * 80)
print("ERROR BY EQUIPMENT")
print("=" * 80)

equipment = (
    val.groupby("equipment")
    .agg(
        rows=("load_id", "size"),
        actual_rpm=("actual_rate_per_mile", "mean"),
        predicted_rpm=("predicted_rate_per_mile", "mean"),
        rpm_residual=("rpm_residual", "mean"),
        mae=("absolute_rate_error", "mean"),
    )
    .reset_index()
)

print(
    equipment.to_string(
        index=False,
        formatters={
            "actual_rpm": "{:.3f}".format,
            "predicted_rpm": "{:.3f}".format,
            "rpm_residual": "{:.3f}".format,
            "mae": "${:,.2f}".format,
        },
    )
)


print("\n" + "=" * 80)
print("TOP 20 RPM UNDERPREDICTIONS")
print("=" * 80)

cols = [
    "load_id",
    "pickup",
    "delivery",
    "equipment",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "actual_rate",
    "predicted_rate",
    "actual_rate_per_mile",
    "predicted_rate_per_mile",
    "rpm_residual",
]

print(
    val.sort_values(
        "rpm_residual",
        ascending=False,
    )[cols]
    .head(20)
    .to_string(index=False)
)
