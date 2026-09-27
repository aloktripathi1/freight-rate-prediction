# Freight Rate Prediction

An end-to-end machine learning system for predicting freight load rates from shipment, route, equipment, market, and quote-level information.

The project focuses on **time-aware validation, leakage-safe feature engineering, target transformation, error analysis, and reproducible model training** rather than simply optimizing a single random train/validation split.

---

## Overview

Freight pricing is influenced by several interacting factors:

* Pickup and delivery locations
* Shipment distance
* Equipment type
* Load weight
* Market conditions
* Quote signals
* Historical pricing patterns
* Temporal effects

A major challenge in this problem is **temporal generalization**. A model can achieve excellent performance on historical training data while degrading substantially when evaluated on a future month.

The modeling process therefore evolved from a simple baseline into a time-aware pipeline designed to better represent the actual prediction setting.

### Final pipeline

```text
Historical shipment data
        │
        ▼
Data validation & preprocessing
        │
        ▼
Feature engineering
        │
        ├── Geographic / route features
        ├── Distance features
        ├── Equipment features
        ├── Market / quote features
        └── Target transformation
        │
        ▼
Log Rate-per-Mile prediction
        │
        ▼
Inverse transformation
        │
        ▼
Predicted freight rate
        │
        ▼
Submission / evaluation
```

---

## Dataset

The development dataset contains **48,000 labeled loads** covering:

```text
2025-01-01 → 2025-10-31
```

The held-out validation dataset contains **12,000 loads** covering:

```text
2025-11-01 → 2025-12-31
```

The primary input variables include:

| Feature                         | Description             |
| ------------------------------- | ----------------------- |
| `pickup`                        | Pickup location         |
| `delivery`                      | Delivery location       |
| `pickup_lat` / `pickup_lon`     | Pickup coordinates      |
| `delivery_lat` / `delivery_lon` | Delivery coordinates    |
| `distance`                      | Shipment distance       |
| `equipment`                     | Equipment type          |
| `weight`                        | Load weight             |
| `market_index`                  | Market condition signal |
| `quote_signal`                  | Quote-related signal    |

The target is the posted freight rate.

For modeling, the target was transformed into **rate per mile** and subsequently into **log rate per mile**.

---

# Modeling Approach

## 1. Baseline

The project began with a conventional tree-based regression baseline.

The first model achieved:

| Metric | Training | Internal Validation |
| ------ | -------: | ------------------: |
| MAE    |   $50.03 |             $162.30 |
| RMSE   |  $230.92 |             $690.70 |
| R²     |   0.9757 |              0.7958 |

The large difference between training and validation performance immediately indicated a substantial generalization problem.

This became an important direction for subsequent experiments.

---

## 2. Data Quality

During preprocessing, **292 negative training weights** were identified.

Rather than allowing invalid values to enter the modeling pipeline, they were corrected before training.

```text
Negative weights before correction: 292
Negative weights after correction:    0
```

This validation step is part of the final training workflow as well.

---

## 3. Rate-per-Mile Target

Because freight rate is strongly related to shipment distance, the modeling target was transformed from absolute rate into:

```text
posted_rate_per_mile =
    posted_rate / distance
```

The resulting target had:

```text
Mean:    2.2153
Median:  2.1453
Std:     0.5839
Min:     0.3337
Max:    14.1253
```

This representation allowed the model to learn pricing intensity independently of total shipment distance.

---

## 4. Log Target Transformation

The rate-per-mile target was further transformed:

```text
log_posted_rate_per_mile =
    log(posted_rate_per_mile)
```

The model then predicts log-RPM.

The final prediction is reconstructed as:

```text
predicted_rpm = exp(predicted_log_rpm)

predicted_rate =
    predicted_rpm × distance
```

This transformation reduced the influence of extreme rate-per-mile observations and improved validation performance.

---

# Experimentation

The project contains **12 experiments**.

The experiments were used to investigate different hypotheses rather than simply trying arbitrary models.

| Experiment | Focus                                |
| ---------- | ------------------------------------ |
| EXP-001    | Initial baseline                     |
| EXP-002    | Weight correction                    |
| EXP-003    | Alternative model configuration      |
| EXP-004    | Rate-per-mile target                 |
| EXP-005    | Log rate-per-mile target             |
| EXP-006    | Historical pricing features          |
| EXP-007    | Temporal features                    |
| EXP-008    | Explicit route features              |
| EXP-009    | Combined target / feature refinement |
| EXP-010    | Model refinement                     |
| EXP-011    | Further model refinement             |
| EXP-012    | Temporal cross-validation            |

The complete experiment artifacts are stored under:

```text
experiments/
```

Each experiment records the relevant configuration and evaluation metrics.

---

# Error Analysis

After EXP-004, detailed error analysis was performed to understand where the model failed.

### Distance

Validation MAE increased substantially with shipment distance.

| Distance  |     MAE |
| --------- | ------: |
| ≤250      |  $33.90 |
| 251–500   |  $59.71 |
| 501–750   |  $91.40 |
| 751–1000  | $111.19 |
| 1001–1500 | $149.67 |
| 1501–2000 | $200.00 |
| >2000     | $247.82 |

This showed that **long-haul loads were considerably harder to predict** than shorter loads.

### Equipment

The error analysis also showed differences between equipment categories:

| Equipment |     MAE |
| --------- | ------: |
| Dry Van   | $119.13 |
| Flatbed   | $137.08 |
| Reefer    | $168.66 |

The analysis helped identify where additional feature engineering could potentially provide value.

---

# Temporal Validation

Random validation would not accurately represent the intended deployment scenario because future loads should not influence historical training.

Therefore, the project introduced **expanding-window temporal cross-validation**.

### Fold 1

```text
Training:   Jan → Jun
Validation: Jul → Aug

MAE:  $122.28
RMSE: $627.94
R²:   0.8225
```

### Fold 2

```text
Training:   Jan → Jul
Validation: Aug → Sep

MAE:  $118.59
RMSE: $619.98
R²:   0.8290
```

### Fold 3

```text
Training:   Jan → Sep
Validation: Oct

MAE:  $122.53
RMSE: $653.30
R²:   0.8173
```

### Temporal CV Summary

```text
Mean validation MAE:  $121.13
Std validation MAE:     $2.21

Mean validation RMSE: $633.74
Mean validation R²:      0.8230
```

The relatively small variation in validation MAE across the three temporal folds provided a more reliable estimate of model behavior than relying on a single split.

---

# Final Model

After experimentation and temporal validation, the final model was trained using **all 48,000 labeled development samples**.

```text
Training data:
48,000 rows

Training period:
2025-01-01 → 2025-10-31

Prediction period:
2025-11-01 → 2025-12-31
```

The final pipeline predicts log rate-per-mile and converts it back into the predicted freight rate.

### Final prediction distribution

```text
Minimum:  $178.71
Median:   $2,002.81
Mean:     $2,325.31
Maximum:  $6,707.42
```

The trained model is stored at:

```text
final_model/model_pipeline.joblib
```

---

# Reproducing the Project

## 1. Clone the repository

```bash
git clone <repository-url>
cd freight-rate-prediction
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the final training pipeline

```bash
python scripts/train_final.py
```

This trains the final model on all labeled development data and generates:

```text
final_model/model_pipeline.joblib
validation_predictions.csv
final_model/config.json
final_model/prediction_summary.json
```

## 4. Validate the generated predictions

```bash
python score.py \
    --predictions validation_predictions.csv \
    --december-predictions december-chart-inputs.csv
```

---

# Key Engineering Lessons

### 1. A strong training score does not guarantee generalization

The baseline achieved:

```text
Training MAE:    $50.03
Validation MAE: $162.30
```

The large gap motivated deeper investigation instead of simply increasing model complexity.

### 2. Validation strategy matters

Because the prediction task is temporal, random splitting can provide an overly optimistic estimate.

Temporal validation better reflects the actual forecasting setup.

### 3. Domain-derived targets can matter

Predicting absolute freight rate directly was less effective than modeling rate per mile and applying a log transformation.

### 4. Error analysis should drive experimentation

The distance and equipment breakdowns revealed specific regions where the model struggled, providing evidence for subsequent feature engineering.

### 5. More features are not automatically better

Some experiments added historical and temporal features but did not consistently improve future-month validation.

The project therefore prioritizes **validated improvements over feature-count growth**.

---

# Documentation

A detailed technical report covering the problem, experimentation, validation strategy, error analysis, and final modeling decisions is available in:

```text
report/freight_rate_prediction_report.pdf
```

The report source is also provided as LaTeX:

```text
report/freight_rate_prediction_report.tex
```

---

## Status

**Final training pipeline completed.**

The repository contains the complete experimentation history, temporal validation results, final trained model, predictions, evaluation tooling, and technical report.
