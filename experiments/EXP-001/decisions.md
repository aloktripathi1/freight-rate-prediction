## EXP-001 — Baseline Results

### Results

- Training MAE: $50.03
- Training RMSE: $230.92
- Training R²: 0.9757
- October validation MAE: $162.30
- October validation RMSE: $690.70
- October validation R²: 0.7958

### Interpretation

The baseline model achieves a substantially better score on the training
data than on the October holdout. The validation MAE is approximately
3.2x the training MAE and the R² decreases from 0.9757 to 0.7958.

This indicates meaningful generalization error and potential overfitting.
The result establishes the benchmark for subsequent experiments.

### Implementation Note

The initial baseline implementation did not yet convert negative weight
values to missing values, despite the preprocessing decision to treat
negative weights as invalid. This discrepancy will be corrected in
EXP-002 while keeping all other experimental conditions unchanged.