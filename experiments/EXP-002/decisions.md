## EXP-002 — Negative Weight Correction

### Change
Converted negative `weight` values to NaN before the existing
median-imputation pipeline.

- Negative weights corrected: 292
- All other experiment conditions remained unchanged.

### Results
- Training MAE: $49.77
- Training RMSE: $230.53
- Training R²: 0.9758
- October validation MAE: $162.72
- October validation RMSE: $691.21
- October validation R²: 0.7955

### Interpretation
Negative-weight correction did not improve October generalization.
Validation MAE increased slightly from $162.30 to $162.72, RMSE
increased slightly, and R² decreased from 0.7958 to 0.7955.

The negative-weight issue is still a data-quality problem, but this
experiment provides no evidence that correcting it improves predictive
performance for the current model.

### Decision
Do not treat negative-weight correction as a performance improvement.
The next experiment will focus on reducing Random Forest overfitting.