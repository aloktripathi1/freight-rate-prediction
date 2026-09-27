## Final Model Selection

### Selected configuration

The final model was selected after chronological validation across
multiple future windows.

- Model: ExtraTreesRegressor
- n_estimators: 100
- min_samples_leaf: 20
- random_state: 42
- Target: log(posted_rate / distance)
- Prediction reconstruction:
  exp(predicted_log_rpm) * distance

### Preprocessing

- Negative weight values were treated as invalid and converted to NaN.
- Numeric missing values were median-imputed.
- Categorical missing values were mode-imputed.
- Categorical variables were one-hot encoded.
- Unknown categorical values are ignored.

### Validation strategy

The supplied labeled development data covers January through October
2025. The primary development validation was chronological, with
October used as a future holdout.

Temporal cross-validation was then performed using three expanding
future-window folds:

- Jan-Jun → Jul-Aug
- Jan-Jul → Aug-Sep
- Jan-Sep → Oct

Mean temporal validation performance:

- MAE: $121.13
- RMSE: $633.74
- R²: 0.8230
- MAE standard deviation across folds: $2.21

The supplied November-December validation set was kept untouched during
model selection.

### Experiment conclusions

- EXP-001: baseline Random Forest — benchmark established.
- EXP-002: negative-weight correction — no measurable improvement.
- EXP-003: Random Forest min_samples_leaf=5 — improved generalization.
- EXP-004: rate-per-mile target — improved validation performance.
- EXP-005: log rate-per-mile target — slightly lower MAE, but mixed metrics.
- EXP-006: historical target-derived pricing features — rejected due to
  substantially worse future validation performance.
- EXP-007: calendar features — rejected due to worse future validation.
- EXP-008: explicit route categorical feature — negligible improvement.
- EXP-009: Extra Trees — improved MAE.
- EXP-010: Extra Trees min_samples_leaf=10 — further improved MAE.
- EXP-011: Extra Trees min_samples_leaf=20 — further improved all key
  validation metrics.
- EXP-012: temporal cross-validation — confirmed stability of the selected
  configuration across multiple future windows.

### Final prediction

The final model was retrained on all 48,000 labeled development rows and
used to generate predictions for all 12,000 rows in validation.csv.

The official scorer validated:

- 12,000 final predictions
- 31 December predictions
- required prediction structure
- positive prediction values

The generated December chart is:
scorer_results/candidate_december.png