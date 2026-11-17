# Alpaca final candidate freeze audit v1

All pre-freeze checks passed. This freeze is a research candidate; it does not replace the prior official model freeze.

| Check | Result | Evidence |
|---|---|---|
| Only authorized splits used | PASS | Training manifest names train.csv and validation.csv only; this audit has no test-split reader. |
| Training and validation hashes match configuration | PASS | The two allowed split hashes match training_config.json. |
| Chronology and horizon purge | PASS | Every training decision and target endpoint precedes validation. |
| Exact 24-feature order | PASS | Both split inputs and configuration match frozen B feature order. |
| No target/time metadata as model inputs | PASS | The model receives only the explicit feature allowlist. |
| Feature types | PASS | day_of_week is integer/categorical; all features are numeric; validation matches. |
| Candidate loads in fresh process | PASS | The audit is executed as a standalone Python process; joblib loaded the saved candidate. |
| Pipeline accepts 24 ordered inputs | PASS | Pipeline and preprocessor feature_names_in_ equal the expected ordered feature list. |
| Preprocessing and estimator are saved | PASS | The artifact is a fitted sklearn Pipeline containing ColumnTransformer and LightGBM. |
| Class order | PASS | Saved estimator classes: ['DOWN', 'NEUTRAL', 'UP']. |
| Preprocessor fitted from training only | PASS | Imputer medians, weekday mode (2), and encoder categories match train.csv; validation values were not passed to fit. |
| Fresh validation probabilities | PASS | 1161 rows; finite nonnegative probabilities; sums within 1e-10 of 1. |
| Saved validation outputs reproduce | PASS | Fresh-process probabilities/classes/timestamps/actual labels match saved validation_predictions.csv. |
| Artifact reproduces reported metrics | PASS | Fresh validation recalculation matches all eight metrics to 1e-12: {'accuracy': 0.41171404, 'balanced_accuracy': 0.37801106, 'macro_f1_3class': 0.37406731, 'directional_precision_macro': 0.28248022, 'directional_recall_macro': 0.32223329, 'directional_f1_macro': 0.29907348, 'directional_coverage': 0.53402239, 'log_loss': 1.1380294} |
| Model parameters match config | PASS | Saved LightGBM estimator parameters match training_config.json. |
| Old frozen artifacts unchanged | PASS | B: PASS; C: PASS; B_ADVANCED_NO_MOMENTUM: PASS |
| No Twelve Data references | PASS | No provider mentions in new candidate scripts/config/reports. |
| Training code fits on training split only | PASS | Candidate script loads train/validation separately, fits on X_train/y_train, and contains no test loader. |

No test data was opened, read, loaded, hashed, or evaluated by this audit. No live API call or trade occurred.
