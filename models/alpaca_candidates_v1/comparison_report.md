# Alpaca ECL candidate model comparison v1

Validation-only comparison on 1,161 rows (2026-08-06 through 2026-09-24). Models fit on the chronological training split only. No test data was opened.

| Candidate | Accuracy | Balanced accuracy | Macro-F1, 3-class | Directional precision | Directional recall | Directional F1 | Coverage | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Most-frequent dummy | 0.537 | 0.333 | 0.233 | 0.000 | 0.000 | 0.000 | 0.0% | 16.702 |
| Multinomial logistic regression | 0.529 | 0.338 | 0.261 | 0.243 | 0.024 | 0.044 | 4.7% | 1.009 |
| Conservative unweighted LightGBM | 0.489 | 0.354 | 0.336 | 0.282 | 0.133 | 0.178 | 22.0% | 1.031 |
| Balanced LightGBM, old B settings | 0.412 | 0.378 | **0.374** | 0.282 | 0.322 | **0.299** | 53.4% | 1.138 |

The majority baseline predicts NEUTRAL for every validation row: accuracy 0.537, balanced accuracy 0.333, three-class macro-F1 0.233, and zero directional coverage. Its degenerate probabilities make log loss very poor, so that score is not a useful comparison of fitted probability quality.

The balanced LightGBM has the strongest validation macro-F1 and directional macro-F1, and the highest balanced accuracy. It predicts directions on 53.4% of rows. However, its accuracy (0.412) is below the majority baseline, and its log loss (1.138) is worse than logistic regression (1.009). The validation results support carrying it forward as the leading **experimental directional-classification candidate**, but do not establish an overall predictive or trading improvement.

The conservative unweighted LightGBM has lower directional coverage (22.0%) and directional F1 (0.178). Logistic regression has the best non-degenerate log loss, but predicts directions on only 4.7% of rows and has directional macro-F1 0.044.

## Recommendation

Carry `lightgbm_balanced_B_start` forward only as the leading experimental candidate because it best balances the stated validation macro-F1 and directional-usefulness goals while retaining signal coverage. Keep its lower accuracy and weaker log loss explicit; this is not a deployment or profitability claim.

## Implementation notes

- The model inputs are the exact 24 frozen B feature columns in the frozen order. `day_of_week` is treated categorically and one-hot encoded; logistic regression scales numeric features. Preprocessing is fitted only on training data.
- Labels, probabilities, and confusion matrices use DOWN, NEUTRAL, UP order. Directional metrics macro-average one-vs-rest metrics for DOWN and UP. Coverage is the fraction predicted UP or DOWN.
- Per-class metrics and confusion matrices are in `validation_metrics_full.json` and `validation_confusion_matrices.csv`; per-row probabilities are in `validation_predictions.csv`.
- The rebuild CSV has a repeated `close` header (metadata close and model feature close). The loader verified that the values match and used the `close` feature. The source datasets were not modified.
- No hyperparameter search, threshold tuning, validation fitting, test access, or trading simulation was performed.
