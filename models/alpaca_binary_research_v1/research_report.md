# ECL Alpaca IEX Binary Direction Research v1

## Scope and method

Trained fixed XGBoost and LightGBM binary classifiers on directional training rows only. Neutral rows were excluded from binary fitting and eligible-direction binary metrics. Selective outputs use UP, DOWN, or NO SIGNAL. Confidence cutoffs target 20%, 30%, 40%, and 50% of all validation candles; thresholds are score quantiles, with ties potentially increasing realized coverage. The frozen three-class candidate is inference-only. No model or threshold used test data.

Training: 4,883 rows (2,519 directional), 2025-11-13 11:10:00 to 2026-08-05 15:40:00.
Validation: 1,161 rows; 538 directional eligible rows and 623 neutral rows; 2026-08-06 14:35:00 to 2026-09-24 12:25:00.

## Core binary comparison

| Model | Eligible binary accuracy | Directional macro F1 | UP precision | DOWN precision | Log loss |
|---|---:|---:|---:|---:|---:|
| xgboost_binary | 0.504 | 0.503 | 0.489 | 0.518 | 0.7124 |
| lightgbm_binary | 0.533 | 0.533 | 0.519 | 0.547 | 0.7213 |

## Selective validation results

UP/DOWN precision, directional accuracy, recall, and F1 in the main table exclude actual neutral labels, as required for the binary directional evaluation. The separate all-emission accuracy and operational precision fields in `selective_metrics.csv` also count emitted signals on neutral actuals as errors; NO SIGNAL is counted explicitly.

| Model | Target coverage | Actual coverage | Signals | NO SIGNAL | UP precision* | DOWN precision* | Macro precision* | Directional accuracy* | All-emission accuracy** | Macro F1* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgboost_binary | 20% | 20.1% | 233 | 928 | 0.556 | 0.530 | 0.543 | 0.541 | 0.258 | 0.184 |
| xgboost_binary | 30% | 30.1% | 349 | 812 | 0.548 | 0.517 | 0.533 | 0.531 | 0.244 | 0.243 |
| xgboost_binary | 40% | 40.1% | 465 | 696 | 0.530 | 0.464 | 0.497 | 0.495 | 0.224 | 0.279 |
| xgboost_binary | 50% | 50.0% | 581 | 580 | 0.558 | 0.470 | 0.514 | 0.513 | 0.231 | 0.336 |
| lightgbm_binary | 20% | 20.1% | 233 | 928 | 0.533 | 0.532 | 0.533 | 0.533 | 0.245 | 0.176 |
| lightgbm_binary | 30% | 30.1% | 349 | 812 | 0.569 | 0.506 | 0.538 | 0.534 | 0.223 | 0.228 |
| lightgbm_binary | 40% | 40.1% | 465 | 696 | 0.589 | 0.505 | 0.547 | 0.544 | 0.228 | 0.290 |
| lightgbm_binary | 50% | 50.0% | 581 | 580 | 0.538 | 0.508 | 0.523 | 0.523 | 0.219 | 0.326 |

* Binary directional metrics exclude actual NEUTRAL rows as specified. ** All-emission metric treats actual NEUTRAL rows as incorrect directional signals, a stricter operational view.

## Reference baselines

Training-majority directional baseline predicts DOWN for every validation candle; causal one-candle return-sign baseline predicts UP for positive `return_1`, otherwise DOWN. Their metrics on eligible actual directional rows are in `metrics.json`. Always-NEUTRAL obtains 53.7% accuracy on all validation rows as three-class context, but it does not make a directional prediction. The frozen three-class candidate is shown separately with raw argmax and neutral coverage in that file.

## Robustness and calibration

Daily and training-median volatility-split diagnostics for the predeclared 30% policy are saved in `daily_robustness_30pct.csv` and `regime_robustness_30pct.csv`. Probability calibration (log loss, Brier score, ECE, distribution quantiles) is in `calibration_metrics.csv`, with reliability bins in `calibration_bins.csv`; per-date confusion matrices and policy-level matrices are in `confusion_matrices.csv`. Validation has been used in earlier project research and is not a pristine holdout.

## Decision

The predeclared rule selects **lightgbm_binary** at the 30% target (actual coverage 30.1%, 349 signals). Among eligible actual UP/DOWN examples, UP precision is 56.9%, DOWN precision is 50.6%, macro precision is 53.8%, and emitted directional accuracy is 53.4%. Counting neutral candles as false operational signals, macro precision is 22.4% and all-emission accuracy is 22.3%.
The precision and accuracy goals are reported independently below and are not guaranteed; judge each requested coverage row rather than extrapolating from the 30% policy.

## Target check and limitations

The 65% directional precision target is evaluated using macro UP/DOWN precision on eligible actual directional examples. The 55% accuracy target is evaluated using binary accuracy on eligible directional examples, including both all eligible rows and the emitted subset at each coverage. The best validation macro precision is 54.7% (LightGBM at 40% coverage); the best emitted eligible-direction accuracy is 54.4% (same policy). Neither target was reached. Strict all-candle emitted accuracy is much lower because the binary model also emits signals on neutral actuals. Daily variation is substantial: at the predeclared 30% target, among dates with at least five emitted eligible directional examples, 9 of 15 dates per model were above 50% accuracy; date-level standard deviations were about 22–27 percentage points. The simple training-volatility-median split also varies, so results are not stable enough to claim robust gains. This experiment establishes no profitability or production readiness.

No additional historical source, test dataset, live API, or test-period observation was used. This is research only and is not integrated into the live dashboard.
