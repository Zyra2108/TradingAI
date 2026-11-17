# ECL target and horizon research v1

## Scope and data boundary

Only the existing Alpaca IEX `train.csv` and `validation.csv` were used for target generation, model fitting, and evaluation. The CSVs expose close plus the existing 24 causal B features, not raw open/high/low/volume; labels therefore use exact close-to-close returns. No volume was synthesized or used. No frozen test CSV was opened, hashed, or evaluated.
The validation period has prior research use and is not a pristine holdout. A data-inspection process exception is disclosed at the end of this report.

## Predeclared matrix and method

The matrix in `experiment_matrix.json` fixes 15 target definitions across 10/15/30/60-minute horizons: fixed ±0.10%, square-root horizon-scaled fixed thresholds, selected ±0.15% cases, 0.5× historical-volatility-scaled thresholds, and minimum-movement event floors. Small movements are NO_SIGNAL. Volatility-scaled labels use same-row `volatility_20` only; horizon return uses the close at the exact future timestamp.
Targets are generated separately from model inputs. Every endpoint requires all expected 5-minute timestamps through the endpoint and the same trading date/session; unavailable targets are excluded, never interpolated. Three expanding date-ordered folds purge fitting labels ending at or beyond the evaluation start. Fold preprocessing is independently fitted.
Logistic Regression, LightGBM, and XGBoost use fixed modest settings. Thresholds and selection use training-only OOF predictions. The selected OOF policy must have ≥20% eligible event coverage, ≥75 emitted eligible OOF examples, and at least 10 eligible emissions per fold where feasible. No validation labels influence selection.

## OOF selection

Selected target **sqrt_fixed_h60** (60 minutes, fixed_pct_sqrt_horizon), model **logistic_regression**, and OOF-derived 20% eligible-event coverage threshold. Mean fold eligible-only macro precision: 68.5%; mean fold emitted eligible accuracy: 68.5%; mean eligible coverage: 20.0%; OOF eligible emissions: 231.
See fold-by-fold metrics/ranking and selected OOF date robustness CSVs. A threshold that merely creates a tiny number of signals was not eligible for selection.

Selected-candidate fold results:

| Fold | Eligible events | Signals on eligible events | UP precision | DOWN precision | Macro precision | Directional accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 429 | 86 | 73.1% | 80.0% | 76.5% | 77.9% |
| 2 | 415 | 83 | 80.0% | 74.6% | 77.3% | 75.9% |
| 3 | 309 | 62 | 64.5% | 38.7% | 51.6% | 51.6% |

Across 27 OOF dates with at least three eligible signals, daily directional accuracy averaged 71.0% (SD 39.7%). 

## Single selected-candidate validation confirmation

Validation target sqrt_fixed_h60 yielded 376 eligible directional events among 731 available labels. The selected policy emitted 60 predictions and 671 NO_SIGNAL outcomes; eligible coverage was 8.5%. UP precision 90.0%; DOWN precision 59.1%; macro precision 74.5%; emitted eligible accuracy 68.8%; all-candle operational accuracy (neutral event labels count as errors) 36.7%.
The 65% precision / 55% accuracy / 20% coverage goals were not all met on this confirmation. Treat it as a limited confirmation, not independent validation. At validation, 6 dates had at least two eligible emitted signals; their daily directional accuracy averaged 74.4% (SD 35.6%). Daily results are saved in `validation_date_robustness.csv`.

## Recommendation

Do not integrate automatically. If precision or accuracy targets fail at ≥20% eligible coverage, the evidence does not justify another broad model search; consider stopping this modeling direction or revisiting the decision problem with new, independently justified data/labels.

## Process limitation

During initial project inspection, an existing dataset audit report containing test-split aggregate metadata was opened inadvertently. No `test.csv` contents, test-period candle observations, or test predictions were opened or used, and the metadata was not used in this experiment. This exceeded the strict no-test-period-data boundary; this limitation is disclosed rather than hidden.
