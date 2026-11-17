# ECL Alpaca IEX Return Regression v1

## Decision

This experiment does **not** establish the required genuine out-of-sample accuracy above 55%, and no candidate is selected for promotion. Ridge has the strongest pooled walk-forward directional result, but its 54.0% accuracy is only a small improvement over the matched saved XGBoost classifier (52.6%), is not consistently better fold by fold, and does not transfer to the reused validation period (48.9%). HistGradientBoosting reaches 55.4% directional accuracy on validation, but its walk-forward pooled result is 51.5%; validation has already been used in earlier research and is not an untouched test.

The models predict a continuous future return in **percentage points**. For the primary binary result, predicted return greater than zero maps to UP; zero or below maps to DOWN. No threshold search was performed. Neutral-labeled examples remain in regression fitting, while directional accuracy and class metrics are calculated only on actual UP/DOWN examples. A separate all-candle metric counts every actual NEUTRAL example as incorrect.

## Data and protocol

- Instrument/provider/feed: ECL / Alpaca / IEX; five-minute bars.
- Target: existing close-to-close return to the exact 15-minute endpoint, using the established same-session continuous-path eligibility; labels do not cross a session or missing-candle discontinuity.
- Approved raw source: 16,060 rows, `2025-11-13 14:30 UTC` through `2026-09-24 19:55 UTC`.
- Existing chronological train: 4,883 rows through `2026-08-05`; validation: 1,161 rows from `2026-08-06` through `2026-09-24`.
- Features: the exact 24 ordered B features. No feature engineering was added.
- Walk-forward: four expanding chronological folds; fitted imputation/scaling is inside each training fold. The split dates leave the 15-minute labels inside their source session; the runner additionally purges any training row whose target endpoint reaches the held-out decision boundary.
- Predeclared regressors: Ridge, Random Forest Regressor, XGBoost Regressor, and HistGradientBoosting Regressor, with fixed modest configurations in `configuration.json`.
- Baseline comparison: previously saved v3 RF and XGBoost binary predictions, aligned to identical decision timestamps. These saved baseline configurations were chosen in prior work, so this is a matched descriptive comparison, not nested model selection.
- The existing validation period was evaluated once after the walk-forward configuration was fixed. It is a limited confirmation only.

Target audit recomputed the saved continuous return against permitted raw close prices and verified the decision, +5, +10, and +15 minute timestamps and target class. All 16,060 source rows are within the approved pre-test range. No network access was performed. The protected test CSV, its predictions, and post-2026-09-24 observations were not accessed.

## Pooled walk-forward comparison

The 2,248 pooled fold-evaluation rows contain 1,188 actual directional observations (635 DOWN, 553 UP) and 1,060 NEUTRAL observations. The table's directional metrics exclude NEUTRAL; the regression MAE/RMSE/correlation use all 2,248 returns. Confusion matrices use actual rows and predicted columns in `[DOWN, UP]` order.

| Model | Directional accuracy | Balanced accuracy | Macro F1 | DOWN P / R | UP P / R | Directional prediction mix UP | MAE (pp) | RMSE (pp) | Return corr. | All-candle accuracy* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge | **54.0%** | **53.2%** | **52.9%** | 56.0% / 65.0% | 50.8% / 41.4% | 38.0% | **0.1363** | **0.1808** | 0.024 | 28.6% |
| Random Forest Regressor | 50.9% | 50.4% | 50.3% | 53.8% / 58.1% | 47.0% / 42.7% | 42.3% | 0.1394 | 0.1860 | 0.016 | 26.9% |
| XGBoost Regressor | 51.3% | 50.9% | 50.8% | 54.2% / 57.8% | 47.6% / 43.9% | 43.0% | 0.1382 | 0.1847 | 0.046 | 27.1% |
| HistGradientBoosting Regressor | 51.5% | 51.0% | 51.0% | 54.4% / 57.8% | 47.8% / 44.3% | 43.2% | 0.1438 | 0.1929 | 0.011 | 27.2% |
| Saved binary Random Forest | 51.3% | 50.2% | 49.4% | 53.6% / 66.5% | 46.9% / 34.0% | 33.8% | — | — | — | — |
| Saved binary XGBoost | 52.6% | 52.2% | 52.1% | 55.3% / 58.7% | 49.0% / 45.6% | 43.3% | — | — | — | — |
| Always DOWN (majority) | 53.5% | 50.0% | 34.0% | 53.5% / 100% | 0% / 0% | 0% | — | — | — | — |

\* All-candle accuracy divides correctly predicted directional rows by all rows, so actual NEUTRAL rows count as errors. The saved baseline aggregate metrics cover directional rows; their all-candle result is therefore not reported in this table.

Pooled confusion matrices (actual rows, predicted columns; `[DOWN, UP]`):

| Model | Confusion matrix |
|---|---|
| Ridge | `[[413, 222], [324, 229]]` |
| Random Forest Regressor | `[[369, 266], [317, 236]]` |
| XGBoost Regressor | `[[367, 268], [310, 243]]` |
| HistGradientBoosting Regressor | `[[367, 268], [308, 245]]` |
| Saved binary Random Forest | `[[422, 213], [365, 188]]` |
| Saved binary XGBoost | `[[373, 262], [301, 252]]` |

Ridge's 54.0% is 1.4 percentage points above the matched saved XGBoost classifier and 2.7 points above the saved RF classifier, but remains below 55% and only narrowly above the always-DOWN majority baseline. Its UP recall is 41.4%, versus DOWN recall 65.0%, so its class behavior is not balanced enough to establish a dependable directional model. Regression errors are also weakly associated with realized returns: correlation is at most 0.046 pooled, indicating little continuous-return signal despite modest sign accuracy.

## Fold-by-fold results

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Folds ≥55% |
|---|---:|---:|---:|---:|---:|
| Ridge | 54.6% | 51.3% | **57.4%** | 54.3% | 1/4 |
| Random Forest Regressor | 51.0% | 49.3% | **56.3%** | 49.3% | 1/4 |
| XGBoost Regressor | 52.1% | 48.1% | **56.3%** | 51.0% | 1/4 |
| HistGradientBoosting Regressor | 52.6% | 52.2% | 54.7% | 47.4% | 0/4 |
| Saved binary Random Forest | 55.1% | 46.3% | 57.9% | 48.3% | 2/4 |
| Saved binary XGBoost | **57.1%** | 47.2% | **61.1%** | 48.0% | 2/4 |

Fold samples were 361, 335, 190, and 302 directional examples, respectively. Ridge is not uniformly stronger: it is below saved XGBoost in folds 1 and 3 and above it in folds 2 and 4. Each regressor has material period variation. Full fold metrics, including class-wise precision/recall, prediction mix, MAE, RMSE, correlation, training/prediction time, and confusion matrices, are in `walk_forward_fold_metrics.csv`; the saved binary baselines are in `walk_forward_matched_baseline_metrics.csv`.

## Limited validation confirmation

Validation has 1,161 rows: 538 directional (277 DOWN, 261 UP) and 623 NEUTRAL. All model comparisons use those same timestamps. Binary directional prediction coverage is 100% of eligible UP/DOWN observations because the fixed sign rule always emits a class.

| Model | Directional accuracy | Balanced accuracy | Macro F1 | DOWN P / R | UP P / R | UP share of directional predictions | MAE (pp) | RMSE (pp) | Return corr. | All-candle accuracy* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 48.9% | 48.2% | 45.9% | 50.3% / 70.4% | 45.3% / 26.1% | 27.9% | 0.1202 | 0.1620 | -0.003 | 22.7% |
| Random Forest Regressor | 52.6% | 51.8% | 48.3% | 52.6% / 79.1% | 52.5% / 24.5% | 22.7% | **0.1193** | **0.1609** | 0.056 | 24.4% |
| XGBoost Regressor | 51.9% | 51.2% | 48.9% | 52.3% / 73.6% | 50.7% / 28.7% | 27.5% | 0.1202 | 0.1619 | 0.035 | 24.0% |
| HistGradientBoosting Regressor | **55.4%** | **55.0%** | **54.3%** | 55.4% / 69.0% | 55.4% / 41.0% | 35.9% | 0.1211 | 0.1634 | 0.068 | 25.7% |
| Saved binary Random Forest | 53.5% | 52.8% | 50.5% | 53.4% / 76.2% | 53.8% / 29.5% | 26.6% | — | — | — | — |
| Saved binary XGBoost | 47.2% | 46.8% | 45.9% | 49.0% / 61.0% | 44.0% / 32.6% | 35.9% | — | — | — | — |

\* Neutral actuals count as errors; this is not trading profitability.

Validation confusion matrices, `[DOWN, UP]`:

| Model | Confusion matrix |
|---|---|
| Ridge | `[[195, 82], [193, 68]]` |
| Random Forest Regressor | `[[219, 58], [197, 64]]` |
| XGBoost Regressor | `[[204, 73], [186, 75]]` |
| HistGradientBoosting Regressor | `[[191, 86], [154, 107]]` |
| Saved binary Random Forest | `[[211, 66], [184, 77]]` |
| Saved binary XGBoost | `[[169, 108], [176, 85]]` |

HistGradientBoosting is the only candidate above 55% on validation, but that single result conflicts with its lower pooled walk-forward accuracy and comes from a previously reused period. It also gets only 25.7% of all validation candles directionally correct when neutrals count as errors. It cannot be treated as proof of the stated objective.

## Interpretation and recommendation

Continuous-return fitting did not produce repeatable improvement. Ridge's pooled directional result is modestly above the saved classifiers but not consistently so across time; tree regressors have lower pooled sign accuracy, and all regressors have near-zero return correlation. The validation leader is not the walk-forward leader. A high point estimate on the reused validation period is insufficient to choose or promote a model.

**Recommendation:** retain the existing operational model unchanged and do not integrate a regression candidate. Do not claim the >55% genuine out-of-sample objective was met. A future test would require a newly reserved chronological period that has not been used for model selection; the existing test set remains out of scope and is not reopened here.

## Reproduction and files

From the project root:

```powershell
.\.venv\Scripts\python.exe models\alpaca_return_regression_v1\run_return_regression.py
```

Main files:

- `models/alpaca_return_regression_v1/configuration.json` — fixed configurations and protocol.
- `models/alpaca_return_regression_v1/run_return_regression.py` — standalone runner with approved-path checks and target audit.
- `models/alpaca_return_regression_v1/test_return_regression.py` — methodology and leakage tests.
- `models/alpaca_return_regression_v1/reproducibility_manifest.json` — input/output SHA-256 values, package versions, seeds and access declarations.
- `reports/alpaca_return_regression_v1/walk_forward_fold_metrics.csv` and `walk_forward_overall_metrics.csv` — fold and pooled model metrics.
- `reports/alpaca_return_regression_v1/walk_forward_matched_baseline_metrics.csv` — saved RF/XGBoost classifier comparisons on matching timestamps.
- `reports/alpaca_return_regression_v1/validation_metrics.csv` and `validation_baseline_metrics.csv` — one-time limited confirmation.
- `reports/alpaca_return_regression_v1/walk_forward_*_predictions.csv` and `validation_*_predictions.csv` — auditable predictions and returns.
- `reports/alpaca_return_regression_v1/walk_forward_majority_baselines.csv` and `validation_majority_baselines.csv` — always-UP/DOWN and training-majority references.

## Verification

- Focused tests: **12 passed**.
- Python compile check: **passed**.
- Reproducibility manifest: all 9 approved-input hashes and all 36 recorded artifact hashes matched.
- XGBoost 3.4.1 and scikit-learn 1.9.1 were available; no package substitution or installation was made.
- No network access, live application changes, production model changes, or previous research modifications were made.
- `test_csv_accessed=false`; `post_2026_09_24_data_accessed=false` in the experiment manifest.
