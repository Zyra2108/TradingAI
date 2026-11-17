# ECL Alpaca IEX OHLCV Sequence Research v1

## Decision

**No sequence candidate achieved reliable out-of-sample accuracy above 55%. Do not promote or integrate a sequence model.** The highest mean fold accuracy was 54.3% for a 12-candle CNN, but its mean fold balanced accuracy was 50.5% and macro-F1 was 38.4%. It predicted UP only 70 times in 1,188 OOF directional cases. On validation it predicted DOWN for every case at lengths 6 and 12. Longer windows were more balanced in some folds but did not produce stable accuracy above 55% and substantially reduced eligible coverage.

Chronological OOF predictions are out-of-sample within the training period. The existing validation period has prior research exposure and is **not a pristine holdout**. No protected test rows, predictions or metrics were accessed. This experiment does not establish independent future performance.

## Data and eligibility

Source: `models/alpaca_1min_research_v1/raw_ecl_iex_5min_train_validation.csv`, whose metadata specifies ECL, Alpaca IEX, 5Min, November 13, 2025–September 24, 2026. It contains 16,060 bars over 216 New York dates; duplicate timestamps: 0; invalid OHLCV rows: 0; zero-volume rows: 0. True missing intervals were kept as gaps and reset the sequence. See `source_data_audit.json`.

The experiment reuses the approved train/validation labels. It verifies that decision and future endpoint candles exist in the permitted source, the endpoint is exactly 15 minutes later in the same New York session date, and the saved `future_return` maps to the strict existing ±0.10% labels. NEUTRAL rows are excluded from binary fitting and directional metrics.

| Split | All labels | Sequence length | Sequence rows | Directional total | Directional eligible | Coverage | Eligible dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6 | 4883 | 4883 | 2519 | 2519 | 100.0% | 152 |
| train | 12 | 4883 | 4883 | 2519 | 2519 | 100.0% | 152 |
| train | 24 | 4883 | 4338 | 2519 | 2227 | 88.4% | 140 |
| train | 36 | 4883 | 2676 | 2519 | 1367 | 54.3% | 106 |
| validation | 6 | 1161 | 1161 | 538 | 538 | 100.0% | 33 |
| validation | 12 | 1161 | 1161 | 538 | 538 | 100.0% | 33 |
| validation | 24 | 1161 | 1040 | 538 | 472 | 87.7% | 32 |
| validation | 36 | 1161 | 649 | 538 | 287 | 53.3% | 23 |

L6/L12 retained all original labels. L24 retained 87.7% of validation directional samples; L36 retained 53.3%. Every input window ends at its decision candle and requires exact consecutive 5-minute timestamps within the same New York session. No gap is filled or bridged. Volume is the IEX-feed volume only and must not be interpreted as consolidated market-wide volume.

## Input and model

The learned input is a fixed rolling window of raw OHLCV, not indicator formulas: OHLC is expressed as percent distance from the window's final close; volume uses `log1p`. Channel scaling is fitted only on training windows within each fold. The compact CNN is causal Conv1D(8 filters, kernel 3), global average pooling, a regularized 8-unit dense layer, dropout, and sigmoid output. Settings are fixed at 20 epochs, batch 32, Adam 0.001, no class weighting, and no threshold search. UP is predicted when P(UP) ≥ 0.5; otherwise DOWN.

Four established expanding chronological folds were reused. Training examples whose 15-minute endpoint reached the first held-out timestamp were purged. RF and XGBoost predictions saved from the existing v3 experiment were joined by exact timestamp for matched comparisons; they were not re-inferred here. The v3 baseline configuration had been selected using the aggregate of those same OOF folds, so baseline fold results are a descriptive reference rather than a nested, configuration-selection-unbiased comparison. The validation baseline configuration was selected using training OOF only.

## Matched OOF comparison

This table aggregates the four OOF folds on identical eligible timestamps per length. Complete per-fold rows for the CNN and both baselines are saved in `walk_forward_fold_metrics.csv`; overall metrics are in `walk_forward_overall_metrics.csv`.

| Model | L | OOF n | Coverage | Accuracy | Balanced acc. | Macro F1 | Macro precision | DOWN recall | DOWN precision | UP recall | UP precision | Log loss | Confusion [DOWN,UP] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| causal_1d_cnn | 6 | 1188 | 100.0% | 52.9% | 49.6% | 34.9% | 36.6% | 98.7% | 53.2% | 0.4% | 20.0% | 0.692 | [[627, 8], [551, 2]] |
| random_forest | 6 | 1188 | 100.0% | 51.3% | 50.2% | 49.4% | 50.3% | 66.5% | 53.6% | 34.0% | 46.9% | 0.698 | [[422, 213], [365, 188]] |
| xgboost | 6 | 1188 | 100.0% | 52.6% | 52.2% | 52.1% | 52.2% | 58.7% | 55.3% | 45.6% | 49.0% | 0.706 | [[373, 262], [301, 252]] |
| causal_1d_cnn | 12 | 1188 | 100.0% | 53.5% | 50.4% | 39.8% | 51.8% | 94.5% | 53.7% | 6.3% | 50.0% | 0.692 | [[600, 35], [518, 35]] |
| random_forest | 12 | 1188 | 100.0% | 51.3% | 50.2% | 49.4% | 50.3% | 66.5% | 53.6% | 34.0% | 46.9% | 0.698 | [[422, 213], [365, 188]] |
| xgboost | 12 | 1188 | 100.0% | 52.6% | 52.2% | 52.1% | 52.2% | 58.7% | 55.3% | 45.6% | 49.0% | 0.706 | [[373, 262], [301, 252]] |
| causal_1d_cnn | 24 | 1057 | 89.0% | 52.6% | 50.0% | 41.8% | 50.0% | 89.9% | 53.3% | 10.1% | 46.7% | 0.690 | [[506, 57], [444, 50]] |
| random_forest | 24 | 1057 | 89.0% | 50.7% | 49.6% | 48.7% | 49.6% | 66.3% | 53.0% | 33.0% | 46.2% | 0.700 | [[373, 190], [331, 163]] |
| xgboost | 24 | 1057 | 89.0% | 52.1% | 51.6% | 51.5% | 51.7% | 59.3% | 54.7% | 43.9% | 48.7% | 0.709 | [[334, 229], [277, 217]] |
| causal_1d_cnn | 36 | 685 | 57.7% | 51.8% | 50.0% | 46.0% | 49.9% | 79.7% | 53.1% | 20.2% | 46.8% | 0.692 | [[290, 74], [256, 65]] |
| random_forest | 36 | 685 | 57.7% | 48.3% | 47.4% | 46.6% | 47.1% | 62.4% | 51.1% | 32.4% | 43.2% | 0.704 | [[227, 137], [217, 104]] |
| xgboost | 36 | 685 | 57.7% | 50.2% | 49.9% | 49.9% | 49.9% | 54.4% | 53.1% | 45.5% | 46.8% | 0.720 | [[198, 166], [175, 146]] |

The CNN's accuracy did not translate to balanced direction prediction. At L6/L12, UP recall was especially weak (0.4%/6.3% pooled OOF), while XGBoost was better on balanced accuracy and macro-F1 on the same rows. L24 remained behind XGBoost on balanced accuracy and macro-F1. L36 had only about 57.7% OOF eligibility coverage and unstable folds.

## CNN fold results

Confusion matrices use actual rows and predicted columns in DOWN, UP order.

| L | Fold | n | Actual D | Actual U | Pred D | Pred U | Accuracy | Balanced acc. | Macro F1 | D P | D R | U P | U R | Log loss | Confusion [D,U] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 1 | 361 | 200 | 161 | 355 | 6 | 54.8% | 49.6% | 36.5% | 55.2% | 98.0% | 33.3% | 1.2% | 0.691 | [[196, 4], [159, 2]] |
| 6 | 2 | 335 | 170 | 165 | 331 | 4 | 49.6% | 48.8% | 33.1% | 50.2% | 97.6% | 0.0% | 0.0% | 0.695 | [[166, 4], [165, 0]] |
| 6 | 3 | 190 | 119 | 71 | 190 | 0 | 62.6% | 50.0% | 38.5% | 62.6% | 100.0% | 0.0% | 0.0% | 0.681 | [[119, 0], [71, 0]] |
| 6 | 4 | 302 | 146 | 156 | 302 | 0 | 48.3% | 50.0% | 32.6% | 48.3% | 100.0% | 0.0% | 0.0% | 0.697 | [[146, 0], [156, 0]] |
| 12 | 1 | 361 | 200 | 161 | 292 | 69 | 55.7% | 52.4% | 49.0% | 56.8% | 83.0% | 50.7% | 21.7% | 0.691 | [[166, 34], [126, 35]] |
| 12 | 2 | 335 | 170 | 165 | 334 | 1 | 50.4% | 49.7% | 33.5% | 50.6% | 99.4% | 0.0% | 0.0% | 0.694 | [[169, 1], [165, 0]] |
| 12 | 3 | 190 | 119 | 71 | 190 | 0 | 62.6% | 50.0% | 38.5% | 62.6% | 100.0% | 0.0% | 0.0% | 0.682 | [[119, 0], [71, 0]] |
| 12 | 4 | 302 | 146 | 156 | 302 | 0 | 48.3% | 50.0% | 32.6% | 48.3% | 100.0% | 0.0% | 0.0% | 0.697 | [[146, 0], [156, 0]] |
| 24 | 1 | 327 | 181 | 146 | 299 | 28 | 55.4% | 50.9% | 42.8% | 55.9% | 92.3% | 50.0% | 9.6% | 0.689 | [[167, 14], [132, 14]] |
| 24 | 2 | 298 | 150 | 148 | 278 | 20 | 51.0% | 50.7% | 39.5% | 50.7% | 94.0% | 55.0% | 7.4% | 0.693 | [[141, 9], [137, 11]] |
| 24 | 3 | 163 | 102 | 61 | 104 | 59 | 57.1% | 53.8% | 53.8% | 65.4% | 66.7% | 42.4% | 41.0% | 0.683 | [[68, 34], [36, 25]] |
| 24 | 4 | 269 | 130 | 139 | 269 | 0 | 48.3% | 50.0% | 32.6% | 48.3% | 100.0% | 0.0% | 0.0% | 0.694 | [[130, 0], [139, 0]] |
| 36 | 1 | 220 | 126 | 94 | 131 | 89 | 54.1% | 52.8% | 52.8% | 59.5% | 61.9% | 46.1% | 43.6% | 0.690 | [[78, 48], [53, 41]] |
| 36 | 2 | 198 | 100 | 98 | 176 | 22 | 52.5% | 52.1% | 43.8% | 51.7% | 91.0% | 59.1% | 13.3% | 0.690 | [[91, 9], [85, 13]] |
| 36 | 3 | 97 | 64 | 33 | 79 | 18 | 61.9% | 52.0% | 50.8% | 67.1% | 82.8% | 38.9% | 21.2% | 0.686 | [[53, 11], [26, 7]] |
| 36 | 4 | 170 | 74 | 96 | 160 | 10 | 42.4% | 48.0% | 32.8% | 42.5% | 91.9% | 40.0% | 4.2% | 0.700 | [[68, 6], [92, 4]] |

The later folds show substantial instability. In particular, a DOWN-heavy fold can make a model that predicts mostly DOWN look accurate; fold 4 drops below 50% for shorter windows and to 42.4% for L36. No length produces consistent >55% accuracy.

## Validation confirmation (limited diagnostic)

Each predeclared length was trained on train only and evaluated once on the existing validation set. The validation period was not used to change architecture, length, preprocessing or threshold. Saved RF/XGBoost predictions were aligned on the same timestamp subsets.

| Model | L | Directional n | Coverage | Accuracy | Balanced acc. | Macro F1 | DOWN P | DOWN R | UP P | UP R | Macro precision | Log loss | Confusion [D,U] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| causal_1d_cnn | 6 | 538 | 100.0% | 51.5% | 50.0% | 34.0% | 51.5% | 100.0% | 0.0% | 0.0% | 25.7% | 0.693 | [[277, 0], [261, 0]] |
| random_forest | 6 | 538 | 100.0% | 53.5% | 52.8% | 50.5% | 53.4% | 76.2% | 53.8% | 29.5% | 53.6% | 0.696 | [[211, 66], [184, 77]] |
| xgboost | 6 | 538 | 100.0% | 47.2% | 46.8% | 45.9% | 49.0% | 61.0% | 44.0% | 32.6% | 46.5% | 0.712 | [[169, 108], [176, 85]] |
| causal_1d_cnn | 12 | 538 | 100.0% | 51.5% | 50.0% | 34.0% | 51.5% | 100.0% | 0.0% | 0.0% | 25.7% | 0.693 | [[277, 0], [261, 0]] |
| random_forest | 12 | 538 | 100.0% | 53.5% | 52.8% | 50.5% | 53.4% | 76.2% | 53.8% | 29.5% | 53.6% | 0.696 | [[211, 66], [184, 77]] |
| xgboost | 12 | 538 | 100.0% | 47.2% | 46.8% | 45.9% | 49.0% | 61.0% | 44.0% | 32.6% | 46.5% | 0.712 | [[169, 108], [176, 85]] |
| causal_1d_cnn | 24 | 472 | 87.7% | 52.3% | 50.3% | 35.9% | 52.3% | 98.8% | 57.1% | 1.8% | 54.7% | 0.691 | [[243, 3], [222, 4]] |
| random_forest | 24 | 472 | 87.7% | 53.4% | 52.4% | 50.3% | 53.8% | 75.2% | 52.3% | 29.6% | 53.1% | 0.695 | [[185, 61], [159, 67]] |
| xgboost | 24 | 472 | 87.7% | 47.5% | 46.9% | 46.1% | 49.7% | 60.6% | 43.6% | 33.2% | 46.6% | 0.711 | [[149, 97], [151, 75]] |
| causal_1d_cnn | 36 | 287 | 53.3% | 49.5% | 49.6% | 40.0% | 49.6% | 89.5% | 48.3% | 9.7% | 48.9% | 0.692 | [[128, 15], [130, 14]] |
| random_forest | 36 | 287 | 53.3% | 51.2% | 51.3% | 49.4% | 50.8% | 70.6% | 52.3% | 31.9% | 51.5% | 0.697 | [[101, 42], [98, 46]] |
| xgboost | 36 | 287 | 53.3% | 44.3% | 44.3% | 43.3% | 45.3% | 57.3% | 42.5% | 31.2% | 43.9% | 0.717 | [[82, 61], [99, 45]] |

CNN validation accuracy was 51.5%, 51.5%, 52.3% and 49.5% for lengths 6, 12, 24 and 36. L6/L12 predicted all DOWN; L24 predicted UP only 7 times among 472 directional cases; L36 had 53.3% sequence eligibility and only 9.7% UP recall. Matched Random Forest validation accuracy was higher at each length. All-candle operational accuracy, where neutral actuals count as wrong, was 23.9%, 23.9%, 23.8% and 21.9% for the CNN.

## Date stability

Daily statistics include dates with at least 10 directional labels; the full date-level details are in `date_stability_metrics.csv`.

| CNN L | Dates n≥10 | Median daily accuracy | Dates >50% | Dates >55% |
| --- | --- | --- | --- | --- |
| 6 | 45 | 54.8% | 60.0% | 46.7% |
| 12 | 45 | 54.5% | 60.0% | 46.7% |
| 24 | 43 | 54.2% | 53.5% | 48.8% |
| 36 | 31 | 55.6% | 61.3% | 51.6% |

Daily medians are not enough to establish an edge: minima and maxima vary widely, and fold-level performance remains unstable.

## Reproduction and safeguards

Run from the project root to reproduce the training and evaluation, then rebuild the narrative report from its saved outputs:

```powershell
.\.venv\Scripts\python.exe models\alpaca_sequence_research_v1\run_sequence_research.py
.\.venv\Scripts\python.exe models\alpaca_sequence_research_v1\build_research_report.py
```

The script allowlists only the pre-test raw cache and metadata, train/validation splits, established fold schedule, and saved v3 baseline prediction outputs. It rejects other input paths. It makes no network calls, does not read `test.csv` or post-September 24 data, and does not access or modify the live application or frozen production model.

Focused tests: **8 passed**. Syntax compilation passed. TensorFlow 2.22.0-rc0 was available; a small synthetic CNN trained, saved and reloaded with matching probabilities. Six Keras/NumPy deprecation warnings occurred in tests, and TensorFlow emitted a graph-attribute compatibility warning during fitting. The installed TensorFlow build is a release candidate, a reproducibility limitation. The full project suite was not run.

No model is selected or promoted. The existing frozen three-class model remains unchanged. No profitability or live performance is claimed.
