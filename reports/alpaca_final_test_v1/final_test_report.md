# Alpaca ECL final held-out test evaluation

**One-time evaluation:** this report records the first and only evaluation of the held-out Alpaca test split. No tuning, feature changes, threshold changes, retraining, or model modifications followed the results.

Candidate: `lightgbm_balanced_B_start`. Rows: 1,314. Dates: 2026-09-25 through 2026-11-11.
Actual class counts (DOWN / NEUTRAL / UP): 319 / 660 / 335.

## Classification metrics

| Metric | Frozen candidate | Training-majority dummy | Validation candidate | Test minus validation |
|---|---:|---:|---:|---:|
| Accuracy | 0.3813 | 0.5023 | 0.4117 | -0.0304 |
| Balanced accuracy | 0.3497 | 0.3333 | 0.3780 | -0.0283 |
| 3-class macro F1 | 0.3484 | 0.2229 | 0.3741 | -0.0257 |
| Directional precision, macro UP/DOWN | 0.2583 | 0.0000 | 0.2825 | -0.0242 |
| Directional recall, macro UP/DOWN | 0.2882 | 0.0000 | 0.3222 | -0.0341 |
| Directional F1, macro UP/DOWN | 0.2722 | 0.0000 | 0.2991 | -0.0269 |
| Directional prediction coverage | 0.5540 | 0.0000 | 0.5340 | +0.0200 |
| Log loss | 1.1758 | 17.9395 | 1.1380 | +0.0378 |

## Per-class report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| DOWN | 0.2411 | 0.2539 | 0.2473 | 319 |
| NEUTRAL | 0.5324 | 0.4727 | 0.5008 | 660 |
| UP | 0.2755 | 0.3224 | 0.2971 | 335 |

## Confusion matrix

Rows are actual classes; columns are predicted classes (DOWN, NEUTRAL, UP).

| Actual \ Predicted | DOWN | NEUTRAL | UP |
|---|---:|---:|---:|
| DOWN | 81 | 137 | 101 |
| NEUTRAL | 165 | 312 | 183 |
| UP | 90 | 137 | 108 |

## Prediction checks and frozen hashes

Predictions: 1,314; UP 392, DOWN 336, NEUTRAL 586.
All predictions belong to the configured class order: True. All probabilities are finite and nonnegative: True. Maximum row-sum error: 2.22e-16.

SHA-256 before and after evaluation (unchanged):

- `candidate_pipeline.joblib`: `31188a09eba09d58d441324fcfb130a23268a833a459709e4c0c6ad9ca45d08d`
- `lightgbm_estimator.joblib`: `a46470a26360002b213cb764d228ee18b0040d7842956f235ec42cda284ea4cf`
- `preprocessing_pipeline.joblib`: `0b5bb1609decec0712846762fd55d8becca956e541091a967041b3070f597af6`

## Interpretation and limitations

The dummy baseline predicts `NEUTRAL` for every test row, with its majority class determined from the training split distribution, not from test labels. The test period has only 33 trading dates, so results are a limited historical holdout measurement.

Classification accuracy and class metrics do not measure trading profitability. This evaluation includes no backtest, transaction costs, paper/live trading, or production-readiness assessment. The test set was not used to select or alter the model.

**Disclosure:** Before the evaluation, an earlier filesystem path-existence check queried metadata for `test.csv`; it did not read test contents. This report is the only test-content read and only model evaluation. No model tuning or changes followed it.

**Execution note:** Two initial script launches stopped at report-directory setup checks before reading `test.csv`; after correcting the directory handling, the successful run read the test CSV once and made one frozen-model probability prediction pass. No evaluation was produced by the failed launches.
