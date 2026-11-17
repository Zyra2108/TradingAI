# Alpaca IEX ECL baseline comparison

Validation was used once for model comparison and selection. The validation period has previously appeared in project research, so it is not a pristine holdout.
The new final test CSV was not opened during model fitting or model selection.

Selected by the predeclared validation rule: **xgboost**.

| Model | Accuracy | Balanced accuracy | Macro F1 | UP precision / recall | DOWN precision / recall | Directional macro precision | Directional macro F1 | Directional coverage | Directional-signal accuracy | Log loss | Fit sec | Predict val sec |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.4470 | 0.3620 | 0.3600 | 0.2362 / 0.1801 | 0.2526 / 0.2671 | 0.2444 | 0.2320 | 0.4238 | 0.2459 | 1.0734 | 0.1732 | 0.0028 |
| random_forest | 0.4470 | 0.3623 | 0.3615 | 0.2317 / 0.2299 | 0.2970 / 0.2166 | 0.2643 | 0.2406 | 0.3971 | 0.2603 | 1.0543 | 1.7503 | 0.1107 |
| xgboost | 0.4031 | 0.3658 | 0.3620 | 0.2424 / 0.3372 | 0.2660 / 0.2708 | 0.2542 | 0.2752 | 0.5556 | 0.2527 | 1.0943 | 0.9063 | 0.0067 |
| hist_gradient_boosting | 0.3885 | 0.3585 | 0.3538 | 0.2355 / 0.3257 | 0.2664 / 0.2924 | 0.2510 | 0.2761 | 0.5728 | 0.2496 | 1.1118 | 3.4840 | 0.0178 |
| always_neutral | 0.5366 | 0.3333 | 0.2328 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | — | — |
| fixed_rsi_rule | 0.4488 | 0.3397 | 0.3312 | 0.2435 / 0.1801 | 0.2422 / 0.1408 | 0.2429 | 0.1926 | 0.3049 | 0.2429 | — | — | — |
