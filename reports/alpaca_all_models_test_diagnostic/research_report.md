# All-model ECL test-period diagnostic

## Scope and interpretation

This is the authorized one-time diagnostic comparison using existing saved models. It is **not independent future confirmation**: this test period and its class information were previously exposed in research. No model was retrained, tuned, refit, recalibrated, or selected from these results; no production artifact was changed. Results must not be described as proof of future performance or profitability.

Source: `data/alpaca_ml_pipeline_v1/test.csv`, ECL / Alpaca IEX, five-minute candles, 15-minute horizon, strict ±0.10% target, classes DOWN / NEUTRAL / UP. The evaluated file contained 1,369 rows spanning 2026-09-25 through 2026-11-12. Actual labels: DOWN 334, NEUTRAL 695, UP 340; directional population 674. Each three-class model was scored on all test rows. Binary/regression models were scored directionally only on the 674 actual UP/DOWN rows, and separately on all rows with actual NEUTRALs counted as incorrect. CNN models have shorter eligible sequence populations; their row counts are shown and matched-timestamp results are available in `sequence_matched_directional_metrics.csv`.

## Three-class ranking

Ranked by three-class accuracy (DOWN / NEUTRAL / UP metrics are classwise). Confusion matrices use actual rows and predicted columns in that class order and are saved individually under `confusion_matrices/`.

| Model | N | Accuracy | Balanced acc. | Macro F1 | D P | D R | D F1 | N P | N R | N F1 | U P | U R | U F1 | Predicted D/N/U |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_3class | 1369 | 42.07% | 37.79% | 37.69% | 29.26% | 32.93% | 30.99% | 56.02% | 54.24% | 55.12% | 27.81% | 26.18% | 26.97% | DOWN:376, NEUTRAL:673, UP:320 |
| random_forest_classifier_3class | 1369 | 42.07% | 36.81% | 36.62% | 25.64% | 20.96% | 23.06% | 55.95% | 56.83% | 56.39% | 28.46% | 32.65% | 30.41% | DOWN:273, NEUTRAL:706, UP:390 |
| xgboost_classifier_3class | 1369 | 38.57% | 36.23% | 35.56% | 26.10% | 23.05% | 24.48% | 56.09% | 45.04% | 49.96% | 26.74% | 40.59% | 32.24% | DOWN:295, NEUTRAL:558, UP:516 |
| frozen_balanced_lightgbm_B_reference | 1369 | 38.42% | 34.89% | 34.75% | 23.96% | 24.25% | 24.11% | 54.02% | 48.35% | 51.03% | 26.65% | 32.06% | 29.11% | DOWN:338, NEUTRAL:622, UP:409 |
| hist_gradient_boosting_classifier_3class | 1369 | 37.47% | 34.60% | 34.19% | 22.26% | 21.86% | 22.05% | 54.58% | 45.47% | 49.61% | 26.84% | 36.47% | 30.92% | DOWN:328, NEUTRAL:579, UP:462 |

Always-NEUTRAL accuracy was 50.77%; it exceeded every learned three-class model. Highest learned three-class accuracy was a tie: Logistic Regression and Random Forest, both 42.07%.

## Directional ranking

Ranked by accuracy on actual directional observations only. Binary/regression confusion matrices use actual rows and predicted columns in `[DOWN, UP]` order. Regression decisions use the pre-existing zero boundary. CNNs always emit a direction when a sequence exists; no NEUTRAL output was invented.

| Model | Directional N | Directional acc. | Balanced acc. | Macro F1 | DOWN P | DOWN R | UP P | UP R | Directional coverage | CM [D,U] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge_regression | 674 | 54.45% | 54.62% | 52.83% | 52.90% | 73.65% | 57.89% | 35.59% | 100.00% | [[246, 88], [219, 121]] |
| binary_xgboost_v3 | 674 | 51.34% | 51.34% | 51.34% | 50.88% | 51.80% | 51.80% | 50.88% | 100.00% | [[173, 161], [167, 173]] |
| cnn_6_candles | 674 | 49.55% | 50.00% | 33.13% | 49.55% | 100.00% | 0.00% | 0.00% | 100.00% | [[334, 0], [340, 0]] |
| cnn_12_candles | 674 | 49.55% | 50.00% | 33.13% | 49.55% | 100.00% | 0.00% | 0.00% | 100.00% | [[334, 0], [340, 0]] |
| random_forest_regression | 674 | 48.66% | 48.77% | 47.97% | 48.56% | 60.78% | 48.83% | 36.76% | 100.00% | [[203, 131], [215, 125]] |
| binary_random_forest_v3 | 674 | 48.22% | 48.28% | 48.03% | 48.03% | 54.79% | 48.46% | 41.76% | 100.00% | [[183, 151], [198, 142]] |
| cnn_24_candles | 619 | 47.98% | 48.21% | 33.22% | 48.84% | 95.45% | 17.65% | 0.96% | 91.84% | [[294, 14], [308, 3]] |
| xgboost_regression | 674 | 47.63% | 47.68% | 47.47% | 47.48% | 53.59% | 47.81% | 41.76% | 100.00% | [[179, 155], [198, 142]] |
| binary_xgboost_v1_additional | 674 | 47.33% | 47.23% | 46.60% | 45.98% | 35.93% | 48.18% | 58.53% | 100.00% | [[120, 214], [141, 199]] |
| hist_gradient_boosting_regression | 674 | 46.29% | 46.29% | 46.29% | 45.81% | 45.81% | 46.76% | 46.76% | 100.00% | [[153, 181], [181, 159]] |
| cnn_36_candles | 416 | 45.91% | 49.72% | 35.82% | 45.52% | 93.68% | 52.00% | 5.75% | 61.72% | [[178, 12], [213, 13]] |

The top directional result was Ridge Regression at 54.45% (DOWN recall 73.65%, UP recall 35.59%). Binary XGBoost v3 reached 51.34%. These do not meet a 55% directional-accuracy goal.

Regression error and return correlation (errors in percentage points, as saved by the original regression evaluation):

| Regressor | MAE | RMSE | Predicted/actual correlation |
| --- | --- | --- | --- |
| ridge_regression | 0.12624 | 0.16643 | +0.0378 |
| random_forest_regression | 0.12748 | 0.16793 | -0.0654 |
| xgboost_regression | 0.12808 | 0.16917 | -0.0827 |
| hist_gradient_boosting_regression | 0.13197 | 0.17282 | -0.0261 |


## All-candle accuracy

This separate comparison counts an actual NEUTRAL row as incorrect for models that always emit UP/DOWN.

| Model | Rows | All-candle accuracy |
| --- | --- | --- |
| logistic_regression_3class | 1369 | 42.07% |
| random_forest_classifier_3class | 1369 | 42.07% |
| xgboost_classifier_3class | 1369 | 38.57% |
| frozen_balanced_lightgbm_B_reference | 1369 | 38.42% |
| hist_gradient_boosting_classifier_3class | 1369 | 37.47% |
| ridge_regression | 1369 | 26.81% |
| binary_xgboost_v3 | 1369 | 25.27% |
| cnn_6_candles | 1369 | 24.40% |
| cnn_12_candles | 1369 | 24.40% |
| random_forest_regression | 1369 | 23.96% |
| binary_random_forest_v3 | 1369 | 23.74% |
| xgboost_regression | 1369 | 23.45% |
| binary_xgboost_v1_additional | 1369 | 23.30% |
| hist_gradient_boosting_regression | 1369 | 22.79% |
| cnn_24_candles | 1253 | 21.69% |
| cnn_36_candles | 861 | 13.95% |

Always-NEUTRAL scored 50.77%, the highest all-candle accuracy. Among the learned models, the best all-candle accuracy was 42.07%. The all-candle and directional rankings answer different questions and should not be conflated.

## Baselines

| Baseline | 3-class accuracy | Directional accuracy | All-candle accuracy | 3-class CM [D,N,U] |
| --- | --- | --- | --- | --- |
| always_neutral | 50.77% | 0.00% | 50.77% | [[0, 334, 0], [0, 695, 0], [0, 340, 0]] |
| training_majority_3class | 50.77% | 0.00% | 50.77% | [[0, 334, 0], [0, 695, 0], [0, 340, 0]] |
| training_majority_direction | 24.40% | 49.55% | 24.40% | [[334, 0, 0], [695, 0, 0], [340, 0, 0]] |
| always_up | 24.84% | 50.45% | 24.84% | [[0, 0, 334], [0, 0, 695], [0, 0, 340]] |
| always_down | 24.40% | 49.55% | 24.40% | [[334, 0, 0], [695, 0, 0], [340, 0, 0]] |

The training-majority three-class class was NEUTRAL; on this test population its accuracy matches always-NEUTRAL at 50.77%. The directional training-majority baseline always predicted DOWN and scored 49.55%; always-UP scored 50.45%.

## Previous walk-forward and validation comparison

These are comparisons to previously saved OOF/validation records only; validation was reused in earlier research and is not pristine. See `previous_performance_comparison.csv` for exact sources and values.

| Model | Prior period | Prior accuracy | Diagnostic accuracy | Difference | Availability | Saved source |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_3class | validation | 44.70% | 42.07% | -2.63 pp | AVAILABLE | reports/alpaca_baselines_v1/validation_comparison.csv |
| random_forest_classifier_3class | validation | 44.70% | 42.07% | -2.63 pp | AVAILABLE | reports/alpaca_baselines_v1/validation_comparison.csv |
| xgboost_classifier_3class | validation | 40.31% | 38.57% | -1.74 pp | AVAILABLE | reports/alpaca_baselines_v1/validation_comparison.csv |
| hist_gradient_boosting_classifier_3class | validation | 38.85% | 37.47% | -1.37 pp | AVAILABLE | reports/alpaca_baselines_v1/validation_comparison.csv |
| binary_random_forest_v3 | validation | 53.53% | 48.22% | -5.31 pp | AVAILABLE | reports/alpaca_binary_research_v3/validation_model_comparison.csv |
| binary_xgboost_v3 | validation | 47.21% | 51.34% | +4.12 pp | AVAILABLE | reports/alpaca_binary_research_v3/validation_model_comparison.csv |
| binary_xgboost_v1_additional | validation | 50.37% | 47.33% | -3.04 pp | AVAILABLE | models/alpaca_binary_research_v1/metrics.csv |
| ridge_regression | validation | 48.88% | 54.45% | +5.57 pp | AVAILABLE | reports/alpaca_return_regression_v1/validation_metrics.csv |
| random_forest_regression | validation | 52.60% | 48.66% | -3.94 pp | AVAILABLE | reports/alpaca_return_regression_v1/validation_metrics.csv |
| xgboost_regression | validation | 51.86% | 47.63% | -4.23 pp | AVAILABLE | reports/alpaca_return_regression_v1/validation_metrics.csv |
| hist_gradient_boosting_regression | validation | 55.39% | 46.29% | -9.10 pp | AVAILABLE | reports/alpaca_return_regression_v1/validation_metrics.csv |
| frozen_balanced_lightgbm_B_reference | validation | 41.17% | 38.42% | -2.75 pp | AVAILABLE | models/alpaca_final_freeze_v1/audit_report.md |
| cnn_6_candles | validation | 51.49% | 49.55% | -1.93 pp | AVAILABLE | reports/alpaca_sequence_research_v1/validation_metrics.csv |
| cnn_12_candles | validation | 51.49% | 49.55% | -1.93 pp | AVAILABLE | reports/alpaca_sequence_research_v1/validation_metrics.csv |
| cnn_24_candles | validation | 52.33% | 47.98% | -4.35 pp | AVAILABLE | reports/alpaca_sequence_research_v1/validation_metrics.csv |
| cnn_36_candles | validation | 49.48% | 45.91% | -3.56 pp | AVAILABLE | reports/alpaca_sequence_research_v1/validation_metrics.csv |
| binary_random_forest_v3 | walk_forward_OOF | 51.91% | 48.22% | -3.69 pp | AVAILABLE | reports/alpaca_binary_research_v3/walk_forward_config_comparison.csv |
| binary_xgboost_v3 | walk_forward_OOF | 53.32% | 51.34% | -1.99 pp | AVAILABLE | reports/alpaca_binary_research_v3/walk_forward_config_comparison.csv |
| ridge_regression | walk_forward_OOF | 54.04% | 54.45% | +0.41 pp | AVAILABLE | reports/alpaca_return_regression_v1/walk_forward_overall_metrics.csv |
| random_forest_regression | walk_forward_OOF | 50.93% | 48.66% | -2.26 pp | AVAILABLE | reports/alpaca_return_regression_v1/walk_forward_overall_metrics.csv |
| xgboost_regression | walk_forward_OOF | 51.35% | 47.63% | -3.72 pp | AVAILABLE | reports/alpaca_return_regression_v1/walk_forward_overall_metrics.csv |
| hist_gradient_boosting_regression | walk_forward_OOF | 51.52% | 46.29% | -5.22 pp | AVAILABLE | reports/alpaca_return_regression_v1/walk_forward_overall_metrics.csv |
| cnn_6_candles | walk_forward_OOF | 52.95% | 49.55% | -3.39 pp | AVAILABLE | reports/alpaca_sequence_research_v1/walk_forward_overall_metrics.csv |
| cnn_12_candles | walk_forward_OOF | 53.45% | 49.55% | -3.90 pp | AVAILABLE | reports/alpaca_sequence_research_v1/walk_forward_overall_metrics.csv |
| cnn_24_candles | walk_forward_OOF | 52.60% | 47.98% | -4.62 pp | AVAILABLE | reports/alpaca_sequence_research_v1/walk_forward_overall_metrics.csv |
| cnn_36_candles | walk_forward_OOF | 51.82% | 45.91% | -5.91 pp | AVAILABLE | reports/alpaca_sequence_research_v1/walk_forward_overall_metrics.csv |

Overall, the diagnostic test accuracy was lower than validation for most models. Ridge was a notable exception relative to its prior validation and OOF summaries, but this exposed period cannot establish durable superiority. CNN lengths 6/12 again collapsed to all-DOWN predictions; length 24 emitted very few UP calls, and length 36 had reduced sequence eligibility. The matched-timestamp table controls the population for these CNN comparisons.

## Integrity, artifacts, and tests

- All 14 model slots explicitly named in the request evaluated successfully, including the saved binary RF/XGBoost artifacts. The frozen Balanced LightGBM B was added as the production reference, and an older separate binary XGBoost artifact was also evaluated. The frozen candidate and identical candidate copy have the same hash and were evaluated once.
- Additional saved artifacts are listed in `model_inventory.csv` with explicit status and reasons. Different-horizon/one-minute models were incompatible with this task; selective policy and unrequested candidate/ensemble artifacts were not part of the predeclared comparison.
- Artifact hash files contain 20 entries; before/after maps are **identical**. Frozen hash is unchanged.
- Frozen candidate SHA256: `31188a09eba09d58d441324fcfb130a23268a833a459709e4c0c6ad9ca45d08d`. Full before/after SHA256 maps are in `artifact_hashes_before.json` and `artifact_hashes_after.json`.
- Every prediction and confusion matrix is preserved under `predictions/` and `confusion_matrices/`. `per_model_metrics.csv`, `prediction_distribution.csv`, ranked tables, baseline metrics, matched-sequence metrics, and previous-performance comparisons provide machine-readable results.
- Focused tests: 4 passed. Python syntax compilation passed for evaluator, summary helper, and focused test module. Summary generation emitted scikit-learn warnings for directional scoring when a model predicts a class absent from the directional truth subset; this is expected and does not invalidate the scores (zero-division behavior is explicit).
- No production/live application, frozen model, research artifact, or dataset was modified. Evaluation inputs were read-only; no inference was rerun while producing this report.

## Conclusion

No evaluated model exceeded 50% overall three-class accuracy; all were below the 50.77% always-NEUTRAL baseline. Ridge had the best directional accuracy at 54.45%, narrowly below 55%, but its recall was asymmetric (DOWN 73.65%, UP 35.59%). Binary XGBoost v3 was more balanced at 51.34% (DOWN recall 51.80%, UP recall 50.88%). None establishes a reliable, balanced improvement, and no candidate should be promoted automatically. Because this test period has prior exposure, none of these numbers is genuine independent future evidence.
