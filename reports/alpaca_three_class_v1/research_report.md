# ECL Three-Class ML Research v1

## Decision

**No model demonstrates a repeatable, useful improvement over the existing baselines. Do not promote a candidate.** XGBoost Multiclass was selected from training OOF results for the single limited validation confirmation. Its validation accuracy was 40.83%, far below the validation always-NEUTRAL baseline (53.66%). It was only +0.52 percentage points relative to the stored existing three-class XGBoost baseline on validation, with lower directional macro-F1 (27.11% versus 27.52%). This does not establish meaningful directional improvement.

This validation period has been repeatedly used in prior work and is **not a pristine holdout**. The experiment read allowlist excluded the protected test CSV and saved test predictions; no test rows were loaded or evaluated. No live or frozen model was modified.

## Data and integrity

- Alpaca IEX ECL, 5-minute bars; 15-minute future return and original strict ±0.10% thresholds.
- Training: 4,883 rows, 2025-11-13–2026-08-05; validation: 1,161 rows, 2026-08-06–2026-09-24.
- Train labels: {"DOWN": 1323, "NEUTRAL": 2364, "UP": 1196}; validation labels: {"DOWN": 277, "NEUTRAL": 623, "UP": 261}.
- Exact ordered 24-feature contract is taken from the frozen specification. The split CSV has two `close` columns by role (label reference close and model feature close); they were verified equal, renamed in memory to distinct roles, and only the feature close entered the model. Volume and metadata were excluded.
- Split inputs match the frozen specification hashes. Decision times are unique and chronological; labels match saved `future_return`; endpoints equal exactly +15 minutes and remain on the same New York session date. Each fold uses only earlier training rows and purges endpoints at/after the first OOF decision timestamp.
- Four chronological expanding folds:

| Fold | Training through | OOF dates |
| --- | --- | --- |
| 1 | 2026-04-29 | 2026-04-30–2026-05-21 |
| 2 | 2026-05-21 | 2026-05-22–2026-06-17 |
| 3 | 2026-06-17 | 2026-06-18–2026-07-14 |
| 4 | 2026-07-14 | 2026-07-15–2026-08-05 |


The existing 24 feature calculations were reused from the already-created Alpaca dataset; this experiment did not add indicators or recalculate features. Dataset provenance identifies Alpaca IEX. Feature names contain no target, future timestamp, or volume input.

## Packages and fixed configurations

- scikit-learn 1.9.1, XGBoost 3.4.1; XGBoost 3.4.1 was present and used with `multi:softprob`.
- CatBoost was not installed. It was not installed or silently substituted; therefore CatBoost was not run.
- Ridge alpha=10 with fold-local median imputation and scaling. XGBoost and HistGradientBoosting use fold-local median imputation and training-fold balanced sample weights. Configurations/seeds are in `experiment_config.json`; no broad search was run.

## Training-only walk-forward results

OOF rows are 2,248 pooled held-out training-period predictions across four folds. Scores below use the selected Ridge boundary where applicable. The Ridge threshold sweep used only these OOF predictions, with macro-F1 selection subject to directional coverage, neutral-rate and minimum-per-class guardrails. **No threshold in the predeclared positive threshold grid passed that guardrail.** The diagnostic unconstrained macro-F1 choice was 0.025 percentage points, but Ridge failed the guardrail and was not eligible as the candidate. The separate zero-boundary diagnostic is binary only and is not the Ridge three-class neutral boundary.

| Candidate | OOF accuracy | Balanced acc. | Macro F1 | Weighted F1 | Mean fold accuracy ± SD | Mean fold macro-F1 ± SD | Mean fold balanced acc. ± SD | Directional macro P | Directional macro R | Directional macro F1 | Directional output coverage | Guardrail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge_regression_to_3class | 43.91% | 35.73% | 33.80% | 39.26% | 44.12% ± 1.54 pp | 33.73% ± 1.13 pp | 35.68% ± 1.08 pp | 29.85% | 16.33% | 20.91% | 29.18% | False |
| xgboost_multiclass | 38.52% | 38.04% | 37.51% | 39.46% | 39.49% ± 5.51 pp | 36.70% ± 3.89 pp | 38.36% ± 1.44 pp | 29.46% | 36.78% | 32.71% | 65.93% | True |
| hist_gradient_boosting_multiclass | 38.39% | 36.14% | 36.10% | 38.88% | 38.79% ± 3.57 pp | 35.49% ± 2.46 pp | 36.16% ± 2.05 pp | 28.23% | 30.86% | 29.47% | 57.92% | True |


Detailed fold-level training/OOF accuracy, balanced accuracy, macro/weighted F1, class P/R/F1 and confusion matrices are in `walk_forward_fold_table.csv` and `walk_forward_confusion_matrices.csv`. The OOF and in-sample training scores show substantial overfit for the tree models: XGBoost mean in-sample fold accuracy was about 66.0%, versus mean OOF fold accuracy 39.5%; HistGradientBoosting showed the same pattern. Ridge did not overfit comparably, but produced too many neutral calls and weak UP/DOWN recognition under its three-class policy.

The new XGBoost candidate was the highest OOF mean-fold macro-F1 among guardrail-valid candidates (36.70%). Fold accuracy varied materially; it was particularly weak in fold 1 and did not exceed 50% in any fold. Its pooled OOF accuracy was below the training-period always-NEUTRAL accuracy (48.41%).

## Ridge zero-boundary diagnostic and saved controls

The new Ridge regressor's zero-sign diagnostic, restricted to actual UP/DOWN OOF examples, scored 54.04%; all-candle accuracy (actual neutral counted incorrect) was 28.56%. It is binary only and makes no neutral class prediction. The already-saved Ridge and binary XGBoost control outputs, plus the prior three-class XGBoost validation predictions, are in `saved_baseline_comparison.csv`; no comparator model was refit or reinferred.

Among saved controls, existing binary Ridge OOF directional accuracy was 54.04% pooled (fold mean differs due unequal fold sizes) and validation was 48.88%; existing binary XGBoost v3 was 52.61% pooled OOF and 47.21% validation. These binary models remain directional controls, not three-class competitors. The previous three-class XGBoost baseline scored 40.31% validation accuracy and 36.20% macro-F1.

## One-time limited validation confirmation

After selection and freezing the candidate config/hash, only the selected XGBoost Multiclass artifact was evaluated on validation. Class order is DOWN, NEUTRAL, UP.

- N=1,161; overall accuracy 40.83%; balanced accuracy 36.73%; macro-F1 36.33%; weighted-F1 41.91%.
- Directional macro precision/recall/F1: 25.06% / 29.89% / 27.11%.
- Directional output coverage over all rows 54.87%; among actual directional rows 60.97%; accuracy among emitted directional predictions 25.12%.
- Prediction distribution: `{"DOWN": 281, "NEUTRAL": 524, "UP": 356}`.
- Log loss: 1.0876.

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| DOWN | 24.56% | 24.91% | 24.73% | 277 |
| NEUTRAL | 59.92% | 50.40% | 54.75% | 623 |
| UP | 25.56% | 34.87% | 29.50% | 261 |


Validation confusion matrix, actual rows / predicted columns `[DOWN, NEUTRAL, UP]`:

```text
[[69, 115, 93], [137, 314, 172], [75, 95, 91]]
```

Validation always-NEUTRAL accuracy was 53.66%. The selected model predicted 524 NEUTRAL rows of 1161, so its weak accuracy is not caused by excessive NEUTRAL output; the model instead makes many incorrect directional calls. The result is near the old three-class XGBoost baseline, not a meaningful gain.

Existing binary controls on their saved validation predictions (actual directional rows only): Ridge zero-sign 48.88%; binary XGBoost v3 47.21%. Validation is reused and only a limited confirmation.

## Integrity, artifacts and reproduction

- Frozen selected research candidate: `xgboost_multiclass`; train-only artifact SHA256 `08aea3085131c3e20625a7b9c56fd161741f58408c02a7d30e807bbc4bea228b`. This is a research artifact, not the production model.
- Existing protected model artifacts hashed before/after are unchanged (`reproducibility_manifest.json`). Train/validation input hashes and all new output hashes are recorded there.
- No test CSV/predictions/metrics were included in the read allowlist. This experiment code contains no path to the protected test split.
- CatBoost unavailable; no installation attempted.
- Focused tests: 7 passed. Python syntax compilation passed. The first execution completed all folds and one selected validation inference; its final saved-baseline join failed due string/datetime mismatch. The timestamps were parsed and the remaining reports finalized from the already saved outputs; no model fit or validation inference was repeated.

Reproduce the experiment from a clean new output directory with:

```powershell
.\.venv\Scripts\python.exe models\alpaca_three_class_v1\run_experiment.py
```

The runner refuses to overwrite existing outputs. Outputs are isolated under `models/alpaca_three_class_v1/` and `reports/alpaca_three_class_v1/`.

## Conclusion

Ridge regression converted to three classes failed its OOF directional-coverage/class-distribution guardrail. XGBoost Multiclass and HistGradientBoosting passed the distribution guardrail but had low directional macro precision/recall and under-40% pooled OOF accuracy. XGBoost was the strongest by predeclared OOF macro-F1 tie hierarchy, but its validation accuracy (40.83%) remained well below always-NEUTRAL (53.66%), and its directional macro-F1 did not improve on the existing three-class XGBoost baseline. **No candidate demonstrated repeatable useful improvement; do not promote or integrate any candidate.** A future genuinely untouched evaluation would be needed before making out-of-sample claims.
