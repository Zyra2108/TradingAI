# ECL Alpaca IEX Binary UP/DOWN Research v3

## Decision

No candidate satisfies the predeclared validation guardrails. **Do not promote or integrate a binary model.** The best validation directional accuracy was Random Forest at 53.5%, below the 55% objective; it also had only 29.5% UP recall. Its result does not establish balanced directional usefulness. Validation was used repeatedly in earlier project research, so this is limited confirmation rather than an independent holdout.

This isolated experiment uses only the pre-test Alpaca IEX training and validation splits ending 2026-09-24. NEUTRAL examples are excluded from binary fitting and directional metrics. The binary classifier always emits UP or DOWN; a low-confidence flag is informational and does not abstain. No protected test data or post-2026-09-24 data were accessed.

The requested v1 output directory already contained prior research. To preserve it, this experiment and its outputs are isolated in `models/alpaca_binary_research_v3/` and `reports/alpaca_binary_research_v3/`.

## Data and target

| Split | All rows | Directional rows | Neutral excluded | DOWN | UP | Directional rows / all rows |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4,883 | 2,519 | 2,364 | 1,323 | 1,196 | 51.59% |
| Validation | 1,161 | 538 | 623 | 277 | 261 | 46.34% |

Train dates: 2025-11-13–2026-08-05. Validation dates: 2026-08-06–2026-09-24. The target preserves the existing 15-minute forward return: UP strictly above +0.10%, DOWN strictly below −0.10%, otherwise NEUTRAL. The 24 frozen B feature names/order are retained. Audit found no non-finite feature values, duplicate decision timestamps, or invalid target horizons in these splits. The detailed audit is `models/alpaca_binary_research_v3/dataset_label_audit.json`.

## Method

Four expanding chronological folds were evaluated within training, grouped by whole trading dates. Each fold trains only on earlier dates and purges training examples whose future endpoint reaches the next fold's first decision timestamp. Preprocessing is fit within each training fold. Four modest model families were tested with two predeclared configurations each: Logistic Regression, Random Forest, XGBoost and HistGradientBoosting. XGBoost 3.4.1 was installed and used; no package was installed. Random seed: 42.

Per-family configuration selection used training OOF metrics only. A training-OOF 25th-percentile confidence threshold creates a low-confidence warning; it does not change UP/DOWN output. After the design was fixed, each selected family configuration was evaluated once on the existing validation split. The validation period is not pristine and was not used for further tuning.

## Walk-forward results

The following are the selected configurations per family. Confusion matrices use actual rows and predicted columns in `[DOWN, UP]` order. The full fold table, including classwise precision/recall and probability-derived log loss, is `reports/alpaca_binary_research_v3/walk_forward_fold_metrics.csv`.

| Family / selected config | Fold dates | n | Actual D/U | Predicted D/U | Accuracy | Balanced acc. | Macro F1 | Log loss | Confusion matrix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Logistic / `lr_c1_balanced` | Apr 30–May 21 | 361 | 200/161 | 200/161 | 52.9% | 52.4% | 52.4% | 0.695 | `[[115,85],[85,76]]` |
| Logistic / `lr_c1_balanced` | May 22–Jun 17 | 335 | 170/165 | 168/167 | 51.0% | 51.0% | 51.0% | 0.707 | `[[87,83],[81,84]]` |
| Logistic / `lr_c1_balanced` | Jun 18–Jul 14 | 190 | 119/71 | 93/97 | 55.8% | 56.5% | 55.2% | 0.695 | `[[64,55],[29,42]]` |
| Logistic / `lr_c1_balanced` | Jul 15–Aug 05 | 302 | 146/156 | 147/155 | 52.6% | 52.6% | 52.6% | 0.698 | `[[75,71],[72,84]]` |
| Random Forest / `rf_compact_unweighted` | Apr 30–May 21 | 361 | 200/161 | 216/145 | 55.1% | 54.1% | 54.1% | 0.687 | `[[127,73],[89,72]]` |
| Random Forest / `rf_compact_unweighted` | May 22–Jun 17 | 335 | 170/165 | 206/129 | 46.3% | 46.1% | 45.5% | 0.713 | `[[98,72],[108,57]]` |
| Random Forest / `rf_compact_unweighted` | Jun 18–Jul 14 | 190 | 119/71 | 133/57 | 57.9% | 53.0% | 52.9% | 0.671 | `[[86,33],[47,24]]` |
| Random Forest / `rf_compact_unweighted` | Jul 15–Aug 05 | 302 | 146/156 | 232/70 | 48.3% | 49.2% | 44.9% | 0.713 | `[[111,35],[121,35]]` |
| XGBoost / `xgb_shallow_unweighted` | Apr 30–May 21 | 361 | 200/161 | 135/226 | 57.1% | 58.5% | 56.8% | 0.699 | `[[90,110],[45,116]]` |
| XGBoost / `xgb_shallow_unweighted` | May 22–Jun 17 | 335 | 170/165 | 195/140 | 47.2% | 47.0% | 46.7% | 0.721 | `[[94,76],[101,64]]` |
| XGBoost / `xgb_shallow_unweighted` | Jun 18–Jul 14 | 190 | 119/71 | 135/55 | 61.1% | 56.1% | 56.1% | 0.673 | `[[90,29],[45,26]]` |
| XGBoost / `xgb_shallow_unweighted` | Jul 15–Aug 05 | 302 | 146/156 | 209/93 | 48.0% | 48.6% | 46.4% | 0.717 | `[[99,47],[110,46]]` |
| HistGradientBoosting / `hgb_regular_weighted` | Apr 30–May 21 | 361 | 200/161 | 144/217 | 51.2% | 52.4% | 51.1% | 0.773 | `[[84,116],[60,101]]` |
| HistGradientBoosting / `hgb_regular_weighted` | May 22–Jun 17 | 335 | 170/165 | 163/172 | 50.1% | 50.2% | 50.1% | 0.776 | `[[83,87],[80,85]]` |
| HistGradientBoosting / `hgb_regular_weighted` | Jun 18–Jul 14 | 190 | 119/71 | 127/63 | 60.0% | 56.1% | 56.2% | 0.688 | `[[85,34],[42,29]]` |
| HistGradientBoosting / `hgb_regular_weighted` | Jul 15–Aug 05 | 302 | 146/156 | 189/113 | 49.3% | 49.8% | 48.7% | 0.761 | `[[91,55],[98,58]]` |

Only isolated folds exceed 55%; no family is consistently above 55%. XGBoost and HistGradientBoosting peak in the third fold, then fall below 50% in the fourth. This period sensitivity argues against treating the best fold as a reliable edge. Fold-level baselines are in `walk_forward_baselines.csv`; the always-majority directional baseline varies with the fold's class balance and has 50% balanced accuracy by construction.

Across selected configurations, mean OOF accuracy was 53.1% for Logistic Regression, 51.9% for Random Forest, 53.3% for XGBoost and 52.7% for HistGradientBoosting. Mean OOF balanced accuracy was 53.1%, 50.6%, 52.6% and 52.1%, respectively. RSI baseline fold accuracy was 45.7%, 47.5%, 48.4% and 51.3%; it did not provide a stable high-accuracy baseline either. See `walk_forward_config_comparison.csv` and `walk_forward_baselines.csv` for all configurations and fold details.

## One-time validation confirmation

| Family | Directional accuracy | Balanced acc. | Macro F1 | UP precision / recall / F1 | DOWN precision / recall / F1 | Eligible directional rows | Directional rows / all rows | All-candle accuracy* | Log loss | Low-confidence share |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | 47.6% | 47.3% | 47.0% | 45.2 / 37.9 / 41.3% | 49.2 / 56.7 / 52.7% | 538 | 46.34% | 22.1% | 0.706 | 28.6% |
| Random Forest | 53.5% | 52.8% | 50.5% | 53.8 / 29.5 / 38.1% | 53.4 / 76.2 / 62.8% | 538 | 46.34% | 24.8% | 0.696 | 20.6% |
| XGBoost | 47.2% | 46.8% | 45.9% | 44.0 / 32.6 / 37.4% | 49.0 / 61.0 / 54.3% | 538 | 46.34% | 21.9% | 0.712 | 32.4% |
| HistGradientBoosting | 51.3% | 51.2% | 51.2% | 49.8 / 48.3 / 49.0% | 52.6 / 54.2 / 53.4% | 538 | 46.34% | 23.8% | 0.714 | 28.6% |

All 1,161 validation candles receive a binary UP/DOWN model output (no abstention). The 538 eligible directional actuals are the denominator for directional metrics; 623 neutral actuals are excluded from those binary metrics. **All-candle accuracy*** counts every neutral actual as incorrect, since the binary model cannot emit NEUTRAL. It is included to make the cost of excluding neutral labels visible, not as a comparable three-class metric.

On directional validation rows, always-UP scored 48.5% accuracy and always-DOWN/training-majority scored 51.5%, each with 50.0% balanced accuracy and zero recall for one class. The fixed RSI-50 rule scored 50.4% accuracy, 50.3% balanced accuracy and 50.2% macro-F1. Random Forest's 53.5% is the highest candidate accuracy but falls short of the 55% target and has weak UP recall. No candidate met the predeclared combined guardrails: accuracy ≥55%, balanced accuracy ≥50%, and recall ≥40% for both UP and DOWN. No selected binary model artifact was produced.

Validation confusion matrices (`[DOWN,UP]` actual rows/predicted columns): Logistic `[[157,120],[162,99]]`; RF `[[211,66],[184,77]]`; XGBoost `[[169,108],[176,85]]`; HGB `[[150,127],[135,126]]`. See the CSV for all four baseline matrices too.

## Confidence analysis

Each model's warning boundary is the 25th percentile of its training-OOF maximum class probability. It flags low confidence only; it does not suppress the directional output. Validation confidence quartiles are descriptive and were not used to tune a policy. Accuracy varies irregularly across quartiles; higher confidence does not consistently separate correct predictions. For example, HGB's top validation confidence quartile has 59.6% directional accuracy on 94 directional rows, but its total validation accuracy is only 51.3% and the quartile result is not independent evidence. Full per-model quartile counts, accuracy, UP/DOWN recall and OOF cut points are in `validation_confidence_groups.csv`; confidence distributions are in `confidence_distribution.csv`.

## Interpretation and recommendation

Binary filtering changes the problem: 46.3% of validation candles are directional under the unchanged threshold, and the binary model is forced to choose a direction even when the actual class is NEUTRAL. The eligible directional classes themselves are nearly balanced (277 DOWN / 261 UP), so a majority-class shortcut explains only about 51.5% accuracy. Results near that level and unstable fold performance indicate weak directional separation in these features/labels rather than a useful, robust binary edge. Do not integrate this model. Keep the existing frozen three-class model operational. Further binary work should wait for a predeclared methodological change and a later, untouched chronological evaluation period; do not tune again on this reused validation set.

This is classification research, not evidence of trading profitability. No costs or trading simulation are claimed here.

## Reproduction and artifacts

From project root, run:

```powershell
.\.venv\Scripts\python.exe models\alpaca_binary_research_v3\run_binary_research.py
```

The script uses only `data/alpaca_ml_pipeline_v1/train.csv`, `validation.csv`, and the training manifest. It never opens a test split. Configuration, feature schema, dataset audit, saved family models and reproducibility hashes are under `models/alpaca_binary_research_v3/`; metrics and prediction CSVs are under `reports/alpaca_binary_research_v3/`. The selected model is `null` because none passed the predeclared validation guardrails.
