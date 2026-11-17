# Alpaca IEX ECL ML rebuild and baseline comparison

## Scope and protection

All new data/model/report outputs are isolated under `data/alpaca_ml_pipeline_v1/`, `models/alpaca_baselines_v1/` and `reports/alpaca_baselines_v1/`. The previous train/validation/test CSVs were not used as model inputs; the protected old test CSV was not opened, hashed, or modified. The existing frozen live model and Alpaca application were not changed. The old Twelve Data artifacts were not used. No live connection or order was made.

Important limitation: the newly generated chronological test dates overlap dates previously used in the project’s earlier test research. During the first dataset QA invocation, class counts for the new split were printed from the in-memory build before the output was changed to withhold that summary. Those counts were not used for model fitting, feature selection, threshold selection, or model selection. The new test was still held out from candidate training and validation-based selection, but it cannot be called a completely blind or pristine project-level holdout. The one-time evaluation below is disclosed accordingly.

## Canonical source and reconciliation

| File | Role | Rows | SHA-256 | Evidence |
|---|---|---:|---|---|
| `reports/alpaca_iex_1y_audit_v1/alpaca_iex_ecl_5min_1y_20261112.csv` | Primary one-year Alpaca IEX 5-minute file | 18,735 | `757b35e5426426ed5840282b9817f888664789ffe5f50f17898f8178ed1bd4c2` | ECL / IEX / 5Min metadata and source reconciliation |
| `models/alpaca_1min_research_v1/raw_ecl_iex_5min_train_validation.csv` | Independent native Alpaca IEX cache; verification only | 16,060 | `b6a4d4bcf3562d2d8272a202010090300319c03aaca80280976a3b4ba71c357d` | ECL / IEX / 5Min metadata and source reconciliation |
| `data/alpaca_backfill_v1/ecl_iex_5min_20260925_20261112.csv` | Alpaca IEX 5-minute backfill | 2,682 | `eba280d6eeaac17ebc29d078d23d1afcd039e551579fc0788b4ffa542d60b4f9` | ECL / IEX / 5Min metadata and source reconciliation |

One-year source timestamps and New York timestamp column agree for all 18,735 rows. Across its 16,060 shared timestamps with the independently fetched native cache, OHLC, volume and trade count matched exactly. On the one-year/backfill overlap, all 2,627 rows matched exactly; backfill contributed 55 timestamps absent from the one-year file. No unexplained conflicts were found.

The one-year CSV’s original fetch script is not present. Its existing provenance manifest identifies it as an Alpaca IEX historical audit file; its shared interval is corroborated exactly against the independent cache and the later overlap against the documented backfill. The earliest partial session (Nov 12, 2025) has 48 source bars and is not independently covered by the native cache; those rows retain the upstream one-year provenance declaration.

## Canonical data and quality

- Canonical rows: **18,789**, from 2025-11-12 11:25:00 through 2026-11-12 15:55:00 New York time.
- Unique in-session canonical timestamps: **18,789**; duplicate timestamps: **0**.
- Canonical rows rejected: **1**. Exact row: **2026-09-18 09:25 ET** (2026-09-18 13:25 UTC), before the 09:30 NYSE open. The same invalid row occurs in the one-year and verification-cache files; one canonical observation was omitted.
- Invalid OHLC among accepted candles: **0**; zero-volume candles: **0**.
- Expected regular-session slots: **19,584**; absent IEX slots preserved as gaps: **795**. No missing candle or price was synthesized.
- Feature-ready and exact-target-ready examples before splits: **7,413**. Feature warm-up/nonfinite exclusions: 10,506; exact endpoint/gap exclusions: 2,438; those exclusion categories can overlap.

## Features, labels and splits

Features: exact frozen **24-feature** order from `models/alpaca_final_freeze_v1/frozen_candidate_spec.json`; existing `live_app/features.py::calculate_b_features` and historical functions were reused. Volume is not a model input. Labels use the close exactly 15 minutes after the decision candle in the same uninterrupted session segment: UP > +0.10%, DOWN < -0.10%, otherwise NEUTRAL.

| Split | Rows | Trading dates | Date range | DOWN / NEUTRAL / UP |
|---|---:|---:|---|---|
| train | 4,883 | 154 | 2025-11-13–2026-08-05 | 1323 / 2364 / 1196 |
| validation | 1,161 | 33 | 2026-08-06–2026-09-24 | 277 / 623 / 261 |
| test | 1,369 | 34 | 2026-09-25–2026-11-12 | withheld until final evaluation |

Date allocation was chronological 70/15/15 across **221 eligible dates** (floor at 70% and cumulative 85%). Purging dropped 0 train and 0 validation examples because all target endpoints stayed within their same-session dates. All saved feature cells are finite, and split schemas/order match. Dataset construction validation was run before model fitting; the new test file itself was not loaded by the training script.

## Validation comparison

Candidate selection used validation only, under a predeclared ordering: macro-F1, then balanced accuracy, directional macro-F1, then lower log loss. Validation has been used in prior project research and is not a pristine holdout. Training times are wall-clock seconds for these fixed configurations.

| Model | Accuracy | Balanced acc. | Macro F1 | UP P/R | DOWN P/R | Directional macro precision | Directional macro F1 | Directional coverage | Accuracy on directional predictions | Log loss | Fit sec | Predict val sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.447 | 0.362 | 0.360 | 0.236/0.180 | 0.253/0.267 | 0.244 | 0.232 | 0.424 | 0.246 | 1.073 | 0.173 | 0.003 |
| random_forest | 0.447 | 0.362 | 0.361 | 0.232/0.230 | 0.297/0.217 | 0.264 | 0.241 | 0.397 | 0.260 | 1.054 | 1.750 | 0.111 |
| xgboost | 0.403 | 0.366 | 0.362 | 0.242/0.337 | 0.266/0.271 | 0.254 | 0.275 | 0.556 | 0.253 | 1.094 | 0.906 | 0.007 |
| hist_gradient_boosting | 0.388 | 0.359 | 0.354 | 0.235/0.326 | 0.266/0.292 | 0.251 | 0.276 | 0.573 | 0.250 | 1.112 | 3.484 | 0.018 |
| always_neutral | 0.537 | 0.333 | 0.233 | 0.000/0.000 | 0.000/0.000 | 0.000 | 0.000 | 0.000 | — | — | — |
| fixed_rsi_rule | 0.449 | 0.340 | 0.331 | 0.244/0.180 | 0.242/0.141 | 0.243 | 0.193 | 0.305 | 0.243 | — | — |

Selected validation candidate: **xgboost** (XGBoost), because it narrowly led macro-F1 (0.3620 vs Random Forest 0.3615) and had the highest balanced accuracy. Random Forest had lower log loss and higher directional macro precision; the validation evidence is close, not decisive. Always-neutral accuracy was 53.7%; every trained model’s raw accuracy was lower. XGBoost’s in-sample train macro-F1 was 0.637 versus 0.362 validation, while Random Forest and HistGradientBoosting showed much larger train/validation gaps.

## One-time final evaluation of the selected candidate

Selected model: **XGBoost**, artifact SHA-256 `343915e56469b6e80711e31c93d7d94070651a37d8f95bedb848eccfdbbeab79`. New held-out rows: **1,369**; date range 2026-09-25–2026-11-12; actual counts DOWN/NEUTRAL/UP = 334/695/340. Predictions: DOWN 295, NEUTRAL 558, UP 516.

| Evaluation | Accuracy | Balanced accuracy | Macro F1 | UP precision / recall / F1 | DOWN precision / recall / F1 | Directional macro precision / recall / F1 | Directional coverage | Directional-signal accuracy | Log loss |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| Selected XGBoost | 0.3857 | 0.3623 | 0.3556 | 0.267 / 0.406 / 0.322 | 0.261 / 0.231 / 0.245 | 0.264 / 0.318 / 0.284 | 0.592 | 0.2651 | 1.0979 |
| Always neutral | 0.5077 | 0.3333 | 0.2245 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | — | — |
| Fixed RSI rule | 0.4558 | 0.3446 | 0.3245 | 0.338 / 0.141 / 0.199 | 0.217 / 0.123 / 0.157 | 0.277 / 0.132 / 0.178 | 0.242 | 0.2688821752265861 | — |

Test confusion matrix (actual rows; predicted columns; order DOWN, NEUTRAL, UP):

```text
[77, 119, 138]
[142, 313, 240]
[76, 126, 138]
```

The selected model’s test accuracy (38.6%) is below always-neutral (50.8%), although balanced accuracy and macro-F1 are higher. Test macro-F1 (0.3556) and balanced accuracy (0.3623) were slightly lower than validation; log loss changed from 1.0943 to 1.0979. This is classification performance, not trading profitability. No strategy replay or costs were evaluated.

XGBoost float32 probability sums differed from 1 by at most 8.94e-08 in the saved outputs, within the 1e-6 validation tolerance. scikit-learn emitted a normalization warning during test log-loss calculation; returned probabilities were finite, nonnegative and approximately normalized. No inference was run again to address that warning.

## Integrity, artifacts and limitations

The existing live frozen B artifact hash remains `31188a09eba09d58d441324fcfb130a23268a833a459709e4c0c6ad9ca45d08d` (equal to its saved frozen specification hash); it was not loaded for training or modified. The new candidate hash is recorded in `models/alpaca_baselines_v1/training_manifest.json`. There was exactly one call of `final_evaluate_selected.py`; its manifest marks the one-time evaluation complete, with no tuning or model modification afterward.

The candidate’s validation advantage is small and uneven by class; UP/DOWN precision remains weak (approximately 0.24–0.30 on validation/test), and raw accuracy loses to always-neutral. **Do not promote or integrate this candidate into the live dashboard.** The original frozen three-class model remains untouched and operational. The new validation/test dates are historical and do not establish future performance or profitability.

Key outputs:

- Canonical data and split CSVs: `data/alpaca_ml_pipeline_v1/`.
- Candidate models, exact feature schema, configs and hashes: `models/alpaca_baselines_v1/`.
- Validation predictions/metrics and one-time test predictions/metrics/confusion matrices: `reports/alpaca_baselines_v1/`.
- Dataset audit: `reports/alpaca_baselines_v1/dataset_validation_report.md`.
- Reproduction commands: `README.md`.
