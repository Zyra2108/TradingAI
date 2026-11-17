# ECL Binary UP/DOWN Research v2

## Prior-round audit

Independent recalculation from saved v1 validation predictions matched reported UP/DOWN eligible precision, emitted eligible accuracy, all-emission accuracy, and emission counts for every model/coverage row: **True**.
V1 reports UP and DOWN precision separately. It distinguishes eligible directional precision (actual NEUTRAL excluded) from operational precision (a signal on an actual NEUTRAL candle counts as a false positive). Its main coverage target is percent of all validation candles; eligible-direction coverage is separately reported. The near-balanced predictions and training directional counts do not suggest the small v1 result was simply majority-class exploitation; performance was near the directional majority and return-sign baselines, with weak confidence separation/calibration.
Likely limits: only 2,519 directional training rows; one stock and one 15-minute labeling rule; noisy thresholded returns; overlapping multi-horizon price-derived features; no volume or raw OHLC fields exist in the supplied train/validation CSVs. V1 XGBoost/LightGBM produced near-even UP/DOWN predictions but only modest accuracy above the directional baselines, and its validation confidence was not strongly calibrated. The prior experiment itself was not edited.

## Controlled approach

Three date-ordered expanding training folds compare Logistic Regression, ExtraTrees, Random Forest, XGBoost, and LightGBM on the exact original 24 features and a small set of row-local causal agreement/ratio features. CatBoost was not installed and was not added. Preprocessing is fitted separately inside every fold. Training examples whose 15-minute label endpoint reaches the fold evaluation start are purged. Model/feature ranking and coverage thresholds use training-only OOF predictions; the corrected validation confirmation follows training-only selection. A provisional v2 evaluation was discarded after an all-candle versus eligible-direction coverage-denominator mismatch was identified; it did not affect model ranking, but the correction means validation predictions were produced in two v2 passes.
The CSVs have no volume, open, high, or low columns, so no volume or new candle structure was fabricated. Added features use only the 24 existing same-candle causal inputs. Reliability-bin CSVs for OOF folds and validation are included; aggregate calibration results are in the JSON metrics. Reliability-bin CSVs for OOF folds and validation are included; aggregate calibration results are in the JSON metrics. Reliability-bin CSVs for OOF folds and validation are included; aggregate calibration results are in the JSON metrics.

## Selection and walk-forward evidence

Selected by the declared OOF rule: **logistic_regression** with **original_24**. OOF 30% policy mean eligible-only macro precision 56.0%; mean emitted eligible directional accuracy 54.9%; mean eligible directional coverage 30.0%.
See `walk_forward_model_ranking.csv`, `walk_forward_fold_metrics.json`, `walk_forward_selective_fold_metrics.json`, `walk_forward_regime_robustness.csv`, and `walk_forward_oof_predictions.csv` for all candidates/folds. Fold consistency is summarized as means and standard deviations; individual UP/DOWN fold values and class confusion matrices are retained.

## Corrected validation confirmation

Validation contains 1,161 candles: 277 DOWN, 623 NEUTRAL, 261 UP; 538 eligible directional examples. At the predeclared 30% OOF threshold: 157 signals, 1004 NO SIGNAL, 13.9% eligible directional coverage; eligible UP precision 48.6%, DOWN precision 44.7%, macro directional precision 46.7%, eligible emitted accuracy 46.7%, and all-emission accuracy counting neutral actuals as wrong 22.3%.
A nominal 20% OOF eligible-coverage threshold produced only 6.5% validation eligible coverage (35 eligible emitted examples); its highest macro precision/accuracy (65.8%/65.7%) does not meet the coverage requirement. Among policies reaching at least 20% validation eligible coverage, best macro precision was 50.2% and best eligible emitted accuracy was 50.0%. The three goals were not met together.
At the validation 30% threshold, 6 dates had at least five eligible emitted predictions; their mean eligible directional accuracy was 41.6% with standard deviation 20.3%. See daily and volatility-regime files. This is variable-date diagnostic evidence, not independent confirmation. The validation set was run through inference once provisionally and once for the corrected coverage policy; the provisional results were discarded and not used for model ranking or threshold selection.

## Recommendation and limitations

Do not integrate this candidate into the live dashboard based on this research. It is worth retaining as an isolated research artifact only if future work has a separately justified target/data source; it did not establish the stated precision and accuracy targets at the required eligible-direction coverage. The validation set was used in prior project research and is not pristine. No frozen test dataset or test-period data was accessed. No live data, trading, or profitability evaluation was performed.
