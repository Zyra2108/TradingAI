# Alpaca ECL custom ensemble research

Training-only OOF rows: 3,375 across expanding folds; validation rows: 1,161. Test was not opened.

## Validation results

All results below use the fixed DOWN, NEUTRAL, UP class order.

| Candidate | Accuracy | Balanced accuracy | Macro-F1 | Directional precision | Directional recall | Directional F1 | Coverage | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Most-frequent dummy | 0.537 | 0.333 | 0.233 | 0.000 | 0.000 | 0.000 | 0.0% | 16.702 |
| Conservative LightGBM | 0.489 | 0.354 | 0.336 | 0.282 | 0.133 | 0.178 | 22.0% | 1.031 |
| Balanced LightGBM B settings | 0.412 | **0.378** | **0.374** | 0.282 | **0.322** | **0.299** | 53.4% | 1.138 |
| Multinomial logistic | 0.529 | 0.338 | 0.261 | 0.243 | 0.024 | 0.044 | 4.7% | **1.009** |
| Equal soft vote | 0.495 | 0.369 | 0.357 | **0.299** | 0.163 | 0.209 | 25.2% | 1.029 |
| OOF-weighted soft vote | 0.520 | 0.335 | 0.266 | 0.218 | 0.032 | 0.055 | 6.8% | 1.008 |
| Class-specific OOF weights | 0.516 | 0.346 | 0.300 | 0.261 | 0.068 | 0.108 | 12.2% | 1.013 |
| OOF stacking logistic | 0.536 | 0.336 | 0.245 | 0.268 | 0.009 | 0.018 | 1.6% | 1.009 |
| Regime-weighted OOF blend | 0.520 | 0.336 | 0.270 | 0.228 | 0.035 | 0.061 | 7.1% | 1.009 |
| Confidence selective weighted vote | 0.537 | 0.333 | 0.233 | 0.000 | 0.000 | 0.000 | 0.0% | 1.008 |
| Agreement selective weighted vote | 0.537 | 0.333 | 0.233 | 0.000 | 0.000 | 0.000 | 0.0% | 1.008 |

The confidence and agreement policies had no OOF threshold satisfying their predeclared minimum 20% directional-coverage constraint, so the frozen policy abstains on every validation row. Their log loss is for the underlying weighted-vote probabilities, before the abstention decision. Confusion matrices and class distributions are saved in the accompanying CSVs.

The equal vote is the strongest ensemble on validation, but still trails balanced LightGBM B settings on three-class macro-F1 (0.357 vs 0.374) and directional F1 (0.209 vs 0.299). None of the ensembles beat the best individual model across those main criteria.

## Training-only OOF base-model diagnostics

These are pooled predictions on later blocks from three expanding folds, with the model refit on earlier dates for each block. They are distinct from the outer validation results above. The ensemble weight learner was fit on these OOF predictions, so these rows are development data for the combiner, not an independent ensemble-performance estimate.

| Base model | Accuracy | Balanced accuracy | Macro-F1 | Directional F1 | Coverage | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Conservative LightGBM | 0.434 | 0.349 | 0.323 | 0.188 | 25.3% | 1.110 |
| Balanced LightGBM B settings | 0.391 | 0.351 | 0.350 | 0.266 | 47.8% | 1.258 |
| Multinomial logistic | 0.469 | 0.349 | 0.275 | 0.094 | 8.5% | 1.067 |

## OOF construction and leakage controls

- Base predictions use expanding chronological folds: train on earlier dates, predict the next block; the training fit is purged where `future_datetime` reaches the first predicted timestamp.
- Each base-model preprocessing pipeline is fitted separately inside each fold. The full-training individual pipelines were reused only for validation inference.
- Global/class-specific/regime weights, stacking logistic meta-model and selective thresholds are learned from training OOF predictions only. Validation labels are used only for final metric calculation and comparison.
- OOF base-model scores are out-of-fold. Scores from ensemble parameters fit on these OOF labels would be in-sample for the combiner, so validation is the valid comparison for the frozen combinations.

## Ensemble behavior

Global weights in order `['lightgbm_conservative', 'lightgbm_balanced_B_start', 'multinomial_logistic']`: `[0.3193, 0.0157, 0.665]`.
OOF regime counts: `{'trending': 516, 'sideways': 2013, 'high_volatility': 846}`; minimum for regime-specific weights: 300. Thin regimes fall back to global weights.
Selective confidence threshold: 1.0000; OOF coverage 0.0%. Agreement policy requires at least 2 of 3 base models to agree on UP or DOWN; its confidence threshold is 1.0000 (OOF coverage 0.0%).
OOF pairwise prediction agreement: `{'lightgbm_conservative__lightgbm_balanced_B_start': 0.7318518518518519, 'lightgbm_conservative__multinomial_logistic': 0.7282962962962963, 'lightgbm_balanced_B_start__multinomial_logistic': 0.5226666666666666}`.

The individual balanced LightGBM from the parent candidate directory remains the fallback. No ensemble is automatically promoted. Review macro-F1, directional precision/recall/F1, coverage and log loss together; complexity alone is not evidence of value.
