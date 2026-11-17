# One-time Alpaca IEX final test evaluation

Selected model by validation only: **xgboost**; artifact SHA-256 `343915e56469b6e80711e31c93d7d94070651a37d8f95bedb848eccfdbbeab79`.
Test rows: 1,369; dates 2026-09-25–2026-11-12.

This is a new Alpaca-only chronological test split. The date period has been represented in earlier project test research, so it is not a pristine project-level date holdout. No test results were used for selection or tuning, and no model changes followed this evaluation.

| Evaluation | Accuracy | Balanced accuracy | Macro F1 | Directional precision | Directional recall | Directional F1 | Directional coverage | Signal accuracy | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selected_model | 0.3857 | 0.3623 | 0.3556 | 0.2642 | 0.3182 | 0.2836 | 0.5924 | 0.2651 | 1.0979 |
| always_neutral_baseline | 0.5077 | 0.3333 | 0.2245 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — |
| train_majority_baseline | 0.5077 | 0.3333 | 0.2245 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — |
| fixed_rsi_baseline | 0.4558 | 0.3446 | 0.3245 | 0.2775 | 0.1320 | 0.1780 | 0.2418 | 0.2689 | — |

## Actual class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| DOWN | 0.2610 | 0.2305 | 0.2448 | 334 |
| NEUTRAL | 0.5609 | 0.4504 | 0.4996 | 695 |
| UP | 0.2674 | 0.4059 | 0.3224 | 340 |

Actual test classes: DOWN 334, NEUTRAL 695, UP 340.
Selected-model predictions: DOWN 295, NEUTRAL 558, UP 516.

Confusion matrix (rows are actual; columns predicted; order DOWN, NEUTRAL, UP):

| Actual \ Predicted | DOWN | NEUTRAL | UP |
|---|---:|---:|---:|
| DOWN | 77 | 119 | 138 |
| NEUTRAL | 142 | 313 | 240 |
| UP | 76 | 126 | 138 |

Validation comparison for the selected candidate: accuracy 0.4031, balanced accuracy 0.3658, macro-F1 0.3620, log loss 1.0943. Test values were 0.3857, 0.3623, 0.3556, 1.0979 respectively.

Probability check: finite, nonnegative, and row sums approximately 1 (within 1e-6). XGBoost float32 softmax caused a maximum saved-row sum deviation of about 8.94e-08; scikit-learn emitted a warning during log-loss evaluation. No second inference/evaluation was run.
