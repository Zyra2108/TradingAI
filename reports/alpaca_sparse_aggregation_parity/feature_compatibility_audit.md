# Sparse IEX aggregation — feature compatibility audit

## Scope and protection

This audit used only the existing Alpaca IEX ECL train/validation caches covering New York dates **2025-11-13 through 2026-09-24**. Training dates were 2025-11-13 through 2026-08-05; validation dates were 2026-08-06 through 2026-09-24. The cache provenance was checked before reading either CSV, and all timestamps were verified to remain within that permitted end date. No network request was made. No test-period candle, test split file, or frozen test dataset was accessed.

Feature generation reused `live_app.features.calculate_b_features`, which invokes the historical functions from `src/11_create_features.py` and `src/17_add_time_features.py`. The frozen B specification supplied the exact 24 feature names and order. No model inference, training, or prediction was performed.

## Data quality and alignment

| Check | 1-minute cache | Native 5-minute cache |
|---|---:|---:|
| Rows in source | 49,348 | 16,060 |
| Valid regular-session rows | 49,347 | 16,059 |
| Duplicate timestamps | 0 | 0 |
| Invalid OHLC rows | 0 | 0 |
| Misaligned in-session timestamps | 0 | 0 |
| Missing volume/trade count | 0 / 0 | 0 / 0 |
| Rows excluded as outside regular session | 1 | 1 |

The excluded minute bar is **2026-09-18 09:27 ET** and the excluded native five-minute bar is **09:25 ET**, both premarket. They were not mapped into regular-session buckets. Bucket starts were aligned to each NYSE session open; the calendar accounts for DST and early closes. There were no duplicate observations to resolve.

## Feature results

- The reconstructed and native streams each produced **16,059** 5-minute rows with identical timestamps.
- The exact frozen feature order contained 24 unique inputs in both calculations.
- The historical feature builder split both streams into **797** continuous segments at session and missing-five-minute boundaries.
- Both streams had **6,792 feature-ready rows**. The remaining rows had the same warm-up NaN pattern in both streams.
- Across all common timestamps, feature missingness was identical. For all **6,792** rows ready in both, all 24 feature values matched exactly (zero absolute difference).
- The original rolling windows therefore warm up and reset after genuine 5-minute gaps exactly as in the native historical path. A missing five-minute candle remains a discontinuity; sparse constituent minutes inside a reconstructed bucket do not create a five-minute gap when the native bar exists.

This demonstrates feature parity for the allowed cached period; it does not itself validate prediction accuracy or guarantee that a future live stream will never lose messages.

## Artifacts

- `feature_compatibility_by_feature.csv` — counts and exact/absolute differences for each frozen input.
- `summary.json` — feature, source-quality, and parity totals.
- `reproducibility_manifest.json` — input hashes, code hashes, date boundary, environment, and aggregation definition.
