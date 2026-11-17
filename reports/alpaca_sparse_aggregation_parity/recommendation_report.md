# Alpaca IEX sparse-bar aggregation parity audit

**Decision: parity supports changing the aggregation policy, but this report does not implement that change.** The tested construction—combine all valid observed 1-minute IEX bars in the aligned 5-minute session bucket, with no fill and no empty candle—reproduced Alpaca's native IEX `5Min` bars exactly for every permitted cached bucket. Do not change the live application until its update, disconnect, and idempotency handling is designed and separately reviewed.

## Data used and safeguards

- Symbol/feed: ECL / Alpaca IEX.
- Cached input files: `models/alpaca_1min_research_v1/raw_ecl_iex_1min_train_validation.csv` and `raw_ecl_iex_5min_train_validation.csv`.
- Permitted date range: 2025-11-13 through 2026-09-24 ET; train 2025-11-13 through 2026-08-05; validation 2026-08-06 through 2026-09-24.
- Both cache metadata files declared those same bounds, and the code rechecked every source timestamp before aggregation. No additional data was fetched, no live connection was made, and no test-period data was accessed.
- No application, production database, frozen model, research artifact, or historical replay implementation was modified. No model inference was run.

## Parity results

| Measure | Result |
|---|---:|
| Valid regular-session one-minute bars | 49,347 |
| Valid native five-minute bars | 16,059 |
| Reconstructed five-minute buckets | 16,059 |
| Matched buckets | **16,059** |
| Native-only / reconstructed-only buckets | **0 / 0** |
| Exact OHLC matches | **16,059 / 16,059 (100%)** |
| Exact volume matches | **16,059 / 16,059 (100%)** |
| Exact trade-count matches | **16,059 / 16,059 (100%)** |
| Maximum absolute difference, each OHLCV/count field | **0** |
| Mismatches | **0** |

All **216 trading dates** with matched data had exact OHLC parity. Sparse constituent counts also matched exactly:

| Observed 1-minute bars in bucket | Matched buckets | Exact OHLC | Exact volume | Exact trade count |
|---:|---:|---:|---:|---:|
| 1 | 2,138 | 2,138 (100%) | 2,138 (100%) | 2,138 (100%) |
| 2 | 3,559 | 3,559 (100%) | 3,559 (100%) | 3,559 (100%) |
| 3 | 4,019 | 4,019 (100%) | 4,019 (100%) | 4,019 (100%) |
| 4 | 3,681 | 3,681 (100%) | 3,681 (100%) | 3,681 (100%) |
| 5 | 2,662 | 2,662 (100%) | 2,662 (100%) | 2,662 (100%) |

One premarket row was excluded from each cache (1-minute 09:27 ET and 5-minute 09:25 ET on 2026-09-18). No valid in-session observations were discarded. The aggregation used observed bars only; it did not create candles for zero-observation buckets, fill missing minutes, or carry prices forward.

The source files were fetched separately (about 21 minutes apart). Despite that, no revision or timestamp-alignment difference appeared in the 16,059 matched buckets. This is evidence against a material revision issue in these files, not a guarantee that live updates can never revise a bar.

## Feature compatibility

The reconstructed stream and native stream produced the same timestamps, the same **797** session/gap-separated feature segments, and the same warm-up availability. The exact frozen **24-feature order** was preserved. **6,792** rows were feature-ready in each stream, and every feature value matched exactly on all of those common ready rows. Feature NaNs also matched at all common timestamps.

This is strong evidence that sparse within-bucket minute observations do not change the frozen model's feature inputs when aggregation follows the tested rules. A completely empty 5-minute interval remains absent, causing the existing historical feature builder to reset its rolling segment as designed. No volume or trade-count field enters the frozen B model features.

## Recommended policy for a future implementation

1. Finalize an aligned five-minute session bucket when it contains **at least one valid, unique ECL IEX minute bar**. The parity audit supports every observed count from 1 through 5; do not impose a five-minute-bar count threshold.
2. For a nonempty bucket, use first observed open, maximum observed high, minimum observed low, last observed close, and sums of observed volume and available trade counts. Keep observed-minute count and missing timestamps as audit metadata.
3. For an entirely empty bucket, write no candle. Never interpolate or forward-fill. A later feature builder should see the missing five-minute timestamp and reset rolling warm-up exactly as it does for native historical gaps.
4. Require the existing session/calendar and bucket-boundary checks. Do not bridge regular-session boundaries, holidays, or early closes.
5. Keep the existing 120-second finalization grace as a reasonable starting point, but process Alpaca `updatedBars` before finalization. Alpaca documents these as late-trade revisions emitted after half-minute marks. The current project client subscribes to ordinary `bars` and expects ordinary bar events; it does not yet implement updated-bar revisions. Preserve revision history, choose the latest valid observation before the bucket becomes immutable, and never rewrite an already-finalized candle or prediction.
6. If a known disconnect or processing error overlaps a bucket, mark it untrusted and do not finalize unless an approved, source-matched reconciliation supplies the missing IEX observations. A connected status alone cannot prove that no individual message was lost.
7. Keep per-bucket persistence idempotent and serialize finalization. Generate a prediction only from a finalized candle whose features are ready, using its own close and timestamp. Preserve the frozen 15-minute outcome path; missing future 5-minute candles must leave settlement pending.

## Official documentation vs. audit inference

**Official Alpaca documentation:** stock minute bars represent trade aggregates and are emitted when qualifying bar fields are available; the FAQ states no bar is emitted if there are no trades in the interval (or prices are not established). It says higher non-minute/non-daily timeframes are built from minute/daily bars with first open, extrema, last close, and summed volume/trade count. The historical bars endpoint supports a `5Min` timeframe. Alpaca's stock stream documents minute bars and `updatedBars`, with late-trade updates after half-minute marks. The Basic plan's real-time equities feed is IEX, not full consolidated SIP. [Market Data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq), [Stock stream documentation](https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data), [Historical bars API](https://docs.alpaca.markets/us/reference/stockbars), [Market data plans](https://docs.alpaca.markets/us/docs/about-market-data-api)

**Inference from this audit:** the absent one-minute bars in the prior live trial are consistent with normal sparse IEX bar emission; they do not establish that ECL had no trading across the overall US market. In this historical cache, omitting absent minute bars and aggregating the actual available bars is exactly equivalent to Alpaca's native IEX five-minute output. The feature equality results support keeping the frozen model unchanged if—and only if—the live implementation matches this audited data construction and handles revisions and known outages safely.

## Artifacts

- `compare_sparse_aggregation.py` — reproducible local-only comparison; refuses dates outside the permitted train/validation range.
- `summary.json` — machine-readable counts, parity, source quality, and feature results.
- `parity_by_observed_minute_count.csv` — detailed sparse-count parity.
- `parity_by_date.csv` — per-date counts and match rates.
- `mismatch_details.csv` — header-only; no matched-bucket mismatches were found.
- `bucket_coverage.csv` — matched/missing bucket inventory and observed counts.
- `excluded_source_rows.csv` — the two excluded premarket rows and reasons.
- `feature_compatibility_by_feature.csv` and `feature_compatibility_audit.md` — per-feature equality and warm-up audit.
- `reproducibility_manifest.json` — source/code/spec hashes and no-test/no-network assertions.
- `test_sparse_aggregation.py` — focused synthetic and cached-result regression tests.

## Limitations

Historical API bars are finalized historical aggregates; live streaming can include delayed updates. The app must process those updates before it finalizes an immutable candle. A stream may also lose a message without a visible disconnect; the current bar event schema has no per-minute “no-trade” heartbeat. The audit therefore establishes aggregation parity from the supplied caches, not complete live-feed delivery or prediction performance. No new model selection or test evaluation was performed.
