# Sparse Live Aggregation Implementation Report

## Outcome

The live five-minute aggregator now follows the parity-verified sparse IEX policy. It accepts a bucket only after its existing 120-second finalization buffer, only when at least one valid and unique ECL minute observation exists, and only when the bucket has not overlapped a recorded stream interruption or processing failure. It sorts the actually observed minutes and calculates OHLC from the first/maximum/minimum/last observed values. Volume and trade count are sums of the observations available for audit. Candle metadata records the observed count and missing minute timestamps. Empty buckets are omitted; no data is filled or inferred.

All bucket boundaries remain anchored to the NYSE session open and use the existing session calendar, including DST and early closes. Sparse source-minute counts do not become model features. A gap between completed five-minute candles remains a real gap and follows the existing feature pipeline's continuity and warmup rules.

## Late updates and interruptions

The provider subscribes to both ECL `bars` and `updatedBars`. A timely `updatedBars` message is treated as a full replacement for that minute and stored separately from the original observation. The latest eligible replacement is used once when the pending five-minute bucket is finalized; repeated identical updates do not increase volume or trade count.

An update arriving after the 120-second cutoff or after candle finalization is preserved in the append-only `late_minute_updates` diagnostics. It does not alter the candle, prediction or outcome. Stream-down intervals and timestamped processing failures are also persisted. Overlapping buckets are rejected. Reconnect recovery does not bridge gaps. The existing engine lock and database uniqueness constraints keep finalization and prediction creation idempotent.

The application status response now includes the latest late-update diagnostic and open interruption count. No dashboard design or Historical Replay code changed.

## Historical verification

The existing parity procedure was rerun in a separate output directory against only the existing ECL IEX training/validation caches (2025-11-13 through 2026-09-24). The live `AlpacaFiveMinuteAggregator` was also replayed in memory over those cached minute bars and compared with the cached native 5-minute bars.

- Live implementation: 16,059 aggregated buckets; all 16,059 matched native buckets.
- Exact OHLC, volume and trade-count parity: 100%; every reported absolute difference was zero.
- Matched bucket counts by observed minute count: 1 → 2,138; 2 → 3,559; 3 → 4,019; 4 → 3,681; 5 → 2,662. Every group matched exactly.
- The source file had 49,348 rows, with one out-of-session row excluded. The native 5-minute source likewise contained one out-of-session row, leaving 16,059 eligible rows. No eligible native-only or reconstructed-only bucket remained.
- Feature audit: all 24 frozen features matched on all 6,792 feature-ready common timestamps; both series had 797 continuity segments and identical missing-value status.
- The permitted source metadata and all New York timestamps were checked against the date ceiling of 2026-09-24 before analysis. No network request or test-period access occurred.

## Frozen contract

The model artifact remains `models/alpaca_final_freeze_v1/candidate_pipeline.joblib`, SHA-256 `31188a09eba09d58d441324fcfb130a23268a833a459709e4c0c6ad9ca45d08d`. The specification still has 24 features in its original order, classes `DOWN, NEUTRAL, UP`, a 15-minute horizon, ±0.10% labels and raw three-class argmax. No model or feature definition changed.

## Tests

- Focused provider, aggregation, engine and app tests: **91 passed, 1 warning**.
- Sparse parity regression tests: **7 passed**.
- Full project suite: **186 passed, 7 warnings**.
- Tests use synthetic events and temporary databases. The test suite did not open the frozen test dataset.

## Live status and remaining blocker

No real Alpaca connection or capture was attempted for this change. The account's previously observed WebSocket 406 connection-limit issue remains unresolved. A new live trial should wait until competing Alpaca stream clients are closed and the account connection slot is available. The sparse policy removes the unsupported requirement for five activity-bearing one-minute bars, but it does not establish live-feed availability or guarantee a candle for an interval with no observed IEX bar.
