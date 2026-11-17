# ECL Model and SMA Indicators

## Implementation

The chart now has one **Indicator** selector with two exclusive choices:

- **Model Indicator**: saved AI prediction markers only. Historical replay markers use the `HB` label and historical color/shape; live-ledger markers use `LIVE` labels and the existing UP/DOWN/NEUTRAL colors.
- **SMA Indicator**: the existing simple trailing average line over 20 actual completed five-minute closes. Its calculations reset at NYSE session boundaries and at genuine missing-candle gaps. A mode change updates overlays without changing the visible time range or selection.

Historical model signals are generated offline with the integrity-checked frozen candidate, canonical `calculate_b_features` implementation, exact 24-feature order, DOWN/NEUTRAL/UP probability order and raw argmax policy. Candle selection and dashboard refresh only read saved rows; neither path runs model inference. Historical records are in a separate append-only database and cannot update or delete existing records. The live prediction and settlement ledger remains the source for genuine live signals and remains unchanged.

Each historical record is explicitly sourced as `historical_backtest`. The selected-candle panel says **Historical Backtest** and calls its decision time retrospective. Historical predictions from the model's original training interval are in-sample and must not be interpreted as out-of-sample performance or as signals that were actually issued live. The original validation interval is also previously used research data, not a pristine holdout.

## Historical generation

Only the integrity-checked Alpaca IEX train/validation candle cache was loaded. The source loader enforces the approved ET date range **2025-11-13 through 2026-09-24**. No model test CSV was loaded; the generator rejects any source candle dated after 2026-09-24. The separate later backfill is not used by the generator.

Source cache SHA256: `7d5bce7c8925b9cb1f27aba122f9ce97b0459a1bdce826bdee1f70f3d109880c`.

| Item | Result |
|---|---:|
| Permitted five-minute candles processed | 16,059 |
| Session/gap-contiguous segments | 797 |
| Candles with all 24 finite features | 6,792 |
| Historical predictions saved | 6,792 |
| Warm-up skips at the start of session/gap segments | 9,267 |
| Unexpected missing-feature rows after 20-candle warm-up | 0 |
| Outcomes settled from the exact same-session decision/+5/+10/+15-minute candle path | 6,044 |
| Outcomes unavailable | 748 |
| Unavailable: endpoint falls outside the same NYSE session | 437 |
| Unavailable: one or more required outcome-path candles is missing | 311 |

The 9,267 skipped rows equal the first 20 observations of each segment, capped by the segment length. Required-feature missing counts are in `historical_generation_report.json`; these overlap by feature and are not additive. The outcome endpoint is exactly 15 minutes after the decision candle timestamp, matching the existing engine's settlement convention. No candles or prices were filled or inferred.

The historical signal database contains 6,792 unique rows, all dated no later than **2026-09-24**, all labeled `historical_backtest`. Probability rows are finite and sum from 0.9999999999999998 to 1.0000000000000002. A repeated generation run inserted zero duplicates. Frozen candidate SHA256 verified as:

`31188a09eba09d58d441324fcfb130a23268a833a459709e4c0c6ad9ca45d08d`

## SMA verification

The SMA remains a trailing mean of 20 consecutive valid closes; it does not use partial windows, missing values as zero, or gap-filling. An independent pass over the same permitted pre-test candles found 7,062 valid SMA 20 points across the 797 session/gap segments, from 2025-11-13 16:05 UTC through 2026-09-24 16:40 UTC. The browser helper's independent-value, warm-up, session-reset, missing-gap whitespace, viewport-independence, and disabled-overlay tests pass.

## Tests and limitations

- `tests/test_historical_model_indicator.py`: 7 passed.
- Relevant existing app/API tests selected by name: 3 passed.
- `tests/test_chart_viewport.cjs`: 13 passed.
- `node --check live_app/static/alpaca.js`: passed.
- The full project suite was not run because the broader unified-market suite opens its separate post-2026-09-24 historical backfill. The focused checks cover the modified generator, API and chart behavior without running that suite.
- During an earlier broad app-test run, an existing unified-timeline test instantiated the standard market loader, which reads the verified Sep 25–Nov 12 historical backfill. That data was not used for model-signal generation or inference, and `test.csv` was not opened. The actual historical signal generation and its report are restricted to the approved source cache ending Sep 24.
- No live WebSocket, live server, or real-time inference was started. This verifies the historical indicator implementation only; it does not establish live end-to-end operation.

## Artifacts

- Historical records: `data/historical/ecl_model_indicator_backtest.sqlite3`
- Reproduction command: `.venv\\Scripts\\python.exe scripts\\generate_historical_model_signals.py`
- Generation report: `reports/model_indicator_v1/historical_generation_report.json`
