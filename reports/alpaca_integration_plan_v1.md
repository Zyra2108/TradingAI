# ECL dashboard Alpaca IEX integration plan

**Scope:** read-only architecture audit and implementation plan. The server was not started, no Alpaca connection/API request was made, no `.env` secret was read or printed, and no test dataset was accessed.

## Current architecture

1. `live_app/app.py::create_app` constructs the engine and starts an async polling task. It calls `IndicatorEngine.poll_once()` in a worker thread every 45 seconds during the NYSE session and only when the Twelve Data display authorization gate is true.
2. `live_app/provider.py::TwelveDataClient.fetch_recent` makes REST `time_series` requests for ECL / NYSE / 5min, parses the response into New York naive timestamps and OHLC rows.
3. `live_app/engine.py::IndicatorEngine.poll_once` filters today's in-session completed 5-minute bars, deduplicates timestamps, stores first observations, calls prediction generation, and retries due outcome settlement.
4. `engine.py::_save_predictions_for_new_bars` calculates the existing 24 causal B inputs using `live_app/features.py::calculate_b_features`, checks readiness, calls the loaded model's `predict_proba`, applies DOWN / NEUTRAL / UP argmax, and inserts immutable prediction rows.
5. `engine.py::_settle_due_predictions` waits until the +15-minute endpoint candle has completed plus the two-minute finalization buffer, verifies the full 0/5/10/15-minute candle sequence with `_has_continuous_outcome_path`, derives the original ±0.10% label, and writes an immutable outcome. Incomplete paths remain due and can retry.
6. `live_app/storage.py::Store` uses SQLite. `/api/candles`, `/api/predictions`, `/api/report`, and `/api/report.csv` in `app.py` serve dashboard data. `static/app.js` passes the candle response to Lightweight Charts with `candles.setData(c)` and places logged predictions with `candles.setMarkers(marks)`.

The SQLite schema was inspected read-only. It has candles, candle conflicts, predictions, outcomes, and app events. Prediction rows store creation/candle time, all three probabilities, confidence, input candle/features, horizon, outcome endpoint, model hash, and provider delay. The reference close is in `input_candle_json`. Settlement is represented by an outcome row; pending status is inferred from its absence. Append-only triggers protect prediction and outcome records.

## Current Twelve Data dependencies

| Component | Existing dependency | Planned replacement/retention |
|---|---|---|
| `config.py` | `TWELVE_DATA_API_KEY`, fixed Twelve Data base URL | Add APCA key/secret loading and a separate Alpaca IEX display authorization setting. Keep secrets server-side. |
| `live_app/app.py` | Imports Twelve key; `TWELVE_DATA_DISPLAY_AUTHORIZED`; REST poll loop | Replace provider-specific constructor inputs and polling task with an Alpaca stream lifecycle task and an Alpaca-specific gate. Keep API routes and response contracts. |
| `live_app/provider.py` | `TwelveDataClient` REST client | Retire from Alpaca mode or retain as a clearly separate legacy adapter. Add a new IEX WebSocket client module; do not silently route it through the Twelve gate. |
| `live_app/engine.py` | Imports Twelve client; status names/strings; `poll_once`; provider error messages | Add methods to accept validated completed 1-minute events, persist/deduplicate them, assemble finalized 5-minute bars, then call the existing prediction and settlement paths. Replace provider status and error text. |
| `live_app/storage.py` | Default candle provider is Twelve Data; candle table lacks volume/trade count and raw receive time | Add additive schema migration for minute-bar staging and receive/source metadata, with named-column inserts. Use a separate Alpaca database by default to avoid collisions or mixing rows with the existing Twelve Data database. |
| `live_app/static/index.html` and `static/app.js` | UI says NYSE and has generic “Feed status” / delay messaging; no explicit feed identity | Label source “Alpaca IEX (exchange-specific; not consolidated)” and expose feed, stream state, last event, receive delay, reconnect/error state. Keep candlesticks, markers, probabilities, history, and export. |
| `live_app/README.md` | Twelve Data plan, endpoint, quota, delay, and authorization guidance | Document Alpaca IEX configuration, stream-only vs. aggregate data distinction, permission confirmation, Windows setup, outages and source limitations. |

No existing code references APCA environment names. The checked-in `.env.example` documents Twelve Data only; `.gitignore` excludes `.env`. `config.py` calls `load_dotenv` before reading the Twelve key, and `app.py` redundantly calls `load_dotenv` again. The actual `.env` was not opened. Proposed variable names are `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, and `ALPACA_IEX_DISPLAY_AUTHORIZED=false`; do not set the gate until the account's applicable display permission is confirmed. Never include credentials in logs, API status, dashboard responses, or exception text.

## Frozen candidate compatibility

`models/alpaca_final_freeze_v1/candidate_pipeline.joblib` was loaded read-only during this audit. It is a fitted scikit-learn `Pipeline` with 24 ordered raw input names and class order `DOWN, NEUTRAL, UP`; it owns the preprocessing and LightGBM estimator. It can be called through `predict_proba` with the existing `calculate_b_features` output and the frozen argmax policy.

The present engine cannot load it as-is: `engine.py` hardcodes `models/final_freeze_v1/B.joblib`, expects a `frozen_model_spec.json` with a `models` list and `primary_candidate == "B"`, and labels health/status as Balanced LightGBM B. The candidate freeze instead has `frozen_candidate_spec.json` and is named `lightgbm_balanced_B_start`. The integration must point explicitly to the candidate pipeline and validate its feature order, class order, 15-minute horizon, thresholds, and SHA-256 against that candidate spec. Do not route it through or overwrite the old B artifact.

## Proposed Alpaca-to-dashboard data flow

1. Load APCA credentials privately at process start. Keep the poller disabled unless the separate IEX display permission gate is true and both credentials exist. Status returns only key-present/missing, never values.
2. Start a supervised WebSocket task connected to `wss://stream.data.alpaca.markets/v2/iex`. Authenticate using APCA headers and the documented stream protocol; subscribe only to ECL minute bars. The earlier one-off diagnostics verified ECL IEX trade and minute-bar events, but that does not establish display rights or guarantee continuous bars.
3. Parse each event's timestamp as timezone-aware UTC, retain raw event time and local receive UTC, validate ECL symbol/feed and OHLC, and convert to New York local time for calendar checks.
4. Persist/deduplicate source one-minute bars. Assemble each 5-minute bar only from the five expected, unique, consecutive one-minute bars aligned to the official NYSE session open. Aggregate open from the first minute, high=max, low=min, close=last; volume/trade count may be summed for audit but must not enter the B feature list. Emit only after the bucket ends and a configurable finalization buffer elapses. Never fill or interpolate a missing minute.
5. Store the derived 5-minute OHLC with source metadata, then pass it to existing `calculate_b_features`. The feature builder already resets at trading dates and 5-minute gaps. Skip prediction if any required B feature is missing; keep all 24 names/order and use the candidate pipeline's raw argmax.
6. Keep `_save_predictions_for_new_bars` behavior for live-only candles, with a restart/start-time guard so reconnect/backfill cannot create historical predictions. Preserve 15-minute session-end filtering.
7. Continue settlement using the derived 5-minute candles. Require all four exact timestamp starts (decision, +5, +10, +15), same session, no duplicates, and endpoint candle close plus finalization delay. Missing bars leave the outcome pending and allow retry. Preserve first finalized observations and immutable predictions/outcomes.
8. Existing API response shapes can remain stable: `/api/candles` yields derived 5-minute OHLC for Lightweight Charts; `/api/predictions` provides logged markers/history/probabilities; `/api/report.csv` exports outcomes. Extend status with `provider=Alpaca`, `feed=IEX`, WebSocket/auth/subscription state, last message time, measured event-to-receive delay, reconnect count, and last safe error.

## Risks and safeguards

- **IEX is venue-specific.** The chart must not imply consolidated US market coverage. IEX trade activity may be thinner than SIP/consolidated data, so some minute bars or complete five-minute buckets may not arrive. No candle should be fabricated to fill that gap.
- **Aggregation parity:** training used Alpaca IEX historical five-minute OHLC; live aggregation from five IEX one-minute bars must be checked against historical five-minute bars on identical timestamps before launch. Minute gaps make aggregation unavailable; differing vendor aggregation or bar timestamp conventions could change features.
- **Completed bars and corrections:** require a finalization delay after all five minute bars arrive. Log later revisions as conflicts; never repaint a prediction or overwrite an outcome.
- **Reconnects:** heartbeat/ping, bounded exponential backoff, re-authentication/resubscription, idempotent timestamp keys, and explicit gap reporting are required. Reconnected data may support candle reconstruction, but must not create retroactive predictions.
- **Session timing:** use `pandas_market_calendars` in `live_app/calendar.py` for holidays and early closes. Anchor aggregation to each session's open, reject pre/post-market events, and reject prediction horizons that extend beyond the same session.
- **SQLite identity:** existing candle primary key is timestamp only, so mixing providers in the current DB could collide. Start Alpaca integration with a new DB file (for example `data/live/ecl_alpaca_iex.sqlite3`) and retain the existing Twelve Data DB untouched. Make inserts explicit by column name for migration safety.
- **Outcome status:** pending is currently inferred by no outcome row; there is no explicit unverifiable status. If the UI needs this distinction, add an append-only settlement status/event table rather than updating immutable prediction rows.
- **Credentials/permissions:** keep `.env` out of source control; add only variable names to `.env.example`. Use an Alpaca-specific display gate, default false. A successful socket connection is not proof that dashboard display is licensed; account entitlement or written permission remains a prerequisite.
- **One-minute WebSocket:** the feed supplies minute bars, not ready-made five-minute B inputs. The in-process aggregator and durable staging table need crash/restart tests so a partial bucket cannot become a prediction.

## Staged implementation plan

1. **Configuration and frozen loader:** add APCA variables and a separate default-off display gate; parameterize model/spec/database paths; load only the Alpaca candidate freeze and validate hashes/contracts. Keep existing official model and settings unchanged.
2. **Provider adapter:** implement authenticated ECL IEX WebSocket connection, minute-bar subscription, heartbeat, event validation, safe errors, reconnect/backoff and status telemetry. No credentials in logs.
3. **Persistence and aggregation:** add versioned additive SQLite schema for minute bars, timestamps/feed metadata and settlement diagnostics. Default to a new Alpaca-only DB. Implement exactly-five-minute aggregation with strict continuity and session alignment.
4. **Prediction and settlement integration:** connect finalized aggregate bars to existing features/predict/save methods; retain immutability and full 15-minute continuity checks. Reject incomplete horizon/session paths.
5. **Dashboard:** preserve API shape where practical; identify Alpaca IEX and non-consolidated coverage; display source status and measured delay; retain history, probabilities, markers, outcomes and CSV export.
6. **Documentation and controlled local validation:** Windows startup instructions, `.env.example` variable names, permission gate, provider limitations, backup/export procedure. Keep live polling off for tests.

## Required tests before considering integration complete

- Frozen candidate loads from the new path; exact 24 raw feature order, preprocessing inclusion, class order and model hash match its specification; no old model artifacts change.
- Test WebSocket client using synthetic messages: auth/subscription success/failure, ping/pong, disconnect, reconnect, server error and secret redaction.
- Minute aggregation tests: complete five-minute bucket; missing each constituent minute; duplicate timestamps; corrected/revised minute; out-of-order event; malformed OHLC; UTC/DST conversion; regular session and early close.
- Completed-candle buffer tests: no prediction before full 5-minute interval plus buffer; no partial aggregate; no retroactive prediction after restart/reconnect.
- Live feature parity against historical Alpaca IEX bars for identical completed timestamps: exact feature names/order, values within stated tolerance, and missing-value behavior.
- Database migration tests on a copy/synthetic DB; existing Twelve Data DB and records remain untouched; unique/idempotent candles; immutable prediction/outcome triggers; durable partial minute buckets and reconnect recovery.
- Outcome settlement tests for exact +15-minute endpoint and continuous 0/5/10/15 sequence, missing or duplicate bar, session boundary/early close, delayed retry, and duplicate outcome prevention.
- FastAPI route tests with poller disabled; dashboard status states, OHLC/markers/probabilities, export CSV and provider/source labels.
- End-to-end local test with fake WebSocket events and fake clock only. Do not claim a live trial until an authorized, actually logged trial is run.

## Blockers and decisions

1. Confirm Alpaca account terms explicitly permit displaying IEX data in the professor's browser demonstration. The app should remain gated until then.
2. Approve the separate Alpaca database path so existing Twelve Data rows are not mixed with IEX bars.
3. Decide the production aggregation rule for absent one-minute IEX bars. Recommended: reject the incomplete five-minute bucket, log the gap, and suppress prediction until the feature pipeline has warmed up again.
4. Approve reusing the current two-minute buffer for completed five-minute aggregates. The value is an application safeguard, not a provider latency guarantee.

No code was edited, no server was started, no Alpaca request was made, no prediction was generated, and no test evaluation was repeated for this plan.
