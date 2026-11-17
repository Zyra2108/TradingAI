# Alpaca IEX Sparse Aggregation Policy

## Finalized five-minute candles

The live aggregator groups minute-bar event timestamps into five-minute buckets aligned to the official NYSE regular-session open. After the existing 120-second finalization buffer, it finalizes a bucket only when at least one valid, unique observed ECL IEX minute bar exists. It sorts those observed bars by timestamp and calculates open from the first observed open, high as the maximum observed high, low as the minimum observed low, close from the last observed close, and volume/trade count as sums of observed values. The candle metadata records the observed minute count and every expected minute timestamp that was absent. Empty buckets produce no candle. No price or volume is imputed.

The regular session calendar governs holidays, daylight-saving changes and early closes. Buckets that would extend beyond the session close are rejected. Feature generation sees the same session-anchored five-minute timestamps as before; a completely absent five-minute bucket remains a genuine gap and existing feature continuity/warmup handling applies.

## Late `updatedBars`

The Alpaca stock stream documents `updatedBars` as event type `T="u"`, emitted after the half-minute mark for late trades; its timestamp identifies the minute being revised. The provider now subscribes to both `bars` and `updatedBars` for ECL.

An updated bar is a full replacement observation for that minute, not an increment. Before finalization, and only when received no later than bucket end plus the existing 120-second buffer, the update is stored as a separate immutable revision and the pending five-minute candle is rebuilt from the latest eligible minute version. Volume and trade count are replaced for that minute and are never added twice. Identical repeated updates are idempotent.

An update received after the cutoff or after the five-minute candle is persisted is stored in `late_minute_updates` with source-minute time, receive time, and reason. It cannot change the finalized candle, prediction, or outcome. The API status exposes the late-update count and most recent diagnostic timestamp.

## Interruptions and processing failures

The provider emits safe connection-state transitions. Unavailable and disconnected intervals are persisted in the separate Alpaca database and any overlapping five-minute bucket is rejected. Successful subscription closes the current interruption. At application restart, a previously open interval is closed at the new startup time; it remains in the audit history. A malformed ECL minute event with a usable timestamp, or a failed application callback, invalidates its corresponding five-minute bucket. If an invalid event has no usable event time, its receive-time bucket is conservatively invalidated.

Reconnect does not backfill or infer missing minutes. Newly delivered valid bars are stored normally, but a bucket overlapping the recorded outage remains rejected. Existing database primary keys, the engine lock and idempotent inserts prevent duplicate candles or predictions.

## Model compatibility and limits

Only candle construction and source-event handling change. The frozen candidate, artifact hash, 24-feature names/order, feature generation, class order (`DOWN`, `NEUTRAL`, `UP`), raw argmax, 15-minute horizon and ±0.10% outcome rule are unchanged. Volume/trade count remain auditing fields and are not model inputs.

Sparse aggregation matches historical native Alpaca IEX 5-minute bars on the permitted train/validation cache, including 1–5 observed source-minute bars. The parity result supports the transformation for this symbol/feed/cache. It does not guarantee that every live bucket will contain a minute bar or that IEX observations match consolidated-market activity. No live capture was performed for this change.
