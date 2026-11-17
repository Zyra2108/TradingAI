"""Replay permitted local train/validation Alpaca bars through the live class.

This audit uses no network and reads only the named train_validation cache
files. Output is isolated from the original parity report.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from bisect import bisect_left, insort
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "reports" / "alpaca_sparse_aggregation_parity" / "compare_sparse_aggregation.py"
spec = importlib.util.spec_from_file_location("sparse_reference_audit", SOURCE)
ref = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(ref)

from live_app.alpaca_aggregation import AlpacaFiveMinuteAggregator
from live_app.alpaca_provider import MinuteBarEvent

UTC = timezone.utc


class MemoryStore:
    """Minimal in-memory implementation of the store surface used by aggregator."""
    def __init__(self):
        self.minutes = {}
        self.minute_keys = []
        self.candles = {}

    def insert_minute_bar(self, event):
        key = event.event_time_utc.astimezone(UTC)
        if key in self.minutes:
            return False, False
        insort(self.minute_keys, key)
        self.minutes[key] = {
            "event_time_utc": key.isoformat().replace("+00:00", "Z"),
            "received_at_utc": event.received_at_utc.isoformat().replace("+00:00", "Z"),
            "symbol": event.symbol, "open": event.open, "high": event.high,
            "low": event.low, "close": event.close, "volume": event.volume,
            "trade_count": event.trade_count, "source": "Alpaca", "feed": "iex",
            "original_event_timestamp": event.original_event_timestamp,
        }
        return True, False

    def get_minute_bars(self, start, end, *, update_cutoff_utc=None):
        left, right = bisect_left(self.minute_keys, start), bisect_left(self.minute_keys, end)
        return [self.minutes[stamp] for stamp in self.minute_keys[left:right]]

    def record_gap(self, *_args, **_kwargs):
        pass

    def bucket_rejection_reason(self, *_args, **_kwargs):
        return None

    def insert_five_minute_candle(self, candle):
        key = candle.candle_time_utc
        if key in self.candles:
            return False, False
        self.candles[key] = candle
        return True, False

    def minute_bar_timestamps(self):
        return self.minute_keys.copy()


def run():
    metadata = ref._validate_metadata()  # Validates exact source, feed and date provenance first.
    minute_meta = metadata["minute"]
    start_day, end_day = ref.START_DATE, ref.END_DATE
    raw = pd.read_csv(ref.MINUTE_PATH)
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="raise")
    raw["date_ny"] = raw["timestamp_utc"].dt.tz_convert("America/New_York").dt.date
    if raw["date_ny"].min() < start_day or raw["date_ny"].max() > end_day:
        raise ValueError("Minute cache escaped the permitted training/validation dates")
    days = set(raw["date_ny"])
    _, sessions = ref._session_map(days)
    minutes, minute_quality, excluded = ref._read_source(ref.MINUTE_PATH, "1Min", start_day, end_day, sessions)

    memory = MemoryStore()
    aggregator = AlpacaFiveMinuteAggregator(memory, finalization_buffer_seconds=120)
    for row in minutes.sort_values("timestamp_utc").itertuples(index=False):
        stamp = row.timestamp_utc.to_pydatetime().astimezone(UTC)
        volume = None if pd.isna(row.volume) else float(row.volume)
        trades = None if pd.isna(row.trade_count) else int(row.trade_count)
        event = MinuteBarEvent(
            "ECL", stamp.isoformat().replace("+00:00", "Z"), stamp,
            stamp + timedelta(minutes=1), float(row.open), float(row.high),
            float(row.low), float(row.close), volume, trades,
        )
        aggregator.ingest(event, now=event.received_at_utc)
    final_now = max(memory.minutes) + timedelta(minutes=4)
    aggregator.recover_due_buckets(now=final_now)

    native, native_quality, native_excluded = ref._read_source(
        ref.NATIVE_PATH, "5Min", start_day, end_day, sessions
    )
    actual = pd.DataFrame([{
        "timestamp_utc": candle.candle_time_utc,
        "open": candle.open, "high": candle.high, "low": candle.low, "close": candle.close,
        "volume": candle.volume, "trade_count": candle.trade_count,
        "observed_minute_count": candle.aggregation_metadata["observed_minute_count"],
    } for candle in memory.candles.values()])
    actual["date_ny"] = pd.to_datetime(actual["timestamp_utc"], utc=True).dt.tz_convert("America/New_York").dt.date
    merged = actual.merge(native, on=["timestamp_utc", "date_ny"], how="outer", suffixes=("_live", "_native"), indicator=True)
    matched = merged[merged["_merge"] == "both"].copy()
    fields = ["open", "high", "low", "close", "volume", "trade_count"]
    for field in fields:
        matched[f"{field}_exact"] = matched[f"{field}_live"].eq(matched[f"{field}_native"])
    matched["ohlc_exact"] = matched[[f"{field}_exact" for field in ["open", "high", "low", "close"]]].all(axis=1)
    matched["all_fields_exact"] = matched[[f"{field}_exact" for field in fields]].all(axis=1)
    by_count = matched.groupby("observed_minute_count", as_index=False).agg(
        matched_buckets=("timestamp_utc", "count"), exact_ohlc=("ohlc_exact", "sum"),
        exact_all_fields=("all_fields_exact", "sum"))
    date_stats = matched.groupby("date_ny", as_index=False).agg(
        matched_buckets=("timestamp_utc", "count"), exact_ohlc=("ohlc_exact", "sum"),
        exact_all_fields=("all_fields_exact", "sum"))
    date_stats["date_ny"] = date_stats["date_ny"].astype(str)
    mismatches = matched.loc[~matched["all_fields_exact"]]
    summary = {
        "source": {"symbol": "ECL", "feed": "iex", "period": [start_day.isoformat(), end_day.isoformat()],
                   "train_validation_only": True, "network_requests": False,
                   "credential_values_saved": bool(minute_meta.get("credential_values_saved"))},
        "minute_data_quality": minute_quality,
        "native_5min_data_quality": native_quality,
        "excluded_minute_rows": len(excluded),
        "excluded_native_5min_rows": len(native_excluded),
        "live_aggregator": "live_app.alpaca_aggregation.AlpacaFiveMinuteAggregator",
        "aggregation_method": "observed_minute_ohlc_v2",
        "counts": {"valid_minute_rows": len(minutes), "live_aggregated_buckets": len(actual),
                   "native_buckets": len(native), "matched_buckets": len(matched),
                   "live_only_buckets": int((merged["_merge"] == "left_only").sum()),
                   "native_only_buckets": int((merged["_merge"] == "right_only").sum()),
                   "mismatched_matched_buckets": len(mismatches)},
        "parity": {"exact_ohlc": int(matched["ohlc_exact"].sum()),
                   "ohlc_rate": float(matched["ohlc_exact"].mean()) if len(matched) else None,
                   "all_fields_exact": int(matched["all_fields_exact"].sum()),
                   "all_fields_rate": float(matched["all_fields_exact"].mean()) if len(matched) else None},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "live_aggregator_parity.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    by_count.to_csv(OUT / "parity_by_observed_count.csv", index=False)
    date_stats.to_csv(OUT / "parity_by_date.csv", index=False)
    mismatches.to_csv(OUT / "mismatches.csv", index=False)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
