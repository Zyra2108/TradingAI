from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from reports.alpaca_sparse_aggregation_parity.compare_sparse_aggregation import (
    OUT, ROOT, _session_map, reconstruct,
)
from live_app.calendar import session_for_date
from live_app.features import calculate_b_features, ready_feature_row

SPEC = json.loads((ROOT / "models/alpaca_final_freeze_v1/frozen_candidate_spec.json").read_text(encoding="utf-8"))
FEATURES = SPEC["features_exact_order"]


def minute_rows(opened: pd.Timestamp, offsets: list[int]) -> pd.DataFrame:
    rows = []
    prices = {0: (100.0, 100.5, 99.8, 100.2), 1: (100.2, 100.7, 100.0, 100.4),
              2: (100.4, 100.9, 100.1, 100.6), 3: (100.6, 101.0, 100.3, 100.8),
              4: (100.8, 101.2, 100.6, 101.0)}
    for offset in offsets:
        o, h, l, c = prices[offset]
        stamp = opened + pd.Timedelta(minutes=offset)
        rows.append({"timestamp_utc": stamp, "bucket_start_utc": opened,
                     "open": o, "high": h, "low": l, "close": c,
                     "volume": 10 + offset, "trade_count": 1 + offset})
    return pd.DataFrame(rows, columns=["timestamp_utc", "bucket_start_utc", "open", "high", "low", "close", "volume", "trade_count"])


def test_sparse_bucket_aggregates_observed_bars_without_filling_minutes():
    day = date(2026, 11, 13)
    opened_et, closed_et = session_for_date(day)
    opened = pd.Timestamp(opened_et).tz_convert("UTC")
    _, sessions = _session_map({day})
    frame = minute_rows(opened, [0, 2, 4])
    result = reconstruct(frame, sessions).iloc[0]
    assert result.observed_minute_count == 3
    assert result.missing_minutes == [
        (opened + pd.Timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        (opened + pd.Timedelta(minutes=3)).isoformat().replace("+00:00", "Z"),
    ]
    assert (result.open, result.high, result.low, result.close) == (100.0, 101.2, 99.8, 101.0)
    assert result.volume == 36
    assert result.trade_count == 9
    assert result.timestamp_utc == opened
    assert opened + pd.Timedelta(minutes=5) <= pd.Timestamp(closed_et).tz_convert("UTC")


def test_one_observed_minute_is_counted_and_not_duplicated_into_five():
    day = date(2026, 11, 13)
    opened_et, _ = session_for_date(day)
    opened = pd.Timestamp(opened_et).tz_convert("UTC")
    _, sessions = _session_map({day})
    result = reconstruct(minute_rows(opened, [2]), sessions).iloc[0]
    assert result.observed_minute_count == 1
    assert result.volume == 12
    assert result.trade_count == 3
    assert result.missing_minutes == [
        (opened + pd.Timedelta(minutes=i)).isoformat().replace("+00:00", "Z")
        for i in (0, 1, 3, 4)
    ]


def test_empty_bucket_is_omitted_not_fabricated():
    empty = pd.DataFrame(columns=["timestamp_utc", "bucket_start_utc", "open", "high", "low", "close", "volume", "trade_count"])
    assert reconstruct(empty, {}).empty


def test_early_close_final_bucket_stays_inside_session():
    day = date(2026, 11, 27)
    opened_et, closed_et = session_for_date(day)
    opened = pd.Timestamp(opened_et).tz_convert("UTC")
    closed = pd.Timestamp(closed_et).tz_convert("UTC")
    assert closed - opened == pd.Timedelta(hours=3, minutes=30)
    last_start = closed - pd.Timedelta(minutes=5)
    _, sessions = _session_map({day})
    frame = pd.DataFrame([
        {"timestamp_utc": last_start, "bucket_start_utc": last_start,
         "open": 120.0, "high": 120.5, "low": 119.8, "close": 120.2,
         "volume": 10, "trade_count": 2},
        {"timestamp_utc": last_start + pd.Timedelta(minutes=4), "bucket_start_utc": last_start,
         "open": 120.2, "high": 121.0, "low": 120.0, "close": 120.8,
         "volume": 8, "trade_count": 1},
    ])
    result = reconstruct(frame, sessions).iloc[0]
    assert result.timestamp_utc == last_start
    assert result.timestamp_utc + pd.Timedelta(minutes=5) == closed
    assert result.observed_minute_count == 2


def test_dst_transition_preserves_new_york_open_and_utc_offset_change():
    before = date(2026, 3, 6)
    after = date(2026, 3, 9)
    before_open, _ = session_for_date(before)
    after_open, _ = session_for_date(after)
    assert before_open.hour == after_open.hour == 9
    assert before_open.minute == after_open.minute == 30
    assert pd.Timestamp(before_open).tz_convert("UTC").hour == 14
    assert pd.Timestamp(after_open).tz_convert("UTC").hour == 13


def test_historical_feature_builder_keeps_order_resets_on_gap_and_is_causal():
    day = date(2026, 11, 13)
    opened_et, _ = session_for_date(day)
    local_open = pd.Timestamp(opened_et).tz_localize(None)
    times = [local_open + pd.Timedelta(minutes=5 * i) for i in range(32) if i != 24]
    close = np.array([100 + i * .03 for i in range(len(times))], dtype=float)
    data = pd.DataFrame({"datetime": times, "open": close - .02, "high": close + .08,
                         "low": close - .08, "close": close})
    full = calculate_b_features(data, FEATURES)
    assert list(full.columns) == ["datetime", *FEATURES]
    assert len(FEATURES) == 24
    selected = times[20]
    row, missing = ready_feature_row(full, pd.Timestamp(selected))
    assert row is not None and not missing
    # A feature after a genuine 5-minute timestamp gap starts a new segment.
    post_gap = times[24]
    post_row, post_missing = ready_feature_row(full, pd.Timestamp(post_gap))
    assert post_row is None
    assert "return_1" in post_missing
    # Appending a future extreme price cannot alter already-calculated rows.
    extended = pd.concat([data, pd.DataFrame([{"datetime": times[-1] + pd.Timedelta(minutes=5),
        "open": 1000., "high": 2000., "low": 1., "close": 1500.}])], ignore_index=True)
    later = calculate_b_features(extended, FEATURES)
    baseline = full.set_index("datetime").loc[:selected, FEATURES]
    extended_before = later.set_index("datetime").loc[:selected, FEATURES]
    pd.testing.assert_frame_equal(baseline, extended_before)


def test_actual_cached_train_validation_parity_results_are_exact_and_test_free():
    result = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert result["provenance"]["permitted_dates_new_york"] == ["2025-11-13", "2026-09-24"]
    assert result["provenance"]["test_data_accessed"] is False
    assert result["counts"]["matched_buckets"] == 16059
    assert result["parity"]["exact_ohlc_matches"] == 16059
    assert result["parity"]["exact_volume_matches"] == 16059
    assert result["parity"]["exact_trade_count_matches"] == 16059
    assert result["feature_compatibility"]["common_ready_rows_all_features_exact"] == 6792

