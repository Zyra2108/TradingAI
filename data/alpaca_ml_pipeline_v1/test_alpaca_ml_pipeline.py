from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

FILE = Path(__file__).with_name("build_dataset.py")
spec = importlib.util.spec_from_file_location("alpaca_dataset_builder_test", FILE)
builder = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(builder)


def test_provenance_names_only_approved_alpaca_sources():
    assert builder.ONE_YEAR.name.endswith("20261112.csv")
    assert builder.NATIVE.name == "raw_ecl_iex_5min_train_validation.csv"
    assert builder.BACKFILL.name == "ecl_iex_5min_20260925_20261112.csv"
    assert all("test.csv" not in str(p).lower() for p in [builder.ONE_YEAR, builder.NATIVE, builder.BACKFILL])


def test_source_row_validation_flags_invalid_ohlc_and_zero_volume():
    frame = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2026-11-02T14:30:00Z", "2026-11-02T14:35:00Z", "2026-11-02T14:40:00Z"], utc=True),
        "open": [10.0, 10.0, 10.0], "high": [11.0, 9.0, 11.0], "low": [9.0, 9.5, 9.0],
        "close": [10.5, 10.0, 10.5], "volume": [5, 5, 0], "trade_count": [1, 1, 0],
        "source_id": ["test"] * 3, "source_file": ["synthetic"] * 3,
    })
    valid, rejected = builder._validate_source_rows(frame, "synthetic")
    assert len(valid) == 1
    assert set(rejected.reject_reason) == {"invalid_ohlc_relationship", "zero_volume"}


def test_session_filter_accepts_final_regular_candle_and_early_close():
    times = pd.to_datetime(["2026-11-02T20:55:00Z", "2026-11-27T17:55:00Z", "2026-11-27T18:00:00Z"], utc=True)
    # Regular 15:55 ET close and the 12:55 ET final candle on Black Friday.
    frame = pd.DataFrame({"timestamp_utc": times, "datetime_ny": times.tz_convert("America/New_York").tz_localize(None)})
    accepted, rejected, _ = builder._session_filter(frame)
    assert accepted.datetime_ny.tolist() == [pd.Timestamp("2026-11-02 15:55"), pd.Timestamp("2026-11-27 12:55")]
    assert len(rejected) == 1


def test_missing_candle_is_not_filled_or_inserted():
    times = pd.to_datetime(["2026-11-02T14:30:00Z", "2026-11-02T14:40:00Z"], utc=True)
    frame = pd.DataFrame({"timestamp_utc": times, "datetime_ny": times.tz_convert("America/New_York").tz_localize(None)})
    accepted, _, _ = builder._session_filter(frame)
    assert len(accepted) == 2
    assert accepted.timestamp_utc.tolist() == list(times)


def test_target_labeler_rejects_gap_and_accepts_exact_endpoint():
    import sys
    sys.path.insert(0, str(builder.ROOT))
    label_path = builder.ROOT / "src/11_create_features.py"
    label_spec = importlib.util.spec_from_file_location("canonical_target_labeler_test", label_path)
    labeler = importlib.util.module_from_spec(label_spec)
    assert label_spec and label_spec.loader
    label_spec.loader.exec_module(labeler)
    times = pd.to_datetime(["2026-11-02 09:30", "2026-11-02 09:35", "2026-11-02 09:40", "2026-11-02 09:50", "2026-11-02 09:55", "2026-11-02 10:00", "2026-11-02 10:05"])
    closes = [100.0, 100.1, 100.2, 100.3, 100.5, 101.0, 101.0]
    bars = pd.DataFrame({"datetime": times, "open": closes, "high": np.array(closes) + .1, "low": np.array(closes) - .1, "close": closes})
    segmented = labeler.create_segments(bars)
    labeled = pd.concat([labeler.create_labels(g) for _, g in segmented.groupby("segment_id", sort=False)], ignore_index=True)
    assert pd.isna(labeled.loc[0, "target"])  # no cross-gap target
    row = labeled.loc[labeled.datetime.eq(pd.Timestamp("2026-11-02 09:50"))].iloc[0]
    assert row.future_datetime == pd.Timestamp("2026-11-02 10:05")
    assert row.target == "UP"

