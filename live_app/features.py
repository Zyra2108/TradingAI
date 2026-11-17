"""Canonical B features built by calling the original historical functions."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_base = _load("ecl_historical_base_features", ROOT / "src" / "11_create_features.py")
_time = _load("ecl_historical_time_features", ROOT / "src" / "17_add_time_features.py")


def calculate_b_features(ohlc: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """Apply the historical feature functions to ordered NY-local OHLC bars.

    The caller supplies all bars from a session (including enough warmup) and
    only completed candles. Segments reset at sessions and missing 5-minute
    gaps exactly as in src/11_create_features.py. Time features use the same
    NYSE-calendar functions as src/17_add_time_features.py.
    """
    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in ohlc.columns]
    if missing:
        raise ValueError(f"Missing OHLC fields: {missing}")
    frame = ohlc[required].copy()
    frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
    if frame.datetime.dt.tz is not None:
        raise ValueError("Provider timestamps must be naive America/New_York local time")
    if frame.datetime.duplicated().any() or not frame.datetime.is_monotonic_increasing:
        raise ValueError("OHLC timestamps must be unique and increasing")
    for col in ["open", "high", "low", "close"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    values = frame[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("OHLC contains non-finite prices")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid candle high")
    if (frame.low > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid candle low")
    if not frame.datetime.diff().dropna().map(lambda d: d.total_seconds() % 300 == 0).all():
        raise ValueError("Candle timestamps are not aligned to 5-minute boundaries")

    segmented = _base.create_segments(frame)
    outputs = []
    for _, group in segmented.groupby("segment_id", sort=False):
        outputs.append(_base.calculate_features(group))
    featured = pd.concat(outputs, ignore_index=True)
    schedule = _time.get_nyse_schedule(featured)
    featured = _time.add_time_features(featured, schedule)
    if "close" not in feature_order or set(feature_order) - set(featured.columns):
        raise ValueError("Frozen B feature contract does not match historical builder output")
    # The frozen artifact defines the order; no runtime feature inference/reordering.
    return featured[["datetime", *feature_order]].copy()


def ready_feature_row(feature_frame: pd.DataFrame, candle_time: pd.Timestamp) -> tuple[pd.Series | None, list[str]]:
    rows = feature_frame.loc[feature_frame.datetime.eq(candle_time)]
    if len(rows) != 1:
        return None, ["completed candle feature row unavailable"]
    row = rows.iloc[0]
    missing = [c for c in feature_frame.columns if c != "datetime" and pd.isna(row[c])]
    return (None, missing) if missing else (row, [])
