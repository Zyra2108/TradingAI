"""Close-to-close, session-continuous target generation for permitted splits only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "alpaca_rebuild_v1"
MATRIX_PATH = Path(__file__).resolve().parent / "experiment_matrix.json"
CLASSES = ["DOWN", "NO_SIGNAL", "UP"]
DIRECTIONS = ["DOWN", "UP"]
FEATURES = [
    "close", "return_1", "return_3", "return_6", "return_12",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20", "ema_ratio_5",
    "ema_ratio_10", "ema_ratio_20", "rsi_14", "volatility_5",
    "volatility_10", "volatility_20", "candle_body", "candle_range",
    "upper_wick", "lower_wick", "minutes_since_open", "minutes_until_close",
    "day_of_week", "time_sin", "time_cos",
]
FORBIDDEN_FEATURES = {"datetime", "future_datetime", "future_return", "target", "label",
                      "target_end", "threshold_used", "volume", "open", "high", "low"}


def matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def load_allowed_split(name: str) -> pd.DataFrame:
    if name not in {"train.csv", "validation.csv"}:
        raise ValueError("Only existing train.csv and validation.csv may be opened")
    usecols = list(dict.fromkeys(["datetime", *FEATURES]))
    frame = pd.read_csv(DATA / name, usecols=usecols)
    frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")
    frame["day_of_week"] = frame.day_of_week.astype("int64")
    if not frame.datetime.is_monotonic_increasing or frame.datetime.duplicated().any():
        raise ValueError(f"{name} timestamps must be unique and chronological")
    if not np.isfinite(frame[FEATURES].to_numpy(float)).all():
        raise ValueError(f"{name} contains nonfinite model features")
    return frame


def threshold_pct(row: pd.Series, target: dict) -> float:
    horizon = int(target["horizon_minutes"])
    bars = horizon // 5
    if horizon <= 0 or horizon % 5:
        raise ValueError("Horizons must be positive multiples of five minutes")
    method = target["method"]
    if method == "fixed_pct":
        value = float(target["threshold_pct"])
    elif method == "fixed_pct_sqrt_horizon":
        value = float(target["anchor_threshold_pct"]) * np.sqrt(horizon / float(target["anchor_horizon_minutes"]))
    elif method == "vol_scaled":
        value = float(target["multiplier"]) * float(row.volatility_20) * np.sqrt(bars)
    elif method == "no_trade_floor":
        vol_move = float(target["volatility_multiplier"]) * float(row.volatility_20) * np.sqrt(bars)
        value = max(float(target["fixed_floor_pct"]), vol_move)
    else:
        raise ValueError(f"Unknown target method: {method}")
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Target threshold must be finite and positive")
    return value


def generate_target(frame: pd.DataFrame, target: dict) -> pd.DataFrame:
    """Create labels using exact future timestamps, same-date continuity, and current-row vol only."""
    required = {"datetime", *FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing input fields: {sorted(required-set(frame.columns))}")
    if set(FEATURES) & FORBIDDEN_FEATURES:
        raise AssertionError("Feature list contains target/metadata/raw volume fields")
    source = frame.sort_values("datetime").reset_index(drop=True)
    times = source.datetime
    if times.duplicated().any():
        raise ValueError("Duplicate candle timestamps are not allowed")
    close_by_time = dict(zip(source.datetime, source.close.astype(float)))
    vol_by_time = dict(zip(source.datetime, source.volatility_20.astype(float)))
    horizon = int(target["horizon_minutes"])
    if horizon <= 0 or horizon % 5:
        raise ValueError("Horizon must be a positive multiple of five minutes")
    bars = horizon // 5
    result = source.copy()
    future_times, future_returns, thresholds, labels, valid_flags = [], [], [], [], []
    for row in source.itertuples(index=False):
        t = row.datetime
        end = t + pd.Timedelta(minutes=horizon)
        expected = [t + pd.Timedelta(minutes=5 * j) for j in range(1, bars + 1)]
        # No overnight, session, missing-candle, or early-close crossing is accepted.
        same_date = end.date() == t.date()
        continuous = same_date and all(ts in close_by_time for ts in expected)
        available = bool(continuous and end in close_by_time)
        future_times.append(end)
        valid_flags.append(available)
        if not available:
            future_returns.append(np.nan)
            thresholds.append(np.nan)
            labels.append("UNAVAILABLE")
            continue
        # Threshold is computed from the decision row only, never future-row volatility.
        decision = pd.Series({"volatility_20": vol_by_time[t]})
        thr = threshold_pct(decision, target)
        ret = (close_by_time[end] / float(row.close) - 1.0) * 100.0
        future_returns.append(ret)
        thresholds.append(thr)
        labels.append("UP" if ret > thr else "DOWN" if ret < -thr else "NO_SIGNAL")
    result["target_end"] = future_times
    result["future_return_pct"] = future_returns
    result["threshold_pct"] = thresholds
    result["label"] = labels
    result["target_available"] = valid_flags
    return result


def eligible_mask(frame: pd.DataFrame) -> np.ndarray:
    return frame.label.isin(DIRECTIONS).to_numpy()

