"""Target generation and leakage-protection tests; synthetic candles only."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).parents[1] / "models" / "alpaca_target_research_v1" / "target_generation.py"
spec = importlib.util.spec_from_file_location("target_generation", MODULE_PATH)
target_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target_module)


def _frame(times, closes, vols=None):
    n = len(times)
    data = {name: np.zeros(n, dtype=float) for name in target_module.FEATURES}
    data["datetime"] = pd.to_datetime(times)
    data["close"] = np.asarray(closes, dtype=float)
    data["volatility_20"] = np.asarray(vols if vols is not None else [0.1]*n, dtype=float)
    data["day_of_week"] = pd.to_datetime(times).day_of_week
    data["minutes_since_open"] = np.arange(n)*5+5
    data["minutes_until_close"] = 390-data["minutes_since_open"]
    return pd.DataFrame(data)


def _definition(method="fixed_pct", threshold=.1, horizon=10, **kwargs):
    return {"id":"synthetic", "horizon_minutes":horizon, "method":method,
            "threshold_pct":threshold, **kwargs}


def test_exact_future_timestamp_and_close_return_label():
    t = pd.date_range("2026-01-05 09:35", periods=5, freq="5min")
    frame = _frame(t, [100, 100.1, 100.2, 100.3, 100.3])
    result = target_module.generate_target(frame, _definition(horizon=10))
    assert result.loc[0, "target_end"] == t[2]
    assert np.isclose(result.loc[0, "future_return_pct"], .2)
    assert result.loc[0, "label"] == "UP"


def test_missing_intervening_or_endpoint_candle_rejects_target():
    t = pd.to_datetime(["2026-01-05 09:35", "2026-01-05 09:45", "2026-01-05 09:50"])
    frame = _frame(t, [100, 101, 102])
    result = target_module.generate_target(frame, _definition(horizon=10))
    assert result.loc[0, "target_available"] is np.False_ or not bool(result.loc[0, "target_available"])
    assert result.loc[0, "label"] == "UNAVAILABLE"


def test_target_does_not_cross_trading_date_or_session_boundary():
    t = pd.to_datetime(["2026-01-05 15:55", "2026-01-06 09:35", "2026-01-06 09:40"])
    frame = _frame(t, [100, 101, 102])
    result = target_module.generate_target(frame, _definition(horizon=10))
    assert result.loc[0, "label"] == "UNAVAILABLE"
    assert not bool(result.loc[0, "target_available"])


def test_volatility_scaled_threshold_uses_decision_row_only():
    t = pd.date_range("2026-01-05 09:35", periods=4, freq="5min")
    frame = _frame(t, [100, 100.01, 100.02, 100.03], [.10, .10, .10, 9.0])
    target = _definition(method="vol_scaled", horizon=10, multiplier=.5)
    first = target_module.generate_target(frame, target)
    frame.loc[2:, "volatility_20"] = 1000.
    second = target_module.generate_target(frame, target)
    assert np.isclose(first.loc[0, "threshold_pct"], .5*.1*np.sqrt(2))
    assert np.isclose(first.loc[0, "threshold_pct"], second.loc[0, "threshold_pct"])


def test_no_trade_floor_and_sqrt_fixed_threshold_are_predefined():
    row = pd.Series({"volatility_20": .05})
    floor = _definition(method="no_trade_floor", horizon=15, fixed_floor_pct=.1,
                        volatility_multiplier=.5)
    assert np.isclose(target_module.threshold_pct(row, floor), .1)
    scaled = _definition(method="fixed_pct_sqrt_horizon", horizon=60,
                         anchor_threshold_pct=.1, anchor_horizon_minutes=15)
    assert np.isclose(target_module.threshold_pct(row, scaled), .2)


def test_split_loader_rejects_test_file_before_opening_it():
    try:
        target_module.load_allowed_split("test.csv")
    except ValueError as exc:
        assert "Only existing train.csv and validation.csv" in str(exc)
    else:
        raise AssertionError("test split name must be rejected before file access")


def test_model_feature_list_excludes_future_label_and_raw_metadata():
    forbidden = target_module.FORBIDDEN_FEATURES
    assert len(target_module.FEATURES) == 24
    assert not (set(target_module.FEATURES) & forbidden)
