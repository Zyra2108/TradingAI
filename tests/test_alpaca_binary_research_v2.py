"""Causality, fold-purge, threshold, and metric contract tests for binary v2."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).parents[1] / "models" / "alpaca_binary_research_v2" / "train_binary_v2.py"
spec = importlib.util.spec_from_file_location("alpaca_binary_v2", MODULE_PATH)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def _features(n=1):
    data = {col: np.zeros(n, dtype=float) for col in v2.BASE_FEATURES}
    data["day_of_week"] = np.zeros(n, dtype=int)
    return pd.DataFrame(data)


def test_derived_features_are_row_local_and_use_expected_formulas():
    frame = _features(2)
    frame.loc[0, ["return_1", "return_3", "return_6", "return_12"]] = [.02, .01, -.01, -.02]
    frame.loc[0, ["sma_ratio_5", "sma_ratio_10", "sma_ratio_20", "ema_ratio_5", "ema_ratio_10", "ema_ratio_20"]] = [.01, .02, .03, .01, .02, .03]
    frame.loc[0, ["volatility_5", "volatility_10", "volatility_20"]] = [.2, .3, .4]
    frame.loc[0, ["candle_body", "candle_range", "upper_wick", "lower_wick"]] = [.01, .04, .02, .01]
    frame.loc[0, ["minutes_since_open", "minutes_until_close"]] = [60, 330]
    result = v2.engineer_causal_features(frame)
    assert np.isclose(result.loc[0, "momentum_sign_agreement"], 0.)
    assert np.isclose(result.loc[0, "momentum_mean"], 0.)
    assert np.isclose(result.loc[0, "momentum_short_long_gap"], .04)
    assert np.isclose(result.loc[0, "volatility_5_to_20"], .5)
    assert np.isclose(result.loc[0, "body_to_range"], .25)
    assert np.isclose(result.loc[0, "wick_imbalance_to_range"], .25)
    assert np.isclose(result.loc[0, "total_wick_to_range"], .75)
    assert np.isclose(result.loc[0, "session_position"], 60/390)


def test_future_row_changes_cannot_change_prior_features():
    frame = _features(2)
    first = v2.engineer_causal_features(frame).loc[0, v2.DERIVED_FEATURES]
    frame.loc[1, v2.BASE_FEATURES] = np.arange(len(v2.BASE_FEATURES)) * 1000.
    later = v2.engineer_causal_features(frame).loc[0, v2.DERIVED_FEATURES]
    pd.testing.assert_series_equal(first, later)


def test_only_train_and_validation_file_names_are_openable():
    try:
        v2.load_allowed_split("test.csv")
    except ValueError as exc:
        assert "Only the predeclared training and validation" in str(exc)
    else:
        raise AssertionError("Disallowed split name must be rejected before file access")


def test_model_feature_sets_exclude_metadata_and_target_leakage_fields():
    forbidden = {"datetime", "future_datetime", "future_return", "target", "volume",
                 "open", "high", "low", "close.1"}
    for features in v2.FEATURE_SETS.values():
        assert not (set(features) & forbidden)
    assert v2.FEATURE_SETS["original_24"] == v2.BASE_FEATURES
    assert len(v2.DERIVED_FEATURES) == 12


def test_expanding_folds_are_chronological_and_purge_future_endpoints():
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    frame = pd.DataFrame({"datetime": pd.to_datetime([d + pd.Timedelta(hours=10) for d in dates]),
                          "future_datetime": pd.to_datetime([d + pd.Timedelta(hours=10, minutes=15) for d in dates]),
                          "target": ["UP", "DOWN"] * 10})
    folds = v2.chronological_folds(frame, 3)
    assert all(folds[i][1] < folds[i+1][1] for i in range(2))
    eval_start = folds[1][1]
    fold_train = v2.fold_fit_rows(frame, eval_start)
    assert (fold_train.datetime < eval_start).all()
    assert (fold_train.future_datetime < eval_start).all()


def test_score_quantile_threshold_and_tie_overcoverage():
    confidence = np.array([.51, .6, .7, .7, .9])
    threshold = v2.confidence_threshold(confidence, .4)
    assert threshold == .7
    assert int((confidence >= threshold).sum()) == 3


def test_metric_calculation_separates_directional_and_operational_precision():
    y = np.array(["UP", "DOWN", "NEUTRAL", "UP"])
    pred = np.array(["UP", "UP", "DOWN", "DOWN"])
    metrics, output = v2.selective_metrics(y, pred, np.array([.95, .70, .85, .55]), .8)
    assert output.tolist() == ["UP", "NO_SIGNAL", "DOWN", "NO_SIGNAL"]
    assert metrics["emitted_predictions"] == 2
    assert metrics["no_signal_count"] == 2
    assert metrics["eligible_directional_rows"] == 3
    assert metrics["eligible_directional_metrics_on_emissions"]["up_precision"] == 1.
    assert metrics["eligible_directional_metrics_on_emissions"]["down_precision"] == 0.
    assert metrics["macro_precision_operational_neutral_false"] == .5


def test_prior_binary_report_matches_independent_saved_prediction_recalculation():
    rows, summary = v2.audit_previous_v1()
    assert summary["report_metrics_consistent_with_saved_predictions"] is True
    assert len(rows) == 10  # raw metrics plus four coverage policies for each model

