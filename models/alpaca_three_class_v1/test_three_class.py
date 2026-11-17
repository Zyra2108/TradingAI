from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("ecl_three_class_research_v1", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_strict_target_thresholds_and_classes():
    values = np.array([-0.10001, -0.10, 0.0, 0.10, 0.10001])
    assert module.target_from_return(values).tolist() == ["DOWN", "NEUTRAL", "NEUTRAL", "NEUTRAL", "UP"]


def test_ridge_three_class_threshold_and_zero_boundary_are_distinct():
    values = np.array([-0.2, -0.05, 0.0, 0.05, 0.2])
    assert module.ridge_to_classes(values, 0.1).tolist() == ["DOWN", "NEUTRAL", "NEUTRAL", "NEUTRAL", "UP"]
    assert module.ridge_zero_sign(values).tolist() == ["DOWN", "DOWN", "DOWN", "UP", "UP"]
    with pytest.raises(ValueError, match="finite"):
        module.ridge_to_classes([np.nan], 0.1)


def test_ridge_threshold_selection_respects_directional_coverage_guardrail():
    y = np.array(["DOWN"] * 30 + ["NEUTRAL"] * 40 + ["UP"] * 30)
    # Include a spread of useful and ambiguous estimates to make several cutoffs feasible.
    estimates = np.r_[np.full(28, -0.2), np.full(2, -0.04), np.full(40, 0.0), np.full(2, 0.04), np.full(28, 0.2)]
    threshold, rows = module.select_ridge_threshold(y, estimates)
    assert threshold in module.THRESHOLDS
    assert len(rows) == len(module.THRESHOLDS)
    selected = next(r for r in rows if r["threshold_percentage_points"] == threshold)
    assert selected["guardrail"]["passed"]
    assert selected["guardrail"]["predicted_class_share"]["NEUTRAL"] <= selected["guardrail"]["actual_neutral_share"] + 0.05


def test_fold_masks_purge_horizons_and_remain_chronological():
    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-05 15:35", "2026-01-05 15:40", "2026-01-06 09:30"]),
        "future_datetime": pd.to_datetime(["2026-01-05 15:50", "2026-01-06 09:55", "2026-01-06 09:45"]),
    })
    train, valid, first = module.fold_masks(frame, "2026-01-05", "2026-01-06", "2026-01-06")
    assert train.tolist() == [True, False, False]
    assert valid.tolist() == [False, False, True]
    assert frame.loc[train, "future_datetime"].max() < first


def test_target_paths_require_exact_endpoint_and_continuous_same_session_candles():
    start = pd.Timestamp("2026-01-05 09:30")
    source = pd.date_range(start, periods=5, freq="5min")
    module.verify_continuous_target_paths(pd.DatetimeIndex([source[0]]), pd.DatetimeIndex([source[3]]), source, "synthetic")
    with pytest.raises(ValueError, match=r"missing five-minute candle at \+10m"):
        module.verify_continuous_target_paths(pd.DatetimeIndex([source[0]]), pd.DatetimeIndex([source[3]]), source.delete(2), "synthetic")
    session_edge = pd.Timestamp("2026-01-05 23:55")
    next_session = session_edge + pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="crosses a New York session date"):
        module.verify_continuous_target_paths(pd.DatetimeIndex([session_edge]), pd.DatetimeIndex([next_session]), pd.date_range(session_edge, periods=4, freq="5min"), "synthetic")
    with pytest.raises(ValueError, match=r"exactly \+15 minutes"):
        module.verify_continuous_target_paths(pd.DatetimeIndex([source[0]]), pd.DatetimeIndex([source[2]]), source, "synthetic")


def test_class_mapping_metrics_and_probability_validation():
    actual = np.array(["DOWN", "NEUTRAL", "UP"])
    pred = np.array(["DOWN", "UP", "UP"])
    prob = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.1, 0.1, 0.8]])
    result = module.class_metrics(actual, pred, prob)
    assert result["confusion_matrix_actual_rows_predicted_columns"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert result["prediction_distribution"] == {"DOWN": 1, "NEUTRAL": 0, "UP": 2}
    with pytest.raises(ValueError, match="probabilities invalid"):
        module.class_metrics(actual, pred, np.array([[2, 0, 0]] * 3))


def test_inputs_and_outputs_are_isolated_and_allowlisted(tmp_path):
    assert module.TRAIN_CSV.resolve() in module.APPROVED_CSVS
    assert module.VALIDATION_CSV.resolve() in module.APPROVED_CSVS
    assert tmp_path.resolve() not in module.APPROVED_CSVS
    with pytest.raises(ValueError, match="Unapproved experiment input"):
        module.approved_read_csv(tmp_path / "unapproved.csv")
    assert module.REPORT_DIR == module.ROOT / "reports" / "alpaca_three_class_v1"
    assert module.MODEL_DIR == module.ROOT / "models" / "alpaca_three_class_v1"


def test_catboost_not_declared_as_fallback_model():
    assert "catboost_multiclass" not in module.make_pipelines()
    assert module.xgboost.__version__ == "3.4.1"


def test_xgboost_multiclass_probability_and_class_order_contract():
    rng = np.random.default_rng(17)
    x = pd.DataFrame(rng.normal(size=(60, 24)), columns=[f"f{i}" for i in range(24)])
    y = np.array(["DOWN", "NEUTRAL", "UP"] * 20)
    returns = np.tile([-0.2, 0.0, 0.2], 20)
    fitted = module.fit_model("xgboost_multiclass", x, y, returns)
    prediction, probabilities = module.predict_model("xgboost_multiclass", fitted, x.iloc[:9])
    assert len(prediction) == 9
    assert probabilities is not None and probabilities.shape == (9, 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert set(prediction).issubset(module.CLASS_ORDER)

