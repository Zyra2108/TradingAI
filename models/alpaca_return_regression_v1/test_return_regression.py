from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

SCRIPT = Path(__file__).with_name("run_return_regression.py")
spec = importlib.util.spec_from_file_location("ecl_return_regression_v1_test_module", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_existing_return_units_and_strict_target_boundaries():
    values = module.return_pct(np.array([100.0, 100.0, 100.0]), np.array([100.1001, 100.1, 99.8999]))
    assert values[0] > 0.10
    assert values[1] == pytest.approx(0.10)
    assert module.target_from_return(values).tolist() == ["UP", "NEUTRAL", "DOWN"]


def test_exact_endpoint_and_continuous_session_are_required():
    times = pd.date_range("2026-01-05 09:30", periods=4, freq="5min")
    closes = pd.Series([100.0, 100.1, 100.2, 100.3], index=times)
    frame = pd.DataFrame({
        "datetime": [times[0]], "future_datetime": [times[3]],
        "future_return": [(100.3 / 100.0 - 1) * 100], "target": ["UP"],
    })
    module.verify_target_alignment(frame, closes, set(times), "synthetic")
    broken = frame.copy()
    broken["future_datetime"] = broken.datetime + pd.Timedelta(minutes=10)
    with pytest.raises(ValueError, match="exactly 15 minutes"):
        module.verify_target_alignment(broken, closes, set(times), "synthetic")


def test_missing_intervening_candle_rejects_target_path():
    base = pd.Timestamp("2026-01-05 09:30")
    kept = [base, base + pd.Timedelta(minutes=5), base + pd.Timedelta(minutes=15)]
    closes = pd.Series([100.0, 100.1, 100.2], index=pd.DatetimeIndex(kept))
    frame = pd.DataFrame({
        "datetime": [base], "future_datetime": [base + pd.Timedelta(minutes=15)],
        "future_return": [0.2], "target": ["UP"],
    })
    with pytest.raises(ValueError, match=r"missing five-minute interval at \+10m"):
        module.verify_target_alignment(frame, closes, set(kept), "synthetic")


def test_session_boundary_rejects_target():
    decision = pd.Timestamp("2026-01-05 23:55")
    endpoint = pd.Timestamp("2026-01-06 00:10")
    closes = pd.Series([100.0, 101.0], index=[decision, endpoint])
    frame = pd.DataFrame({"datetime": [decision], "future_datetime": [endpoint], "future_return": [1.0], "target": ["UP"]})
    with pytest.raises(ValueError, match="crosses a New York session date"):
        module.verify_target_alignment(frame, closes, {decision, endpoint}, "synthetic")


def test_prediction_sign_conversion_uses_fixed_zero_boundary():
    actual = module.direction_from_return(np.array([-0.01, 0.0, 0.02]))
    assert actual.tolist() == ["DOWN", "DOWN", "UP"]
    with pytest.raises(ValueError, match="finite"):
        module.direction_from_return(np.array([np.nan]))


def test_fold_masks_are_chronological_and_purge_forward_endpoint():
    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-05 15:35", "2026-01-05 15:40", "2026-01-06 09:30"]),
        "future_datetime": pd.to_datetime(["2026-01-05 15:50", "2026-01-06 09:55", "2026-01-06 09:45"]),
    })
    train_mask, val_mask, first_val = module.fold_masks(frame, "2026-01-05", "2026-01-06", "2026-01-06")
    assert train_mask.tolist() == [True, False, False]
    assert val_mask.tolist() == [False, False, True]
    assert frame.loc[train_mask, "future_datetime"].max() < first_val


def test_directional_metric_confusion_orientation_and_distribution():
    y = np.array(["DOWN", "DOWN", "UP", "UP"])
    p = np.array(["DOWN", "UP", "DOWN", "DOWN"])
    m = module.class_metrics(y, p)
    assert m["confusion_matrix_actual_rows_predicted_columns_down_up"] == [[1, 1], [2, 0]]
    assert m["prediction_down"] == 3
    assert m["prediction_up"] == 1
    assert m["up_recall"] == 0.0


def test_only_approved_inputs_are_accepted_and_test_file_is_not_referenced():
    assert module.TRAIN_CSV.resolve() in module.approved_paths()
    with pytest.raises(ValueError, match="Unapproved research input path"):
        module.approved_read_csv(module.MODEL_DIR / "not_approved.csv")


def test_xgboost_regressor_pipeline_save_load_consistency(tmp_path):
    x = pd.DataFrame(np.random.default_rng(4).normal(size=(20, 24)), columns=[f"f{i}" for i in range(24)])
    y = np.linspace(-0.2, 0.2, len(x))
    pipeline = module.make_model_pipelines()["ridge"]
    pipeline.fit(x, y)
    before = pipeline.predict(x[:4])
    path = tmp_path / "ridge.joblib"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(before, loaded.predict(x[:4]), atol=1e-12)


def test_scaling_is_fit_from_train_input_only():
    train_x = pd.DataFrame({f"f{i}": np.arange(5, dtype=float) + i for i in range(24)})
    validation_x = pd.DataFrame({f"f{i}": np.full(2, 1e6 + i) for i in range(24)})
    pipeline = module.make_model_pipelines()["ridge"]
    pipeline.fit(train_x, np.array([-0.2, -0.1, 0.0, 0.1, 0.2]))
    learned_means = pipeline.named_steps["scaler"].mean_
    expected_means = train_x.mean(axis=0).to_numpy()
    np.testing.assert_allclose(learned_means, expected_means)
    assert not np.allclose(learned_means, pd.concat([train_x, validation_x]).mean(axis=0).to_numpy())


def test_approved_dataset_contract_and_raw_return_alignment():
    _, train, validation, folds, _, _, extra = module.load_inputs()
    assert len(extra["features"]) == 24
    assert len(folds) == 4
    assert train.target.isin(["DOWN", "NEUTRAL", "UP"]).all()
    assert validation.target.isin(["DOWN", "NEUTRAL", "UP"]).all()


def test_estimators_include_mandatory_xgboost_without_substitution():
    import xgboost
    pipelines = module.make_model_pipelines()
    assert xgboost.__version__ == "3.4.1"
    assert "xgboost_regressor" in pipelines
    assert pipelines["xgboost_regressor"].named_steps["model"].objective == "reg:squarederror"
