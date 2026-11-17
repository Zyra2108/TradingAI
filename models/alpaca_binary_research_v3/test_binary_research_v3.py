from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).with_name("run_binary_research.py")
spec = importlib.util.spec_from_file_location("ecl_binary_research_v3_test_module", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_strict_binary_thresholds_keep_boundary_neutral():
    labels = module.binary_label_from_return(np.array([0.100001, 0.10, 0.0, -0.10, -0.100001]))
    assert labels.tolist() == ["UP", "NEUTRAL", "NEUTRAL", "NEUTRAL", "DOWN"]


def test_training_and_validation_inputs_are_pretest_and_use_same_feature_order():
    schema = __import__("json").loads((module.DATA_DIR / "reproducibility_manifest.json").read_text())
    features = schema["features"]
    train = module.load_dataset("train", features)
    validation = module.load_dataset("validation", features)
    assert len(features) == 24
    assert list(train.columns[4:]) == features
    assert list(validation.columns[4:]) == features
    assert train.datetime.max() < validation.datetime.min()
    assert validation.datetime.max().normalize() <= module.MAX_DATE
    assert set(train.target.unique()) == {"DOWN", "NEUTRAL", "UP"}


def test_expanding_folds_are_chronological_disjoint_and_purged():
    days = pd.bdate_range("2026-01-05", periods=16)
    rows = []
    for day in days:
        for minute in [30, 35]:
            dt = day.replace(hour=9, minute=minute)
            rows.append({"datetime": dt, "future_datetime": dt + pd.Timedelta(minutes=15)})
    frame = pd.DataFrame(rows)
    folds = module.make_folds(frame, initial_fraction=0.5, n_folds=4)
    seen_valid = []
    for fold in folds:
        tr = frame.loc[fold["train_index"]]
        va = frame.loc[fold["validation_index"]]
        assert set(tr.index).isdisjoint(set(va.index))
        assert tr.datetime.max().date() < va.datetime.min().date()
        assert (tr.future_datetime < va.datetime.min().normalize()).all()
        seen_valid.extend(va.datetime.dt.date.unique().tolist())
    assert len(set(seen_valid)) == len(seen_valid)


def test_binary_metrics_confusion_orientation():
    actual = np.array(["DOWN", "DOWN", "UP", "UP"])
    predicted = np.array(["DOWN", "UP", "DOWN", "UP"])
    result = module.binary_metrics(actual, predicted)
    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["confusion_matrix_rows_actual_columns_predicted"] == [[1, 1], [1, 1]]
    assert result["class_metrics"]["DOWN"]["support"] == 2
    assert result["class_metrics"]["UP"]["support"] == 2


def test_low_confidence_warns_without_neutral_or_abstention():
    probs = np.array([[0.55, 0.45], [0.20, 0.80], [0.50, 0.50]])
    prediction, confidence, low = module.binary_prediction_output(probs, 0.60)
    assert prediction.tolist() == ["DOWN", "UP", "DOWN"]
    assert np.allclose(confidence, [0.55, 0.80, 0.50])
    assert low.tolist() == [True, False, True]
    assert set(prediction).issubset({"DOWN", "UP"})


def test_xgboost_is_available_and_mandatory():
    import xgboost
    assert xgboost.__version__
    assert "xgboost" in module.configurations()
