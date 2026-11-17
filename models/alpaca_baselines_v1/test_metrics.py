from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

FILE = Path(__file__).with_name("train_compare.py")
spec = importlib.util.spec_from_file_location("alpaca_baseline_train_test", FILE)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_probability_class_order_is_down_neutral_up():
    class Estimator:
        classes_ = np.array(["UP", "DOWN", "NEUTRAL"])

        @staticmethod
        def predict_proba(_):
            return np.array([[0.7, 0.2, 0.1]])

    import pandas as pd
    aligned = module._aligned_probabilities(Estimator(), pd.DataFrame({"x": [1]}))
    assert np.allclose(aligned, [[0.2, 0.1, 0.7]])


def test_metrics_confusion_orientation_and_directional_coverage():
    y = np.array(["DOWN", "NEUTRAL", "UP", "NEUTRAL"])
    p = np.array(["DOWN", "UP", "NEUTRAL", "NEUTRAL"])
    probs = np.array([[.8,.1,.1],[.1,.1,.8],[.1,.8,.1],[.1,.8,.1]])
    result = module.evaluate(y, p, probs)
    assert result["confusion_matrix_rows_actual_columns_predicted"] == [[1,0,0],[0,1,1],[0,1,0]]
    assert result["directional_prediction_coverage_all_rows"] == .5
    assert np.isfinite(result["log_loss"])


def test_selection_rule_uses_validation_macro_f1_first():
    assert "macro-F1 highest" in module.CONFIG["selection_rule"]
    assert "test data" in module.CONFIG["selection_rule"]
