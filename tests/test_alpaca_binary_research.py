"""Focused contract tests for isolated binary research utilities."""
import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "models" / "alpaca_binary_research_v1" / "train_binary_research.py"
spec = importlib.util.spec_from_file_location("alpaca_binary_research", MODULE_PATH)
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def test_only_training_and_validation_names_are_accepted():
    try:
        research.load_split("test.csv")
    except ValueError as exc:
        assert "Only predeclared train/validation" in str(exc)
    else:
        raise AssertionError("test split must be rejected before file access")


def test_feature_order_is_exact_frozen_contract():
    assert len(research.FEATURES) == 24
    assert len(set(research.FEATURES)) == 24
    assert research.CLASSES == ["DOWN", "NEUTRAL", "UP"]


def test_quantile_policy_uses_confidence_and_reports_ties_as_overcoverage():
    confidence = np.array([0.51, 0.60, 0.70, 0.70, 0.90])
    threshold = research.threshold_for(confidence, 0.4)
    assert threshold == 0.70
    assert int((confidence >= threshold).sum()) == 3  # ties can exceed target


def test_selective_metrics_count_no_signal_and_neutral_false_signals():
    actual = np.array(["UP", "DOWN", "NEUTRAL", "UP"])
    prediction = np.array(["UP", "UP", "DOWN", "DOWN"])
    confidence = np.array([0.95, 0.70, 0.85, 0.55])
    metrics, output, _ = research.selective(actual, prediction, confidence, 0.5)
    assert metrics["emitted_predictions"] == 2
    assert metrics["no_signal_count"] == 2
    assert output.tolist() == ["UP", "NO_SIGNAL", "DOWN", "NO_SIGNAL"]
    assert metrics["up_precision_operational"] == 1.0
    assert metrics["down_precision_operational"] == 0.0
    assert metrics["confusion_matrix"][1][0] == 1  # neutral actual emitted as DOWN

