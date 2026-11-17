"""Contract tests for isolated horizon research model evaluation."""
from datetime import date, datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_evaluate as r


def test_metric_definitions_count_neutral_actuals_as_signal_errors():
    actual = np.array(["DOWN", "UP", "NO_SIGNAL"], dtype=object)
    pred = np.array(["DOWN", "DOWN", "UP"], dtype=object)
    m = r.calculate_metrics(actual, pred)
    assert m["signal_coverage_all_eligible"] == 1.0
    assert m["eligible_signal_count"] == 3
    assert m["eligible_signal_accuracy"] == pytest.approx(1 / 3)
    assert m["directional_accuracy_on_actual_directional"] == pytest.approx(0.5)
    assert m["confusion_class_order"] == ["DOWN", "NO_SIGNAL", "UP"]
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [1, 0, 0]]


def test_confidence_policy_abstains_only_on_low_confidence_directional_argmax():
    p = np.array([[.60, .30, .10], [.20, .25, .55], [.10, .70, .20]])
    out = r.apply_confidence_threshold(p, .60)
    assert out.tolist() == ["DOWN", "NO_SIGNAL", "NO_SIGNAL"]


def test_threshold_selection_uses_fixed_floor_and_does_not_choose_tiny_coverage():
    p = np.tile(np.array([[.65, .20, .15], [.10, .25, .65], [.15, .70, .15]]), (20, 1))
    y = np.tile(np.array(["DOWN", "UP", "NO_SIGNAL"], dtype=object), 20)
    choice = r.select_threshold(y, p)
    assert choice["selection_source"] == "pooled chronological training OOF only"
    assert choice["selected"]["signal_coverage"] >= r.MIN_SIGNAL_COVERAGE
    assert choice["selected"]["signals"] >= r.MIN_SIGNAL_COUNT
    assert choice["selected"]["threshold"] in r.THRESHOLD_GRID


def test_folds_are_expanding_and_purge_targets_reaching_validation_boundary():
    rows = []
    start = date(2026, 1, 5)
    for i in range(12):
        day = start + timedelta(days=i)
        stamp = pd.Timestamp(datetime.combine(day, datetime.min.time()) + timedelta(hours=15, minutes=55))
        rows.append({"datetime": stamp, "target_end": stamp + timedelta(minutes=10),
                     "session_date": day.isoformat(), "label": ("DOWN", "NO_SIGNAL", "UP")[i % 3]})
    train = pd.DataFrame(rows)
    # Make the last row before the first held-out date overlap that boundary.
    first_fold_start = start + timedelta(days=6)
    boundary = pd.Timestamp(datetime.combine(first_fold_start, datetime.min.time()) + timedelta(hours=15, minutes=55))
    train.loc[train.session_date.eq((first_fold_start - timedelta(days=1)).isoformat()), "target_end"] = boundary
    folds = list(r.chronological_folds(train, horizon=10, n_folds=3))
    assert len(folds) == 3
    for _, _, fit, valid in folds:
        assert fit.datetime.max() < valid.datetime.min()
        assert (fit.target_end < valid.datetime.min()).all()
    assert not folds[0][2].session_date.eq((first_fold_start - timedelta(days=1)).isoformat()).any()


def test_only_10_and_15_minute_allowlisted_datasets_are_readable():
    with pytest.raises(ValueError, match="Only the generated"):
        r.load_dataset(5)

