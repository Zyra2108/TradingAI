import numpy as np
import pandas as pd

from evaluate_all_models import aligned_probabilities, directional_metrics, make_sequences, metrics


def test_three_class_metrics_use_actual_rows_predicted_columns_order():
    out = metrics(np.array(["DOWN", "UP", "NEUTRAL"]), np.array(["DOWN", "DOWN", "NEUTRAL"]), ["DOWN", "NEUTRAL", "UP"])
    assert out["confusion_matrix_actual_rows_predicted_columns"] == [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
    assert out["n_evaluated"] == 3


def test_directional_accuracy_and_all_candle_metric_count_neutral_as_wrong():
    actual = np.array(["UP", "DOWN", "NEUTRAL"])
    pred = np.array(["UP", "UP", "UP"])
    out = directional_metrics(actual, pred, np.isin(actual, ["DOWN", "UP"]))
    assert out["accuracy"] == 0.5
    assert out["all_candle_accuracy_neutral_wrong"] == 1 / 3
    assert out["prediction_distribution"] == {"DOWN": 0, "UP": 2}


def test_encoded_predict_proba_columns_follow_saved_class_order():
    class FakeEstimator:
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, x):
            return np.tile([0.1, 0.2, 0.7], (len(x), 1))

    probs, labels = aligned_probabilities(FakeEstimator(), pd.DataFrame({"x": [1, 2]}), ["DOWN", "NEUTRAL", "UP"])
    assert labels == ["DOWN", "NEUTRAL", "UP"]
    assert np.allclose(probs, [[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])


def test_sequence_builder_uses_only_causal_contiguous_same_session_bars():
    start = pd.Timestamp("2026-10-01 09:30")
    # The missing 09:40 bar breaks the 3-bar window for the 09:45 decision.
    times = [start, start + pd.Timedelta(minutes=5), start + pd.Timedelta(minutes=15), start + pd.Timedelta(minutes=20)]
    raw = pd.DataFrame({
        "datetime": times,
        "open": [10.0, 10.1, 10.3, 10.4],
        "high": [10.1, 10.2, 10.4, 10.5],
        "low": [9.9, 10.0, 10.2, 10.3],
        "close": [10.0, 10.1, 10.3, 10.4],
        "volume": [10, 11, 13, 14],
    })
    test = pd.DataFrame({"datetime": [times[2], times[3]]})
    sequences, eligible = make_sequences(raw, test, 2)
    assert eligible.tolist() == [False, True]
    assert sequences.shape == (1, 2, 5)
    # The retained sample ends at 09:50 and contains only 09:45 and 09:50.
    # Current close is zero after relative-price normalization; no later row exists.
    assert np.isclose(sequences[0, -1, 3], 0.0)
    assert np.isclose(sequences[0, 0, 3], 100.0 * (10.3 / 10.4 - 1.0))
