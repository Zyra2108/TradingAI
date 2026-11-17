from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).with_name("run_sequence_research.py")
spec = importlib.util.spec_from_file_location("ecl_sequence_research_v1_test_module", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def _raw(times: list[str], offset: float = 0.0) -> pd.DataFrame:
    dt = pd.to_datetime(times)
    close = np.arange(100.0 + offset, 100.0 + offset + len(dt))
    return pd.DataFrame({
        "datetime": dt,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.3,
        "close": close,
        "volume": np.arange(len(dt), dtype=float),
    })


def _labels(times: list[str]) -> pd.DataFrame:
    dt = pd.to_datetime(times)
    return pd.DataFrame({"datetime": dt, "future_datetime": dt + pd.Timedelta(minutes=15), "target": ["UP"] * len(dt)})


def test_windows_are_exact_ohlcv_and_end_at_decision_candle():
    times = pd.date_range("2026-01-05 09:30", periods=8, freq="5min").astype(str).tolist()
    raw = _raw(times)
    labels = _labels([times[-1]])
    eligible, x, counts = module.build_sequence_samples(raw, labels, 6)
    assert len(eligible) == 1 and counts["eligible_rows"] == 1
    assert x.shape == (1, 6, 5)
    assert np.isclose(x[0, -1, 3], 0.0)
    assert np.isclose(x[0, 0, 3], 100 * (raw.close.iloc[2] / raw.close.iloc[7] - 1))
    assert np.isclose(x[0, -1, 4], np.log1p(raw.volume.iloc[7]))


def test_missing_five_minute_gap_breaks_sequence_and_never_bridges():
    before = pd.date_range("2026-01-05 09:30", periods=6, freq="5min")
    after = pd.date_range("2026-01-05 10:05", periods=6, freq="5min")
    all_times = before.append(after).astype(str).tolist()
    raw = _raw(all_times)
    labels = _labels([all_times[5], all_times[6], all_times[10], all_times[11]])
    eligible, _, counts = module.build_sequence_samples(raw, labels, 6)
    assert eligible.datetime.tolist() == [pd.Timestamp(all_times[5]), pd.Timestamp(all_times[11])]
    assert counts["insufficient_contiguous_session_warmup"] == 2


def test_session_boundary_resets_sequence_warmup():
    day_one = pd.date_range("2026-01-05 15:30", periods=6, freq="5min")
    day_two = pd.date_range("2026-01-06 09:30", periods=6, freq="5min")
    all_times = day_one.append(day_two).astype(str).tolist()
    raw = _raw(all_times)
    labels = _labels([all_times[5], all_times[6], all_times[10], all_times[11]])
    eligible, _, counts = module.build_sequence_samples(raw, labels, 6)
    assert eligible.datetime.tolist() == [pd.Timestamp(all_times[5]), pd.Timestamp(all_times[11])]
    assert counts["insufficient_contiguous_session_warmup"] == 2


def test_future_candle_changes_do_not_change_prior_sequence():
    times = pd.date_range("2026-01-05 09:30", periods=9, freq="5min").astype(str).tolist()
    raw = _raw(times)
    label = _labels([times[5]])
    _, x_before, _ = module.build_sequence_samples(raw, label, 6)
    raw.loc[8, ["open", "high", "low", "close", "volume"]] = [10000, 11000, 9000, 10500, 999999]
    _, x_after, _ = module.build_sequence_samples(raw, label, 6)
    np.testing.assert_array_equal(x_before, x_after)


def test_fold_masks_are_chronological_and_purge_forward_target_overlap():
    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-05 15:40", "2026-01-05 15:45", "2026-01-06 09:30", "2026-01-06 09:35"]),
        "future_datetime": pd.to_datetime(["2026-01-05 15:55", "2026-01-06 10:00", "2026-01-06 09:45", "2026-01-06 09:50"]),
        "target": ["UP", "DOWN", "UP", "NEUTRAL"],
    })
    train_mask, val_mask, first_val = module.chronological_fold_masks(frame, "2026-01-05", "2026-01-06", "2026-01-06")
    assert val_mask.tolist() == [False, False, True, True]
    assert train_mask.tolist() == [True, False, False, False]
    assert frame.loc[train_mask, "future_datetime"].max() < first_val


def test_only_approved_source_paths_are_readable():
    assert module.assert_approved_path(module.TRAIN_CSV) == module.TRAIN_CSV.resolve()
    with pytest.raises(ValueError, match="Unapproved research input path"):
        module.assert_approved_path(module.MODEL_DIR / "unapproved_input.csv")


def test_probability_metrics_exclude_neutral_from_directional_but_count_it_operationally():
    actual = np.array(["DOWN", "UP", "NEUTRAL"])
    directional = actual[actual != "NEUTRAL"]
    result = module.binary_metrics(
        directional,
        np.array(["DOWN", "DOWN"]),
        np.array([0.1, 0.2]),
        all_targets=actual,
        all_predictions=np.array(["DOWN", "DOWN", "UP"]),
    )
    assert result["n_directional"] == 2
    assert result["confusion_matrix_actual_rows_predicted_columns_down_up"] == [[1, 0], [1, 0]]
    assert result["neutral_rows"] == 1
    assert result["all_candle_operational_accuracy_neutral_wrong"] == pytest.approx(1 / 3)


def test_cnn_save_load_probabilities_are_identical(tmp_path):
    tf = module._tensorflow()
    x = np.random.default_rng(7).normal(size=(12, 6, 5)).astype(np.float32)
    model = module.build_cnn(6, seed=17)
    model.fit(x, np.array([0, 1] * 6, dtype=np.float32), epochs=1, batch_size=4, shuffle=False, verbose=0)
    before = model.predict(x[:3], verbose=0)
    model_path = tmp_path / "tiny.keras"
    model.save(model_path, include_optimizer=False)
    loaded = tf.keras.models.load_model(model_path)
    after = loaded.predict(x[:3], verbose=0)
    np.testing.assert_allclose(before, after, atol=1e-7)
