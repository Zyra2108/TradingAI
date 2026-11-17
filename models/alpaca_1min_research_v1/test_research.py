from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

pipe = load("alpaca_1min_pipe_test", "feature_and_label_pipeline.py")
runner = load("alpaca_1min_runner_test", "run_research.py")


def test_ten_minute_label_threshold_and_exact_endpoint():
    start = pd.Timestamp("2026-11-12 09:30:00")
    stamps = pd.date_range(start, periods=21, freq="1min")
    closes = np.r_[np.full(10, 100.0), 100.2, np.full(10, 100.2)]
    bars = pd.DataFrame({"datetime": stamps, "close": closes})
    labels = pipe.generate_labels(bars, 1)
    first = labels.iloc[0]
    assert first.target_end == start + pd.Timedelta(minutes=10)
    assert first.future_return_pct > 0.10
    assert first.label == "UP"


def test_target_rejects_missing_intermediate_minute_even_with_endpoint():
    stamps = list(pd.date_range("2026-11-12 09:30:00", periods=11, freq="1min"))
    stamps[5] += pd.Timedelta(seconds=30)
    bars = pd.DataFrame({"datetime": stamps, "close": np.linspace(100, 100.2, len(stamps))})
    labels = pipe.generate_labels(bars, 1)
    assert not bool(labels.iloc[0].target_available)
    assert labels.iloc[0].label == "UNAVAILABLE"


def test_target_rejects_session_boundary_and_missing_endpoint():
    stamps = pd.date_range("2026-11-12 15:50:00", periods=11, freq="1min")
    bars = pd.DataFrame({"datetime": stamps, "close": np.full(len(stamps), 100.0)})
    labels = pipe.generate_labels(bars, 1)
    assert not bool(labels.iloc[0].target_available)
    assert not bool(labels.iloc[-1].target_available)


def test_feature_warmup_and_causality_at_completed_candle():
    stamps = pd.date_range("2026-11-12 09:30:00", periods=25, freq="1min")
    close = 100 + np.arange(len(stamps)) * 0.01
    bars = pd.DataFrame({"datetime": stamps, "open": close - .005, "high": close + .01,
                         "low": close - .01, "close": close, "volume": 100 + np.arange(len(stamps))})
    full = pipe.one_minute_features(bars)
    prefix = pipe.one_minute_features(bars.iloc[:20])
    at = pd.Timestamp("2026-11-12 09:49:00")
    full_row = full.loc[full.datetime.eq(at), pipe.ONE_MINUTE_FEATURES].to_numpy(float)
    prefix_row = prefix.loc[prefix.datetime.eq(at), pipe.ONE_MINUTE_FEATURES].to_numpy(float)
    assert full_row.shape == (1, 30)
    assert np.isfinite(full_row).all()
    np.testing.assert_allclose(full_row, prefix_row, rtol=0, atol=0, equal_nan=True)


def test_gap_resets_feature_warmup():
    stamps = list(pd.date_range("2026-11-12 09:30:00", periods=15, freq="1min"))
    stamps += list(pd.date_range("2026-11-12 09:46:00", periods=12, freq="1min"))
    close = 100 + np.arange(len(stamps)) * .01
    bars = pd.DataFrame({"datetime": stamps, "open": close, "high": close + .01,
                         "low": close - .01, "close": close, "volume": 100})
    feats = pipe.one_minute_features(bars)
    # The first row after a gap is retained for audit but rolling features
    # are unavailable until a full fresh segment warmup completes.
    first_after_gap = feats.loc[feats.datetime.eq(pd.Timestamp("2026-11-12 09:46:00")), pipe.ONE_MINUTE_FEATURES]
    warmed = feats.loc[feats.datetime.eq(pd.Timestamp("2026-11-12 09:56:00")), pipe.ONE_MINUTE_FEATURES]
    assert first_after_gap.iloc[0].isna().any()
    imputed_by_train_only = {"volume_log1p", "volume_ratio_10", "volume_zscore_10"}
    core = [c for c in pipe.ONE_MINUTE_FEATURES if c not in imputed_by_train_only]
    assert np.isfinite(warmed[core].to_numpy(float)).all()


def test_abstention_metrics_count_no_signal_as_directional_miss():
    actual = np.array(["UP", "UP", "DOWN", "DOWN", "NO_SIGNAL"])
    p_up = np.array([.9, .55, .1, .45, .95])
    metrics = runner.evaluate(actual, p_up, .8)
    assert metrics["n_emitted_signals_all_rows"] == 3
    assert metrics["n_emitted_on_neutral_actual"] == 1
    assert metrics["n_no_signal_outputs"] == 2
    assert metrics["eligible_directional_coverage"] == .5
    # Emitted directional accuracy ignores neutral targets; all-candle accuracy
    # counts neutral actuals and abstentions as wrong.
    assert metrics["directional_accuracy_emitted_eligible"] == 1.0
    assert metrics["all_candle_operational_accuracy_no_signal_and_actual_neutral_are_errors"] == .4
    assert metrics["macro_directional_f1_all_eligible_no_signal_is_miss"] < 1.0


def test_walk_forward_folds_are_chronological_and_disjoint():
    dates = pd.bdate_range("2025-01-01", periods=30).strftime("%Y-%m-%d").tolist()
    folds = runner.walk_forward_dates(dates)
    all_eval = []
    for fold in folds:
        assert max(fold["train_dates_before_purge"]) < min(fold["evaluation_dates"])
        assert set(fold["train_dates_before_purge"]).isdisjoint(fold["evaluation_dates"])
        all_eval.extend(fold["evaluation_dates"])
    assert len(all_eval) == len(set(all_eval))


def test_binary_model_probability_order_and_save_load(tmp_path):
    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples=80, n_features=4, random_state=19)
    model = runner._make_estimator("logistic_regression", y)
    model.fit(pd.DataFrame(x, columns=list("abcd")), y)
    before = runner._prob_up(model, pd.DataFrame(x[:8], columns=list("abcd")))
    artifact = tmp_path / "model.joblib"
    import pickle
    artifact.write_bytes(pickle.dumps(model))
    loaded = pickle.loads(artifact.read_bytes())
    after = runner._prob_up(loaded, pd.DataFrame(x[:8], columns=list("abcd")))
    np.testing.assert_array_equal(before, after)
