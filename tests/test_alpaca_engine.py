from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import live_app.model_loader as model_loader
from live_app.alpaca_aggregation import AggregatedFiveMinuteCandle, AlpacaFiveMinuteAggregator
from live_app.alpaca_engine import AlpacaPredictionEngine
from live_app.alpaca_storage import AlpacaBarStore
from live_app.calendar import NY, session_for_date
from live_app.features import calculate_b_features
from live_app.model_loader import LoadedCandidate

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "models" / "alpaca_final_freeze_v1" / "frozen_candidate_spec.json"
MODEL_PATH = ROOT / "models" / "alpaca_final_freeze_v1" / "candidate_pipeline.joblib"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
FEATURES = SPEC["features_exact_order"]
CLASSES = ["DOWN", "NEUTRAL", "UP"]
DAY = date(2026, 11, 12)
UTC = timezone.utc


class FakeModel:
    def __init__(self, probabilities=(0.15, 0.70, 0.15)):
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.classes_ = np.asarray(CLASSES, dtype=object)
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.calls = []

    def predict_proba(self, frame):
        self.calls.append(frame.copy())
        return np.tile(self.probabilities, (len(frame), 1))


def bounds(day=DAY):
    opened, closed = session_for_date(day)
    return opened.astimezone(UTC), closed.astimezone(UTC)


def make_candle(stamp, close, *, finalized=None):
    return AggregatedFiveMinuteCandle(
        candle_time_utc=stamp,
        finalized_at_utc=finalized or stamp + timedelta(minutes=7),
        open=close, high=close + 0.1, low=close - 0.1, close=close,
        volume=123.0, trade_count=9, source="Alpaca", feed="iex",
        aggregation_method="five_consecutive_1min_ohlc_v1",
        aggregation_metadata={"constituent_count": 5},
    )


def make_candidate(model=None):
    return LoadedCandidate(model or FakeModel(), SPEC, SPEC["source_candidate_sha256"])


def prepare_engine(tmp_path, *, probabilities=(0.15, 0.70, 0.15), now=None):
    store = AlpacaBarStore(tmp_path / "alpaca.sqlite3")
    model = FakeModel(probabilities)
    current = now or bounds()[0]
    engine = AlpacaPredictionEngine(candidate=make_candidate(model), store=store,
                                    now_fn=lambda: current)
    return engine, model, store


def seed_session_candles(store, *, count=34, day=DAY, start_price=100.0):
    opened, _ = bounds(day)
    stamps = []
    for i in range(count):
        stamp = opened + timedelta(minutes=5 * i)
        price = start_price + i * 0.04 + (0.015 if i % 3 else -0.01)
        store.insert_five_minute_candle(make_candle(stamp, price))
        stamps.append(stamp)
    return stamps


def test_frozen_loader_hash_and_model_contract_are_rechecked(tmp_path, monkeypatch):
    before = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    assert before == SPEC["source_candidate_sha256"]
    model = FakeModel()
    monkeypatch.setattr(model_loader.joblib, "load", lambda _path: model)
    loaded = model_loader.load_alpaca_candidate(MODEL_PATH, SPEC_PATH)
    assert loaded.sha256 == before
    engine = AlpacaPredictionEngine(candidate=loaded, db_path=tmp_path / "loaded.sqlite3",
                                    now_fn=lambda: bounds()[0])
    assert engine.status()["model_ready"] is True
    assert engine.feature_order == FEATURES
    assert engine.classes == CLASSES
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == before


def test_schema_v1_to_v4_migration_is_additive(tmp_path):
    path = tmp_path / "v1.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at_utc TEXT NOT NULL, description TEXT NOT NULL);
            CREATE TABLE five_minute_candles(candle_time_utc TEXT PRIMARY KEY, finalized_at_utc TEXT NOT NULL,
                open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
                volume REAL, trade_count INTEGER, source TEXT NOT NULL, feed TEXT NOT NULL,
                aggregation_method TEXT NOT NULL, aggregation_metadata_json TEXT NOT NULL);
            INSERT INTO schema_migrations VALUES(1, 'synthetic', 'synthetic v1');
            INSERT INTO five_minute_candles VALUES('2026-11-12T14:30:00.000000Z', '2026-11-12T14:37:00.000000Z',
                100, 101, 99, 100, NULL, NULL, 'Alpaca', 'iex', 'synthetic', '{}');
            PRAGMA user_version=1;
        """)
    store = AlpacaBarStore(path)
    with store.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 4
        assert db.execute("SELECT COUNT(*) FROM five_minute_candles").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 4
        assert {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")} >= {"predictions", "outcomes"}


@pytest.mark.parametrize(("probabilities", "expected"), [
    ((0.8, 0.1, 0.1), "DOWN"),
    ((0.1, 0.8, 0.1), "NEUTRAL"),
    ((0.1, 0.1, 0.8), "UP"),
])
def test_prediction_creation_uses_exact_features_argmax_and_omits_volume(tmp_path, probabilities, expected):
    engine, model, store = prepare_engine(tmp_path, probabilities=probabilities)
    stamps = seed_session_candles(store)
    rows = store.five_minute_candles_for_session(*bounds())
    local_times = pd.to_datetime([row["candle_time_utc"] for row in rows], utc=True).tz_convert(NY).tz_localize(None)
    expected_features = calculate_b_features(pd.DataFrame({
        "datetime": local_times,
        "open": [row["open"] for row in rows], "high": [row["high"] for row in rows],
        "low": [row["low"] for row in rows], "close": [row["close"] for row in rows],
    }), FEATURES)
    expected_row = expected_features.loc[expected_features.datetime.eq(pd.Timestamp(local_times[30]))].iloc[0]
    decision = stamps[30]
    result = engine.predict_finalized_candle(make_candle(decision, rows[30]["close"]),
                                             now=decision + timedelta(minutes=7))
    assert result["status"] == "predicted"
    assert result["prediction"] == expected
    assert list(model.calls[-1].columns) == FEATURES
    assert "volume" not in model.calls[-1].columns
    np.testing.assert_allclose(model.calls[-1].iloc[0].astype(float), expected_row[FEATURES].astype(float), rtol=0, atol=1e-12)
    assert store.prediction_count() == 1


def test_duplicate_prediction_is_idempotent_and_keeps_reference_close_endpoint(tmp_path):
    engine, model, store = prepare_engine(tmp_path)
    stamps = seed_session_candles(store)
    decision = stamps[30]
    candle = make_candle(decision, 123.45)
    first = engine.predict_finalized_candle(candle, now=decision + timedelta(minutes=7))
    calls = len(model.calls)
    duplicate = engine.predict_finalized_candle(candle, now=decision + timedelta(minutes=8))
    saved = store.prediction_at(decision)
    assert first["status"] == "predicted"
    assert duplicate["status"] == "already_predicted"
    assert len(model.calls) == calls
    assert store.prediction_count() == 1
    assert saved["reference_close"] == store.five_minute_candle_at(decision)["close"]
    expected_endpoint = (decision + timedelta(minutes=15)).isoformat(timespec="microseconds").replace("+00:00", "Z")
    assert saved["outcome_candle_time_utc"] == expected_endpoint
    assert saved["horizon_minutes"] == 15
    assert saved["model_sha256"] == engine.model_sha256


@pytest.mark.parametrize("probabilities", [
    (0.2, 0.2, 0.2), (-0.1, 0.5, 0.6), (float("nan"), 0.5, 0.5),
    (0.1, 0.9),
])
def test_invalid_probabilities_are_rejected(tmp_path, probabilities):
    engine, _, store = prepare_engine(tmp_path, probabilities=probabilities)
    stamps = seed_session_candles(store)
    with pytest.raises(ValueError, match="probabilities|probability shape"):
        engine.predict_finalized_candle(make_candle(stamps[30], 100.0),
                                        now=stamps[30] + timedelta(minutes=7))
    assert store.prediction_count() == 0


def test_feature_order_drift_is_rejected_before_inference(tmp_path):
    engine, model, store = prepare_engine(tmp_path)
    stamps = seed_session_candles(store)
    engine.model.feature_names_in_ = np.asarray(list(reversed(FEATURES)), dtype=object)
    with pytest.raises(ValueError, match="feature order"):
        engine.predict_finalized_candle(make_candle(stamps[30], 100.0),
                                        now=stamps[30] + timedelta(minutes=7))
    assert not model.calls
    assert store.prediction_count() == 0


def test_inadequate_warmup_leaves_feature_not_ready_and_no_prediction(tmp_path):
    engine, _, store = prepare_engine(tmp_path)
    stamps = seed_session_candles(store, count=10)
    result = engine.predict_finalized_candle(make_candle(stamps[-1], 101.0),
                                             now=stamps[-1] + timedelta(minutes=7))
    assert result["status"] == "features_not_ready"
    assert engine.status()["feature_ready"] is False
    assert store.prediction_count() == 0


def test_prediction_attempt_diagnostic_is_independent_of_later_aggregation_status(tmp_path):
    engine, model, store = prepare_engine(tmp_path)
    stamps = seed_session_candles(store, count=10)
    result = engine.predict_finalized_candle(make_candle(stamps[-1], 101.0),
                                             now=stamps[-1] + timedelta(minutes=7))
    assert result["status"] == "features_not_ready"
    attempt_before = engine.status()["latest_prediction_attempt"]
    assert attempt_before["status"] == "feature_warmup_incomplete"
    assert attempt_before["reason"] == "consecutive_candle_warmup_insufficient"
    assert attempt_before["missing_features"]

    # A later rejected bucket updates aggregation diagnostics only.
    engine._last_aggregation_status = "known_stream_interruption"
    status = engine.status()
    assert status["last_aggregation_status"] == "known_stream_interruption"
    assert status["latest_prediction_attempt"] == attempt_before
    assert model.calls == [] and store.prediction_count() == 0


def seed_prediction_path(store, decision, *, omitted=(), endpoint_close=100.2, prediction="UP"):
    base = 100.0
    for offset in (0, 5, 10, 15):
        if offset in omitted:
            continue
        close = endpoint_close if offset == 15 else base
        store.insert_five_minute_candle(make_candle(decision + timedelta(minutes=offset), close))
    store.insert_prediction({
        "candle_time_utc": decision,
        "prediction_created_utc": decision + timedelta(minutes=7),
        "prediction": prediction,
        "probability_down": .1, "probability_neutral": .2, "probability_up": .7,
        "confidence": .7, "reference_close": base,
        "horizon_minutes": 15,
        "outcome_candle_time_utc": decision + timedelta(minutes=15),
        "model_sha256": "a" * 64, "input_features": {"close": base},
        "provider_delay_seconds": 120,
    })


def test_settlement_requires_full_path_waits_for_buffer_and_retries(tmp_path):
    engine, _, store = prepare_engine(tmp_path)
    decision = bounds()[0]
    seed_prediction_path(store, decision, omitted=(10,))
    endpoint_finalized = decision + timedelta(minutes=22)
    assert engine.settle_due_predictions(now=endpoint_finalized) == 0
    assert store.outcome_count() == 0
    store.insert_five_minute_candle(make_candle(decision + timedelta(minutes=10), 100.0))
    before_buffer = decision + timedelta(minutes=21, seconds=59)
    assert engine.settle_due_predictions(now=before_buffer) == 0
    assert engine.settle_due_predictions(now=endpoint_finalized) == 1
    assert store.outcome_count() == 1
    assert engine.settle_due_predictions(now=endpoint_finalized + timedelta(minutes=1)) == 0
    assert store.outcome_count() == 1


def test_session_boundary_path_is_rejected(tmp_path):
    engine, _, store = prepare_engine(tmp_path)
    _, close = bounds()
    decision = close - timedelta(minutes=15)
    seed_prediction_path(store, decision)
    assert engine.settle_due_predictions(now=close + timedelta(minutes=10)) == 0
    assert store.outcome_count() == 0


def test_prediction_and_outcome_rows_are_immutable(tmp_path):
    engine, _, store = prepare_engine(tmp_path)
    decision = bounds()[0]
    seed_prediction_path(store, decision)
    endpoint = decision + timedelta(minutes=15)
    assert store.insert_outcome({
        "prediction_candle_time_utc": decision,
        "observed_utc": endpoint + timedelta(minutes=7),
        "endpoint_candle_time_utc": endpoint,
        "actual_target": "UP", "actual_return_pct": .2, "correct": True,
    }) is True
    with store.connect() as db:
        with pytest.raises(Exception, match="immutable"):
            db.execute("UPDATE predictions SET prediction='DOWN' WHERE candle_time_utc=?",
                       (decision.isoformat(timespec="microseconds").replace("+00:00", "Z"),))
        with pytest.raises(Exception, match="immutable"):
            db.execute("DELETE FROM outcomes WHERE prediction_candle_time_utc=?",
                       (decision.isoformat(timespec="microseconds").replace("+00:00", "Z"),))


def test_restart_recovery_finalizes_old_bucket_but_never_creates_retroactive_prediction(tmp_path):
    store = AlpacaBarStore(tmp_path / "alpaca.sqlite3")
    opened, _ = bounds()
    bucket = opened
    for i in range(5):
        stamp = bucket + timedelta(minutes=i)
        price = 100 + i * .01
        from live_app.alpaca_provider import MinuteBarEvent
        store.insert_minute_bar(MinuteBarEvent("ECL", stamp.isoformat().replace("+00:00", "Z"), stamp,
                                                stamp + timedelta(minutes=1), price, price + .1,
                                                price - .1, price, 1.0, 1))
    startup = bucket + timedelta(minutes=10)
    engine = AlpacaPredictionEngine(candidate=make_candidate(), store=store,
                                    now_fn=lambda: startup)
    recovered = engine.recover_aggregations(now=startup + timedelta(minutes=1))
    assert any(item["aggregation_status"] == "finalized" for item in recovered)
    assert store.five_minute_count() == 1
    assert store.prediction_count() == 0
    assert not engine.candidate.model.calls


def test_invalid_horizon_definition_is_rejected(tmp_path):
    bad_spec = dict(SPEC)
    bad_spec["target"] = dict(SPEC["target"], up="future return > +0.20%")
    candidate = LoadedCandidate(FakeModel(), bad_spec, "a" * 64)
    with pytest.raises(ValueError, match="thresholds"):
        AlpacaPredictionEngine(candidate=candidate, db_path=tmp_path / "bad.sqlite3",
                               now_fn=lambda: bounds()[0])


def test_engine_rejects_aggregator_using_a_different_database(tmp_path):
    store = AlpacaBarStore(tmp_path / "engine.sqlite3")
    other_store = AlpacaBarStore(tmp_path / "aggregator.sqlite3")
    aggregator = AlpacaFiveMinuteAggregator(other_store)
    with pytest.raises(ValueError, match="same database"):
        AlpacaPredictionEngine(candidate=make_candidate(), store=store, aggregator=aggregator,
                               now_fn=lambda: bounds()[0])
