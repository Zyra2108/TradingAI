from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import pytest

from config import get_alpaca_settings
from live_app.alpaca_storage import AlpacaBarStore
from live_app.alpaca_engine import AlpacaPredictionEngine
from live_app.app import create_app
from live_app.historical_model_signals import (
    HistoricalSignalStore,
    _feature_frame,
    generate_historical_signals,
)
from live_app.model_loader import LoadedCandidate
from fastapi.testclient import TestClient

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
SPEC = __import__("json").loads((ROOT / "models/alpaca_final_freeze_v1/frozen_candidate_spec.json").read_text())
FEATURES = SPEC["features_exact_order"]
UTC = timezone.utc


class Model:
    def __init__(self):
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.classes_ = np.asarray(["DOWN", "NEUTRAL", "UP"], dtype=object)
        self.calls = 0

    def predict_proba(self, frame):
        self.calls += len(frame)
        assert list(frame.columns) == FEATURES
        return np.tile([0.2, 0.3, 0.5], (len(frame), 1))


class Engine:
    def __init__(self, path):
        self.store = AlpacaBarStore(path)
        self.model_sha256 = "a" * 64
        self.candidate = None


def candidate(model=None):
    return LoadedCandidate(model or Model(), SPEC, "b" * 64)


def candles(count=35):
    start = datetime(2026, 9, 24, 13, 30, tzinfo=UTC)
    output = []
    for i in range(count):
        stamp = start + timedelta(minutes=5 * i)
        close = 100 + i * 0.04
        output.append({
            "time": int(stamp.timestamp()),
            "candle_time_utc": stamp.isoformat().replace("+00:00", "Z"),
            "open": close - 0.01, "high": close + 0.03,
            "low": close - 0.03, "close": close, "volume": 100 + i,
            "source": "historical", "feed": "IEX",
        })
    return output


def test_frozen_features_are_causal_ordered_and_gap_segmented():
    rows = candles()
    full = _feature_frame(rows, FEATURES)
    prefix = _feature_frame(rows[:28], FEATURES)
    assert list(full.columns) == ["datetime", *FEATURES]
    pd.testing.assert_frame_equal(full.iloc[:28].reset_index(drop=True), prefix)

    gap_rows = rows[:22] + rows[23:]
    gap = _feature_frame(gap_rows, FEATURES)
    gap_stamp = pd.Timestamp(datetime.fromtimestamp(gap_rows[22]["time"], UTC).astimezone(
        __import__("zoneinfo").ZoneInfo("America/New_York")).replace(tzinfo=None))
    row = gap.loc[gap.datetime.eq(gap_stamp)].iloc[0]
    assert pd.isna(row["return_1"])
    assert pd.isna(row["sma_ratio_5"])

    next_session = candles(24)
    next_session.extend({**row, "time": row["time"] + 86400,
                         "candle_time_utc": (datetime.fromtimestamp(row["time"], UTC) + timedelta(days=1)).isoformat().replace("+00:00", "Z")}
                        for row in candles(3))
    next_session_features = _feature_frame(next_session, FEATURES)
    second_day = next_session_features.iloc[24]
    assert pd.isna(second_day["return_1"])
    assert pd.isna(second_day["sma_ratio_5"])


def test_historical_generation_is_idempotent_separate_and_uses_exact_contract(tmp_path):
    model = Model()
    store = HistoricalSignalStore(tmp_path / "historical.sqlite3")
    rows = candles()
    first = generate_historical_signals(store=store, candles=rows, candidate=candidate(model))
    second = generate_historical_signals(store=store, candles=rows, candidate=candidate(model))
    assert first["historical_predictions_generated"] > 0
    assert first["records_inserted_this_run"] == first["historical_predictions_generated"]
    assert second["records_inserted_this_run"] == 0
    assert store.count() == first["historical_predictions_generated"]
    assert model.calls == first["historical_predictions_generated"] * 2
    saved = store.rows(1)[0]
    assert saved["source"] == "historical_backtest"
    assert saved["horizon_minutes"] == 15
    assert saved["model_sha256"] == "b" * 64
    assert json.loads(saved["input_features_json"]).keys() == set(FEATURES)
    assert saved["prediction"] == "UP"
    with pytest.raises(Exception, match="immutable"):
        with store._connect() as connection:
            connection.execute("UPDATE historical_model_signals SET prediction='DOWN'")


def test_historical_outcome_requires_complete_exact_same_session_path(tmp_path):
    store = HistoricalSignalStore(tmp_path / "outcomes.sqlite3")
    rows = candles()
    rows.pop(3)
    result = generate_historical_signals(store=store, candles=rows, candidate=candidate())
    saved = store.rows(100)
    assert result["historical_predictions_generated"] > 0
    assert any(row["outcome_status"] == "UNAVAILABLE" and row["unavailable_reason"] == "incomplete_or_missing_outcome_path" for row in saved)
    assert any(row["outcome_status"] == "SETTLED" for row in saved)


def test_historical_outcome_rejects_endpoint_beyond_regular_session_close(tmp_path):
    store = HistoricalSignalStore(tmp_path / "session_close.sqlite3")
    rows = candles(78)
    generate_historical_signals(store=store, candles=rows, candidate=candidate())
    decision = datetime(2026, 9, 24, 19, 45, tzinfo=UTC).isoformat().replace("+00:00", "Z")
    signal = next(row for row in store.rows(1000) if row["candle_time_utc"] == decision)
    assert signal["outcome_status"] == "UNAVAILABLE"
    assert signal["unavailable_reason"] == "endpoint_outside_same_session"


def test_pretest_cutoff_rejects_any_later_source_candle(tmp_path):
    rows = candles()
    late = dict(rows[-1])
    late["time"] = int(datetime(2026, 9, 25, 13, 30, tzinfo=UTC).timestamp())
    with pytest.raises(ValueError, match="permitted pre-test cutoff"):
        generate_historical_signals(store=HistoricalSignalStore(tmp_path / "guard.sqlite3"),
                                    candles=[*rows, late], candidate=candidate())


def test_api_returns_saved_historical_signals_and_never_enters_live_ledger(tmp_path):
    historical_store = HistoricalSignalStore(tmp_path / "history.sqlite3")
    rows = candles()
    generate_historical_signals(store=historical_store, candles=rows, candidate=candidate())
    bar_store = AlpacaBarStore(tmp_path / "live.sqlite3")
    engine = AlpacaPredictionEngine(candidate=candidate(), store=bar_store,
                                    now_fn=lambda: datetime(2026, 9, 24, 13, 30, tzinfo=UTC))
    settings = get_alpaca_settings({
        "APCA_API_KEY_ID": "synthetic-id", "APCA_API_SECRET_KEY": "synthetic-secret",
        "ALPACA_IEX_DISPLAY_AUTHORIZED": "true",
    })
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=settings,
                     alpaca_provider_factory=lambda *_: None, historical_signal_store=historical_store)
    with TestClient(app, client=("127.0.0.1", 9181)) as client:
        signals = client.get("/api/historical-model-signals").json()["signals"]
        selected = client.get("/api/market-candle-details", params={"candle_time_utc": signals[-1]["candle_time_utc"]}).json()
    assert signals and all(row["source_type"] == "historical_backtest" for row in signals)
    assert selected["prediction"]["source_type"] == "historical_backtest"
    assert engine.store.prediction_count() == 0


def test_indicator_dropdown_has_exclusive_model_and_sma_modes_without_reload():
    html = (ROOT / "live_app/static/alpaca.html").read_text()
    script = (ROOT / "live_app/static/alpaca.js").read_text()
    assert '<select id="indicator-select"' in html
    assert '<option value="model">Model Indicator</option>' in html
    assert '<option value="sma" selected>SMA Indicator</option>' in html
    assert 'id="indicator-context-title">SMA 20</strong>' in html
    assert 'id="indicator-explanation"' in html
    assert "const selected = el('indicator-select').value" in script
    assert "el('indicator-context-title').textContent = smaSelected ? 'SMA 20' : 'Model Indicator'" in script
    assert "renderModelMarkers();" in script and "indicatorData(timeline, 20, smaSelected)" in script
    assert "indicator-select').addEventListener('change', renderIndicator)" in script
    render_block = script.split("function renderIndicator()", 1)[1].split("function renderIndicatorExplanation()", 1)[0]
    assert "setVisibleLogicalRange" not in render_block and "resetView" not in render_block
    explanation_block = script.split("function renderIndicatorExplanation()", 1)[1].split("function renderModelMarkers()", 1)[0]
    assert "HB means Historical Backtest; LIVE means genuine Live Prediction" in explanation_block
    assert "retrospective, not predictions originally issued live" in explanation_block
    assert "20 consecutive valid closing prices" in explanation_block
    assert "resets after genuine data gaps or session boundaries" in explanation_block
    marker_block = script.split("function renderModelMarkers()", 1)[1].split("function updateTimeline", 1)[0]
    assert "setMarkers([])" in marker_block
    assert "${historical ? 'HB' : 'LIVE'} ${row.prediction}" in marker_block
    selected_block = script.split("function renderSelectedCandle()", 1)[1].split("function selectCandle(time)", 1)[0]
    assert "renderLatestPrediction" not in selected_block
    assert "Historical Backtest" in script and "retrospective, not predictions originally issued live" in script


def test_dashboard_uses_professional_dark_theme_without_changing_indicator_modes():
    html = (ROOT / "live_app/static/alpaca.html").read_text()
    css = (ROOT / "live_app/static/alpaca.css").read_text()
    assert 'id="indicator-context"' in html
    assert ".indicator-context" in css
    assert "color-scheme:dark" in css and "--bg:#090d12" in css
    assert ".panel{min-width:0;background:var(--surface)" in css
    assert "--green:#22b879" in css and "--red:#e35d67" in css
