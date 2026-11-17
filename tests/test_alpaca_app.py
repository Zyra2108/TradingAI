from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from config import get_alpaca_settings
from live_app.alpaca_aggregation import AggregatedFiveMinuteCandle
from live_app.alpaca_engine import AlpacaPredictionEngine
from live_app.alpaca_provider import AlpacaIEXWebSocketClient
from live_app.alpaca_storage import AlpacaBarStore
from live_app.app import create_app
import live_app.app as app_module
import live_app.alpaca_rest_history as rest_history_module
from live_app.calendar import session_for_date
from live_app.model_loader import LoadedCandidate

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "models/alpaca_final_freeze_v1/frozen_candidate_spec.json").read_text(encoding="utf-8"))
FEATURES = SPEC["features_exact_order"]
CLASSES = ["DOWN", "NEUTRAL", "UP"]
UTC = timezone.utc
DAY = date(2026, 11, 12)


class AppModel:
    def __init__(self):
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.classes_ = np.asarray(CLASSES, dtype=object)
        self.calls = 0

    def predict_proba(self, frame):
        self.calls += 1
        assert list(frame.columns) == FEATURES
        return np.tile([0.1, 0.2, 0.7], (len(frame), 1))


class FakeSocket:
    def __init__(self, frames, *, close_when_empty=False):
        self.frames = deque(frames)
        self.close_when_empty = close_when_empty
        self.sent = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        if self.frames:
            return self.frames.popleft()
        if self.close_when_empty:
            raise ConnectionError("synthetic disconnect")
        await asyncio.Future()


def session_bounds():
    opened, closed = session_for_date(DAY)
    return opened.astimezone(UTC), closed.astimezone(UTC)


def candle(stamp, close, *, finalized=None):
    return AggregatedFiveMinuteCandle(
        candle_time_utc=stamp, finalized_at_utc=finalized or stamp + timedelta(minutes=7),
        open=close, high=close + .15, low=close - .15, close=close,
        volume=900.0, trade_count=30, source="Alpaca", feed="iex",
        aggregation_method="five_consecutive_1min_ohlc_v1",
        aggregation_metadata={"constituent_count": 5},
    )


def frame(obj):
    return json.dumps(obj)


def handshake():
    return [frame({"T": "success", "msg": "connected"}),
            frame({"T": "success", "msg": "authenticated"}),
            frame({"T": "subscription", "bars": ["ECL"], "updatedBars": ["ECL"]})]


def event_frame(stamp, i):
    price = 102.0 + i * .02
    return frame({"T": "b", "S": "ECL", "t": stamp.isoformat().replace("+00:00", "Z"),
                  "o": price, "h": price + .1, "l": price - .1, "c": price + .02,
                  "v": 12, "n": 3})


def build_engine(tmp_path, *, started_at):
    store = AlpacaBarStore(tmp_path / "alpaca.sqlite3")
    opened, _ = session_bounds()
    bucket = opened + timedelta(minutes=150)
    for i in range(30):
        stamp = opened + timedelta(minutes=5 * i)
        close = 100 + i * .04 + (.01 if i % 3 else -.005)
        store.insert_five_minute_candle(candle(stamp, close))
    model = AppModel()
    candidate = LoadedCandidate(model, SPEC, SPEC["source_candidate_sha256"])
    engine = AlpacaPredictionEngine(candidate=candidate, store=store, now_fn=lambda: started_at)
    return engine, model, bucket


def authorized_settings(enabled=True):
    return get_alpaca_settings({
        "APCA_API_KEY_ID": "synthetic-key-id",
        "APCA_API_SECRET_KEY": "synthetic-secret",
        "ALPACA_IEX_DISPLAY_AUTHORIZED": "true" if enabled else "false",
    })


def synthetic_provider_factory(frames, receive_times, *, connector_calls=None, backoff=0.01):
    def factory(settings, on_bar, on_connection_state=None, on_processing_failure=None):
        sockets = []
        def connector(_uri, **_kwargs):
            if connector_calls is not None:
                idx = len(connector_calls)
                connector_calls.append(idx)
                socket = FakeSocket(handshake() if idx == 0 else [*handshake(), *frames], close_when_empty=(idx == 0))
            else:
                socket = FakeSocket([*handshake(), *frames])
            sockets.append(socket)
            return socket
        times = iter(receive_times)
        provider = AlpacaIEXWebSocketClient(
            settings, connector=connector, on_bar=on_bar, on_connection_state=on_connection_state,
            on_processing_failure=on_processing_failure,
            receive_clock=lambda: next(times), initial_backoff_seconds=backoff,
            max_backoff_seconds=backoff, ack_timeout_seconds=1,
        )
        provider.synthetic_sockets = sockets
        return provider
    return factory


def wait_for_prediction(engine, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if engine.store.prediction_count():
            return True
        time.sleep(.01)
    return False


def test_app_lifecycle_reconnects_consumes_synthetic_websocket_and_serves_actual_data(tmp_path, monkeypatch):
    def no_network_refresh(refresher, _candles, *, now=None):
        refresher._status.update(state="current", last_success_utc=None, latest_bar_utc=None,
                                 bars_added=0, last_safe_error=None)
        return refresher.status
    monkeypatch.setattr(rest_history_module.AlpacaCurrentBarsRefresher, "refresh", no_network_refresh)
    opened, _ = session_bounds()
    bucket = opened + timedelta(minutes=150)
    receive_times = [bucket + timedelta(minutes=1 + i) for i in range(4)] + [bucket + timedelta(minutes=7)] * 5
    events = [event_frame(bucket + timedelta(minutes=i), i) for i in range(5)]
    engine, model, _ = build_engine(tmp_path, started_at=bucket)
    reconnect_attempts = []
    app = create_app(
        run_poller=True, alpaca_engine=engine, alpaca_settings=authorized_settings(),
        alpaca_provider_factory=synthetic_provider_factory(events, receive_times, connector_calls=reconnect_attempts),
    )
    with TestClient(app, client=("127.0.0.1", 9041)) as client:
        ready = wait_for_prediction(engine)
        assert ready, f"provider={app.state.provider.status}; reconnects={reconnect_attempts}; engine={engine.status()}"
        health = client.get("/api/health").json()
        status = client.get("/api/status").json()
        candles = client.get("/api/candles?limit=500").json()
        predictions = client.get("/api/predictions?limit=20").json()
        report = client.get("/api/report").json()
        csv_response = client.get("/api/report.csv")
        page = client.get("/").text
        assert health["mode"] == "alpaca_iex"
        assert status["provider"] == "Alpaca" and status["feed"] == "IEX"
        assert status["display_authorized"] is True
        assert status["research_preview_recording_enabled"] is True
        assert status["research_preview_log_file"].endswith("_app.jsonl")
        assert status["reconnect_count"] >= 1
        assert len(candles) == 31
        assert predictions[0]["prediction"] == "UP"
        assert predictions[0]["settlement_status"] == "PENDING"
        assert predictions[0]["reference_close"] == candles[-1]["close"]
        assert predictions[0]["outcome_candle_time_utc"].startswith((bucket + timedelta(minutes=15)).isoformat(timespec="microseconds").replace("+00:00", "Z"))
        assert report["predictions"] == 1 and report["pending"] == 1
        assert "Alpaca IEX" in page and "not consolidated" in page.lower()
        assert "ecl_alpaca_iex_prediction_report.csv" in csv_response.headers["content-disposition"]
        assert csv_response.text.count("\n") == 2
        assert model.calls == 1
        # Later completed five-minute candles permit one verified +15m outcome.
        for offset, close in ((5, 102.1), (10, 102.2), (15, 102.4)):
            engine.store.insert_five_minute_candle(candle(bucket + timedelta(minutes=offset), close))
        assert engine.settle_due_predictions(now=bucket + timedelta(minutes=22)) == 1
        settled = client.get("/api/predictions?limit=20").json()[0]
        assert settled["settlement_status"] == "SETTLED"
        assert settled["actual_target"] == "UP"
    assert app.state.provider.status["connected"] is False
    assert reconnect_attempts[:2] == [0, 1]


def test_app_periodic_sweep_finalizes_after_buffer_across_reconnect_and_stops_cleanly(tmp_path, monkeypatch):
    def no_network_refresh(refresher, _candles, *, now=None):
        refresher._status.update(state="current", last_success_utc=None, latest_bar_utc=None,
                                 bars_added=0, last_safe_error=None)
        return refresher.status
    monkeypatch.setattr(rest_history_module.AlpacaCurrentBarsRefresher, "refresh", no_network_refresh)
    monkeypatch.setattr(app_module, "ALPACA_MAINTENANCE_INTERVAL_SECONDS", 0.01)
    opened, _ = session_bounds()
    bucket = opened + timedelta(minutes=150)
    clock = [bucket]
    engine, model, _ = build_engine(tmp_path, started_at=bucket)
    # Startup/recovery and the maintenance sweep must observe the same advancing
    # fake clock, as they do with wall-clock time in the application.
    engine.now_fn = lambda: clock[0]
    engine.aggregator.clock = lambda: clock[0]
    frames = [event_frame(bucket + timedelta(minutes=i), i) for i in range(5)]
    received = iter(bucket + timedelta(minutes=i + 1) for i in range(5))
    connections = []

    def provider_factory(settings, on_bar, on_connection_state=None, on_processing_failure=None):
        def connector(_uri, **_kwargs):
            attempt = len(connections)
            connections.append(attempt)
            if attempt == 0:
                return FakeSocket([*handshake(), *frames[:2]], close_when_empty=True)
            return FakeSocket([*handshake(), *frames[2:]])

        def receive_clock():
            value = next(received)
            # The final source bar has arrived, but the 120-second finalization
            # buffer has not. Advance the engine clock only after its callback
            # has had its event timestamp; the periodic sweep must do the rest.
            if value == bucket + timedelta(minutes=5):
                clock[0] = bucket + timedelta(minutes=7)
            return value

        return AlpacaIEXWebSocketClient(
            settings, connector=connector, on_bar=on_bar, on_connection_state=on_connection_state,
            on_processing_failure=on_processing_failure,
            receive_clock=receive_clock,
            initial_backoff_seconds=.001, max_backoff_seconds=.001,
            ack_timeout_seconds=1,
        )

    app = create_app(run_poller=True, alpaca_engine=engine,
                     alpaca_settings=authorized_settings(),
                     alpaca_provider_factory=provider_factory)
    with TestClient(app, client=("127.0.0.1", 9051)):
        assert wait_for_prediction(engine, timeout=5), (
            f"provider={app.state.provider.status}; engine={engine.status()}"
        )
        assert engine.store.five_minute_candle_at(bucket) is not None
        assert engine.store.prediction_at(bucket) is not None
        assert engine.store.prediction_count() == 1
        assert model.calls == 1
        assert len(connections) >= 2
        assert app.state.provider.status["reconnect_count"] >= 1
        assert app.state.maintenance_task is not None
    assert app.state.maintenance_task.done()
    assert app.state.research_preview_task.done()
    assert app.state.provider.status["connected"] is False


def test_unauthorized_alpaca_mode_blocks_data_routes_and_never_starts_provider(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    factory_calls = []
    app = create_app(alpaca_engine=engine, alpaca_settings=authorized_settings(False),
                     alpaca_provider_factory=lambda *_args: factory_calls.append(True))
    with TestClient(app, client=("127.0.0.1", 9042)) as client:
        status = client.get("/api/status").json()
        assert status["state"] == "authorization_required"
        assert status["display_authorized"] is False
        for endpoint in ("/api/candles", "/api/predictions", "/api/report", "/api/report.csv"):
            assert client.get(endpoint).status_code == 403
        assert "Alpaca IEX" in client.get("/").text
    assert not factory_calls


def test_application_never_falls_back_from_alpaca_when_display_is_disabled(tmp_path):
    engine, _, _ = build_engine(tmp_path / "alpaca", started_at=session_bounds()[0])
    app = create_app(alpaca_engine=engine, alpaca_settings=authorized_settings(False))
    assert app.state.mode == "alpaca_iex"
    assert app.state.provider is None
    with TestClient(app, client=("127.0.0.1", 9043)) as client:
        status = client.get("/api/status").json()
        assert status["mode"] == "alpaca_iex"
        assert status["state"] == "authorization_required"
        assert "Alpaca" in client.get("/api/health").json()["provider"]


def test_alpaca_frontend_uses_live_endpoints_and_has_no_placeholder_market_data():
    html = (ROOT / "live_app/static/alpaca.html").read_text(encoding="utf-8")
    script = (ROOT / "live_app/static/alpaca.js").read_text(encoding="utf-8")
    assert "Alpaca IEX" in html
    assert "/api/candles" in script and "/api/predictions" in script and "/api/status" in script
    assert "candles.setData(candles)" not in script
    assert "America/New_York" in script


def test_dashboard_header_has_no_paper_trading_badge_or_unused_style():
    html = (ROOT / "live_app/static/alpaca.html").read_text(encoding="utf-8")
    css = (ROOT / "live_app/static/alpaca.css").read_text(encoding="utf-8")
    assert "PAPER TRADING · NO ORDERS" not in html
    assert 'class="paper-tag"' not in html
    assert ".paper-tag" not in css
    assert 'class="source-tag"' in html


def test_latest_prediction_sidebar_is_separate_from_selected_candle_and_uses_saved_records():
    html = (ROOT / "live_app/static/alpaca.html").read_text(encoding="utf-8")
    script = (ROOT / "live_app/static/alpaca.js").read_text(encoding="utf-8")
    assert html.index('id="latest-prediction-content"') < html.index('id="detail-open"')
    assert 'id="latest-prediction-empty"' in html and 'No model prediction recorded yet' in script
    assert 'id="latest-prob-down"' in html and 'id="latest-prob-neutral"' in html and 'id="latest-prob-up"' in html
    assert "renderLatestPrediction(predictionRows[0] || null, status)" in script
    selected_handler = script.split("function selectCandle(time)", 1)[1].split("function updateStatus", 1)[0]
    assert "renderLatestPrediction" not in selected_handler
    assert "predict_proba" not in script and "predict(" not in script


def test_predictions_api_supplies_latest_saved_record_pending_and_settled(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    opened, _ = session_bounds()

    def add_prediction(stamp):
        engine.store.insert_prediction({
            "candle_time_utc": stamp,
            "prediction_created_utc": stamp + timedelta(minutes=7),
            "prediction": "UP", "probability_down": .1, "probability_neutral": .2,
            "probability_up": .7, "confidence": .7, "reference_close": 101.25,
            "horizon_minutes": 15, "outcome_candle_time_utc": stamp + timedelta(minutes=15),
            "model_sha256": SPEC["source_candidate_sha256"],
            "input_features": {name: 0.0 for name in FEATURES}, "provider_delay_seconds": 0,
        })

    first, latest = opened + timedelta(minutes=5), opened + timedelta(minutes=10)
    add_prediction(first); add_prediction(latest)
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings())
    with TestClient(app, client=("127.0.0.1", 9070)) as client:
        response = client.get("/api/predictions?limit=1")
        assert response.status_code == 200
        rows = response.json()
        assert len(rows) == 1
        assert rows[0]["candle_time_utc"] == latest.isoformat(timespec="microseconds").replace("+00:00", "Z")
        assert rows[0]["probability_up"] == .7 and rows[0]["confidence"] == .7
        assert rows[0]["settlement_status"] == "PENDING"
        engine.store.insert_outcome({
            "prediction_candle_time_utc": latest, "observed_utc": latest + timedelta(minutes=16),
            "endpoint_candle_time_utc": latest + timedelta(minutes=15), "actual_target": "UP",
            "actual_return_pct": .2, "correct": True,
        })
        settled = client.get("/api/predictions?limit=1").json()[0]
        assert settled["settlement_status"] == "SETTLED"
        assert settled["actual_target"] == "UP" and bool(settled["correct"]) is True


def test_predictions_api_reports_empty_ledger_without_running_inference(tmp_path):
    engine, model, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     alpaca_provider_factory=lambda *_args: None)
    with TestClient(app, client=("127.0.0.1", 9071)) as client:
        response = client.get("/api/predictions?limit=1")
        status = client.get("/api/status").json()
    assert response.status_code == 200 and response.json() == []
    assert status["model_ready"] is True and status["feature_ready"] is False
    assert status["last_aggregation_status"] is None
    assert model.calls == 0


def test_status_keeps_feature_warmup_reason_separate_from_later_interruption(tmp_path):
    opened, _ = session_bounds()
    store = AlpacaBarStore(tmp_path / "diagnostic_status.sqlite3")
    model = AppModel()
    engine = AlpacaPredictionEngine(
        candidate=LoadedCandidate(model, SPEC, SPEC["source_candidate_sha256"]),
        store=store, now_fn=lambda: opened,
    )
    stamps = []
    for index in range(10):
        stamp = opened + timedelta(minutes=5 * index)
        stamps.append(stamp)
        store.insert_five_minute_candle(candle(stamp, 100 + index * .1))
    result = engine.predict_finalized_candle(candle(stamps[-1], 101.0),
                                             now=stamps[-1] + timedelta(minutes=7))
    assert result["status"] == "features_not_ready"
    engine._last_aggregation_status = "known_stream_interruption"
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     alpaca_provider_factory=lambda *_args: None)
    with TestClient(app, client=("127.0.0.1", 9072)) as client:
        status = client.get("/api/status").json()
        script = client.get("/static/alpaca.js").text
    attempt = status["latest_prediction_attempt"]
    assert status["last_aggregation_status"] == "known_stream_interruption"
    assert attempt["status"] == "feature_warmup_incomplete"
    assert attempt["reason"] == "consecutive_candle_warmup_insufficient"
    assert attempt["missing_features"]
    assert status["connected"] is False and status["provider_error"] is None
    diagnostic = script.split("function renderPipelineDiagnostics(status)", 1)[1].split("function renderLatestPrediction", 1)[0]
    assert "attempt.missing_features" in diagnostic
    assert "Latest aggregation rejected" in diagnostic
    assert "latest-attempt-status" in diagnostic and "latest-aggregation-status" in (ROOT / "live_app/static/alpaca.html").read_text(encoding="utf-8")
    attempt_rendering = diagnostic.split("const attempt = status.latest_prediction_attempt", 1)[1].split("el('latest-attempt-status').textContent", 1)[0]
    assert "last_aggregation_status" not in attempt_rendering


def test_alpaca_app_rejects_non_loopback_clients(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    app = create_app(alpaca_engine=engine, alpaca_settings=authorized_settings(False))
    with TestClient(app, client=("192.0.2.10", 9044)) as client:
        response = client.get("/api/status")
        assert response.status_code == 403
        assert response.json()["detail"] == "Dashboard is local-only"


def test_dashboard_rejects_non_loopback_clients_without_provider_fallback(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    app = create_app(run_poller=False, alpaca_engine=engine,
                     alpaca_settings=authorized_settings(False))
    with TestClient(app, client=("192.0.2.11", 9045)) as client:
        response = client.get("/")
        assert response.status_code == 403
        assert response.json()["detail"] == "Dashboard is local-only"


def test_live_dashboard_keeps_api_updates_when_chart_cdn_is_unavailable():
    html = (ROOT / "live_app/static/alpaca.html").read_text(encoding="utf-8")
    script = (ROOT / "live_app/static/alpaca.js").read_text(encoding="utf-8")
    assert 'id="chart-state"' in html
    assert "window.LightweightCharts?.createChart" in script
    assert "Candlestick chart unavailable" in script
    assert "'/api/market-timeline'" in script and "'/api/candles?limit=2000'" in script


def test_single_timeline_assets_and_chart_controls_are_available(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     alpaca_provider_factory=lambda *_args: None)
    with TestClient(app, client=("127.0.0.1", 9055)) as client:
        live_html = client.get("/").text
        replay_response = client.get("/replay", follow_redirects=False)
        helper = client.get("/static/alpaca_chart_viewport.js")
        script_asset = client.get("/static/alpaca.js")
        css_asset = client.get("/static/alpaca.css")
        live_script = client.get("/static/alpaca.js").text
        live_css = client.get("/static/alpaca.css").text
    assert helper.status_code == 200 and "function appendedRange" in helper.text
    assert script_asset.status_code == 200 and css_asset.status_code == 200
    assert live_html.index("alpaca_chart_viewport.js") < live_html.index("lightweight-charts")
    assert replay_response.status_code == 307 and replay_response.headers["location"] == "/"
    assert 'id="chart-reset"' in live_html and 'id="indicator-select"' in live_html
    assert 'id="warnings"' not in live_html and 'id="warning-count"' not in live_html
    assert '<option value="model">Model Indicator</option>' in live_html
    assert '<option value="sma" selected>SMA Indicator</option>' in live_html
    assert 'id="detail-open"' in live_html and 'id="latest-prediction-content"' in live_html
    assert "getVisibleLogicalRange" in live_script and "appendedRange" in live_script
    assert "new ResizeObserver" in live_script and "height: Math.max(1, root.clientHeight)" in live_script
    assert "fetch('/api/data-quality/warnings')" not in live_script
    assert "tool-button" in live_css and "#chart-wrap{height:clamp" in live_css


class StaticHistoricalMarket:
    def __init__(self, candles):
        self._candles = candles
        self.coverage = {"first_candle_utc": "2026-09-24T13:30:00Z", "last_candle_utc": "2026-09-24T19:55:00Z",
                         "historical_candle_count": len(candles), "live_candle_count": 0}

    def candles(self):
        return [dict(row) for row in self._candles]


def test_unified_timeline_merges_historical_and_live_without_duplicates_and_live_wins(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    live_rows = engine.store.dashboard_candles(100)
    overlap = live_rows[4]
    old_epoch = int(datetime(2026, 9, 24, 13, 30, tzinfo=UTC).timestamp())
    historical = StaticHistoricalMarket([
        {"time": old_epoch, "candle_time_utc": "2026-09-24T13:30:00Z", "open": 90, "high": 91,
         "low": 89, "close": 90.5, "volume": 100, "source": "historical", "live": False},
        {"time": int(datetime.fromisoformat(overlap["candle_time_utc"].replace("Z", "+00:00")).timestamp()),
         "candle_time_utc": overlap["candle_time_utc"], "open": 1, "high": 2, "low": .5, "close": 1.5,
         "volume": 1, "source": "historical", "live": False},
    ])
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     unified_historical_market=historical)
    with TestClient(app, client=("127.0.0.1", 9060)) as client:
        result = client.get("/api/market-timeline")
        assert result.status_code == 200
        body = result.json()
        times = [row["time"] for row in body["candles"]]
        assert times == sorted(times) and len(times) == len(set(times))
        assert len(body["candles"]) == len(live_rows) + 1
        selected = next(row for row in body["candles"] if row["time"] == old_epoch)
        assert selected["source"] == "historical" and selected["volume"] == 100
        merged_overlap = next(row for row in body["candles"] if row["time"] == int(datetime.fromisoformat(overlap["candle_time_utc"].replace("Z", "+00:00")).timestamp()))
        assert merged_overlap["source"] == "live" and merged_overlap["close"] == overlap["close"]
        assert body["coverage"]["historical_backfill_status"] == "not_present"


def test_unified_timeline_is_historical_only_without_persisted_live_candles(tmp_path):
    store = AlpacaBarStore(tmp_path / "empty_alpaca.sqlite3")
    engine = AlpacaPredictionEngine(candidate=LoadedCandidate(AppModel(), SPEC, SPEC["source_candidate_sha256"]),
                                    store=store, now_fn=lambda: session_bounds()[0])
    timestamp = int(datetime(2026, 9, 24, 13, 30, tzinfo=UTC).timestamp())
    historical = StaticHistoricalMarket([{"time": timestamp, "candle_time_utc": "2026-09-24T13:30:00Z",
        "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 15, "source": "historical", "live": False}])
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     unified_historical_market=historical)
    with TestClient(app, client=("127.0.0.1", 9063)) as client:
        body = client.get("/api/market-timeline").json()
    assert len(body["candles"]) == 1
    assert body["candles"][0]["source"] == "historical"
    assert body["coverage"]["live_candle_count"] == 0


def test_market_timeline_tail_returns_latest_chart_candles_with_full_coverage(tmp_path):
    store = AlpacaBarStore(tmp_path / "tail_alpaca.sqlite3")
    engine = AlpacaPredictionEngine(candidate=LoadedCandidate(AppModel(), SPEC, SPEC["source_candidate_sha256"]),
                                    store=store, now_fn=lambda: session_bounds()[0])
    rows = []
    for index in range(3):
        stamp = datetime(2026, 9, 24, 13, 30 + 5 * index, tzinfo=UTC)
        epoch = int(stamp.timestamp())
        rows.append({"time": epoch, "candle_time_utc": stamp.isoformat().replace("+00:00", "Z"),
                     "open": 100 + index, "high": 101 + index, "low": 99 + index,
                     "close": 100.5 + index, "volume": 20 + index,
                     "source": "historical", "feed": "IEX", "live": False})
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     unified_historical_market=StaticHistoricalMarket(rows))
    with TestClient(app, client=("127.0.0.1", 9065)) as client:
        response = client.get("/api/market-timeline?tail=2")
        invalid = client.get("/api/market-timeline?tail=2001")
    assert response.status_code == 200
    body = response.json()
    assert len(body["candles"]) == 2
    assert [row["time"] for row in body["candles"]] == [rows[-2]["time"], rows[-1]["time"]]
    assert body["coverage"]["timeline_candle_count"] == 3
    assert body["coverage"]["timeline_end_utc"] == rows[-1]["candle_time_utc"]
    assert invalid.status_code == 400


def test_market_timeline_api_exposes_verified_backfill_range_without_live_records(tmp_path, monkeypatch):
    import live_app.unified_market as unified_module
    from live_app.unified_market import UnifiedHistoricalMarket
    # This test asserts the backfill-only range; isolate it from the user's
    # separate append-only current IEX cache.
    monkeypatch.setattr(unified_module, "CURRENT_CACHE_CSV", tmp_path / "no-current-cache.csv")
    monkeypatch.setattr(unified_module, "CURRENT_CACHE_MANIFEST", tmp_path / "no-current-cache.json")
    import live_app.alpaca_rest_history as rest_history
    monkeypatch.setattr(rest_history, "CURRENT_CACHE_CSV", tmp_path / "no-current-cache.csv")
    monkeypatch.setattr(rest_history, "CURRENT_CACHE_MANIFEST", tmp_path / "no-current-cache.json")
    monkeypatch.setattr(unified_module, "load_verified_current_cache", lambda: [])
    store = AlpacaBarStore(tmp_path / "empty_backfill_api.sqlite3")
    engine = AlpacaPredictionEngine(candidate=LoadedCandidate(AppModel(), SPEC, SPEC["source_candidate_sha256"]),
                                    store=store, now_fn=lambda: session_bounds()[0])
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     unified_historical_market=UnifiedHistoricalMarket())
    with TestClient(app, client=("127.0.0.1", 9064)) as client:
        response = client.get("/api/market-timeline")
    assert response.status_code == 200
    body = response.json()
    timestamps = [row["time"] for row in body["candles"]]
    assert len(timestamps) == 18741 and len(timestamps) == len(set(timestamps))
    assert timestamps == sorted(timestamps)
    assert body["coverage"]["historical_backfill_status"] == "PASS"
    assert body["coverage"]["timeline_end_utc"].startswith("2026-11-12T20:55:00")
    assert body["coverage"]["live_candle_count"] == 0


def test_research_preview_log_is_selectable_by_candle_and_never_enters_production_ledger(tmp_path, monkeypatch):
    import live_app.research_preview_history as preview_log
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = datetime(2026, 11, 16, 16, 37, tzinfo=UTC)
            return fixed.astimezone(tz) if tz is not None else fixed.replace(tzinfo=None)

    # The fixture's 15-minute outcome is still pending at 16:37 UTC. Keep the
    # route's clock fixed so this test remains valid after that session passes.
    monkeypatch.setattr(app_module, "datetime", FrozenDateTime)
    stamp = "2026-11-16T16:25:00Z"
    preview_dir = tmp_path / "preview-runs"
    preview_dir.mkdir()
    monkeypatch.setattr(preview_log, "PREVIEW_RUN_DIR", preview_dir)
    monkeypatch.setattr(preview_log, "PREVIEW_DB_PATH", tmp_path / "research-preview.sqlite3")
    record = {
        "checked_at_utc": "2026-11-16T16:37:00Z", "candle_time_utc": stamp,
        "reference_price": 308.57, "models": [
            {"model_id": "xgboost_multiclass", "status": "preview", "prediction": "NEUTRAL",
             "signal": "HOLD", "probabilities": {"DOWN": .3, "NEUTRAL": .4, "UP": .3},
             "generated_at_utc": "2026-11-16T16:36:00Z", "horizon_minutes": 15,
             "model_sha256": "a" * 64, "persisted": False},
            {"model_id": "ridge_return", "status": "preview", "prediction": "DOWN", "signal": "SELL",
             "predicted_return_percentage_points": -.02, "generated_at_utc": "2026-11-16T16:36:00Z",
             "horizon_minutes": 15, "model_sha256": "b" * 64, "persisted": False},
            {"model_id": "xgboost_binary_v3", "status": "preview", "prediction": "UP", "signal": "BUY",
             "probabilities": {"DOWN": .49, "UP": .51}, "generated_at_utc": "2026-11-16T16:36:00Z",
             "horizon_minutes": 15, "model_sha256": "c" * 64, "persisted": False},
        ],
    }
    (preview_dir / "preview.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    stamp_epoch = int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp())
    historical = StaticHistoricalMarket([{"time": stamp_epoch, "candle_time_utc": stamp,
        "open": 308.5, "high": 309.0, "low": 308.0, "close": 308.57, "volume": 55,
        "source": "historical_current_cache", "feed": "IEX"}])
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     unified_historical_market=historical)
    with TestClient(app, client=("127.0.0.1", 9066)) as client:
        ledger_before = client.get("/api/predictions").json()
        selected = client.get("/api/research-model-previews-at-candle", params={"candle_time_utc": stamp})
        history = client.get("/api/research-model-preview-history?limit=20")
        ledger_after = client.get("/api/predictions").json()
    assert selected.status_code == 200
    assert [row["model_id"] for row in selected.json()["models"]] == [
        "ridge_return", "xgboost_binary_v3", "xgboost_multiclass"
    ]
    assert all(row["persisted_to_production_ledger"] is False for row in selected.json()["models"])
    assert all(row["outcome_status"] == "PENDING" for row in selected.json()["models"])
    assert history.status_code == 200 and history.json()["count"] == 3
    assert len(history.json()["records"]) == 3
    assert ledger_after == ledger_before == []


def test_research_preview_outcome_requires_completed_continuous_endpoint_and_stays_separate(tmp_path):
    from live_app.research_preview_history import ResearchPreviewStore
    store = ResearchPreviewStore(tmp_path / "research-only.sqlite3")
    opened, _ = session_bounds()
    decision = opened + timedelta(minutes=5)
    stamp = decision.isoformat(timespec="seconds").replace("+00:00", "Z")
    record = {"candle_time_utc": stamp, "checked_at_utc": opened.isoformat().replace("+00:00", "Z"),
        "reference_price": 100.0, "models": [{"model_id": "ridge_return", "status": "preview",
            "prediction": "UP", "signal": "BUY", "predicted_return_percentage_points": .2,
            "horizon_minutes": 15, "model_sha256": "d" * 64,
            "generated_at_utc": opened.isoformat().replace("+00:00", "Z")} ]}
    assert store.import_log_records([record], "fixture.jsonl") == 1
    close_by_minute = {0: 100.0, 5: 100.1, 10: 100.2, 15: 101.0}
    candles = []
    for minute, close in close_by_minute.items():
        ts = decision + timedelta(minutes=minute)
        candles.append({"time": int(ts.timestamp()), "candle_time_utc": ts.isoformat(timespec="seconds").replace("+00:00", "Z"),
                        "open": close, "high": close, "low": close, "close": close,
                        "source": "historical_current_cache", "feed": "IEX"})
    endpoint = decision + timedelta(minutes=15)
    assert store.settle_available(candles, now=endpoint + timedelta(minutes=6, seconds=59)) == 0
    assert store.rows(candle_time_utc=decision, now=endpoint + timedelta(minutes=6, seconds=59))[0]["outcome_status"] == "PENDING"
    assert store.settle_available(candles, now=endpoint + timedelta(minutes=7)) == 1
    settled = store.rows(candle_time_utc=decision, now=endpoint + timedelta(minutes=7))[0]
    assert settled["actual_target"] == "UP" and settled["correct"] == 1
    assert settled["actual_return_pct"] == pytest.approx(1.0)
    assert settled["persisted_to_production_ledger"] is False


def test_research_preview_log_rejects_naive_candle_timestamp(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings())
    with TestClient(app, client=("127.0.0.1", 9067)) as client:
        response = client.get("/api/research-model-previews-at-candle?candle_time_utc=2026-11-16T16%3A25%3A00")
    assert response.status_code == 400


def test_data_quality_warning_count_and_expanded_details_include_all_unresolved_buckets(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    opened, _ = session_bounds()
    detected = opened + timedelta(minutes=300)
    for index in range(12):
        bucket = opened + timedelta(minutes=200 + index * 5)
        engine.store.record_gap(bucket, detected + timedelta(seconds=index),
                                [bucket + timedelta(minutes=2)], reason="sparse_observed_minutes")
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     alpaca_provider_factory=lambda *_args: None)
    with TestClient(app, client=("127.0.0.1", 9065)) as client:
        status = client.get("/api/status").json()
        details = client.get("/api/data-quality/warnings").json()
    assert status["missing_data_warning_count"] == 12
    assert len(status["missing_data_warnings"]) == 10  # bounded status preview remains intact
    assert details["count"] == 12 and len(details["warnings"]) == 12
    assert all(item["reason"] == "sparse_observed_minutes" for item in details["warnings"])


def test_selected_candle_details_only_expose_recorded_pending_or_settled_outcomes(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    opened, _ = session_bounds()
    candle_time = opened + timedelta(minutes=50)
    endpoint = candle_time + timedelta(minutes=15)
    row = engine.store.five_minute_candle_at(candle_time)
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings())
    base = {"candle_time_utc": candle_time, "prediction_created_utc": candle_time + timedelta(minutes=5),
            "prediction": "UP", "probability_down": .1, "probability_neutral": .2, "probability_up": .7,
            "confidence": .7, "reference_close": row["close"], "horizon_minutes": 15,
            "outcome_candle_time_utc": endpoint, "model_sha256": SPEC["source_candidate_sha256"],
            "input_features": {name: 0.0 for name in FEATURES}, "provider_delay_seconds": 0.0}
    assert engine.store.insert_prediction(base)
    no_prediction_time = opened + timedelta(minutes=55)
    with TestClient(app, client=("127.0.0.1", 9061)) as client:
        no_record = client.get("/api/market-candle-details", params={"candle_time_utc": no_prediction_time.isoformat()}).json()
        assert no_record["status"] == "no_prediction" and no_record["message"] == "No prediction recorded"
        query = {"candle_time_utc": candle_time.isoformat()}
        pending = client.get("/api/market-candle-details", params=query).json()
        assert pending["status"] == "pending"
        assert pending["prediction"]["settlement_status"] == "PENDING"
        assert "actual_target" not in pending["prediction"] and "actual_close" not in pending["prediction"]
        endpoint_row = engine.store.five_minute_candle_at(endpoint)
        assert endpoint_row is not None
        assert engine.store.insert_outcome({"prediction_candle_time_utc": candle_time,
            "observed_utc": endpoint + timedelta(minutes=2), "endpoint_candle_time_utc": endpoint,
            "actual_target": "UP", "actual_return_pct": .5, "correct": True})
        settled = client.get("/api/market-candle-details", params=query).json()
        assert settled["status"] == "settled"
        assert settled["prediction"]["actual_target"] == "UP"
        assert settled["prediction"]["correct"] == 1
        assert settled["prediction"]["actual_close"] == pytest.approx(endpoint_row["close"])
        assert engine.store.prediction_count() == 1 and engine.store.outcome_count() == 1


def test_unified_timeline_requires_display_authorization(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(False))
    with TestClient(app, client=("127.0.0.1", 9062)) as client:
        assert client.get("/api/market-timeline").status_code == 403
        assert client.get("/api/market-candle-details?candle_time_utc=2026-09-24T13%3A30%3A00Z").status_code == 403


def test_safe_provider_auth_diagnostics_are_exposed_in_status_without_secrets(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])

    class FailedProvider:
        status = {
            "connected": False, "authenticated": False, "subscribed": False,
            "last_safe_error": "authentication_rejected",
            "last_error_category": "authentication_rejected",
            "last_error_transport": "websocket",
            "last_error_status_code": 402,
        }

    provider = FailedProvider()
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     alpaca_provider_factory=lambda *_args: provider)
    with TestClient(app, client=("127.0.0.1", 9051)) as client:
        status = client.get("/api/status").json()
        assert status["state"] == "authentication_rejected"
        assert status["credentials_configured"] is True
        assert status["last_error_category"] == "authentication_rejected"
        assert status["last_error_transport"] == "websocket"
        assert status["last_error_status_code"] == 402
        assert "status 402" in status["message"]
        assert "synthetic-secret" not in json.dumps(status)


def test_active_subscription_surfaces_safe_malformed_event_warning(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])

    class WarningProvider:
        status = {
            "connected": True, "authenticated": True, "subscribed": True,
            "last_safe_error": "malformed_json", "reconnect_count": 0,
        }

    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     alpaca_provider_factory=lambda *_args: WarningProvider())
    with TestClient(app, client=("127.0.0.1", 9053)) as client:
        status = client.get("/api/status").json()
        assert status["state"] == "connected_with_warning"
        assert "malformed_json" in status["message"]
        assert status["last_safe_error"] == "malformed_json"
        assert "synthetic-secret" not in json.dumps(status)


def test_status_keeps_transport_authentication_and_subscription_independent(tmp_path):
    engine, _, _ = build_engine(tmp_path, started_at=session_bounds()[0])

    class TransportOnlyProvider:
        status = {
            "connected": True, "authenticated": False, "subscribed": False,
            "authentication_success_observed": False, "subscription_success_observed": False,
            "authentication_success_count": 0, "subscription_success_count": 0,
        }

    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=authorized_settings(),
                     alpaca_provider_factory=lambda *_args: TransportOnlyProvider())
    with TestClient(app, client=("127.0.0.1", 9054)) as client:
        status = client.get("/api/status").json()
    assert status["state"] == "transport_connected"
    assert status["connected"] is True
    assert status["authenticated"] is False
    assert status["subscribed"] is False
    assert status["authentication_success_observed"] is False
    assert status["subscription_success_observed"] is False
