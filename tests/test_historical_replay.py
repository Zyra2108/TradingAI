from __future__ import annotations

import json
from datetime import date, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config import get_alpaca_settings
from live_app.alpaca_engine import AlpacaPredictionEngine
from live_app.alpaca_storage import AlpacaBarStore
from live_app.app import create_app
from live_app.calendar import NY, UTC, session_for_date
from live_app.historical_replay import DEFAULT_CANDLE_PATH, HistoricalReplay, _is_regular_session_candle
from live_app.model_loader import LoadedCandidate, load_alpaca_candidate
import live_app.historical_replay as replay_module

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "alpaca_final_freeze_v1"
SPEC = json.loads((MODEL_DIR / "frozen_candidate_spec.json").read_text(encoding="utf-8"))
FEATURES = SPEC["features_exact_order"]
CLASSES = ["DOWN", "NEUTRAL", "UP"]
DAY = date(2026, 9, 24)


class CaptureModel:
    def __init__(self):
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.classes_ = np.asarray(CLASSES, dtype=object)
        self.last_input = None

    def predict_proba(self, frame):
        self.last_input = frame.copy()
        return np.asarray([[0.15, 0.25, 0.60]])


def candidate(model=None):
    model = model or CaptureModel()
    return LoadedCandidate(model, SPEC, SPEC["source_candidate_sha256"])


def write_day(path: Path, *, missing: set[int] | None = None, after_change: float = 0.0,
              include_out_of_range: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    opened, closed = session_for_date(DAY)
    stamps = []
    current = opened
    while current < closed:
        stamps.append(current)
        current += timedelta(minutes=5)
    missing = missing or set()
    rows = []
    for index, stamp in enumerate(stamps):
        if index in missing:
            continue
        close = 100.0 + index * 0.015
        if index == 27:
            close += after_change
        opening = close - 0.006
        rows.append({
            "timestamp_utc": stamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "datetime": stamp.replace(tzinfo=None).isoformat(sep=" "),
            "open": opening, "high": close + 0.02, "low": opening - 0.02, "close": close,
        })
    if include_out_of_range:
        stamp = pd.Timestamp("2026-09-25T13:30:00Z")
        rows.append({"timestamp_utc": stamp.isoformat().replace("+00:00", "Z"),
                     "datetime": stamp.tz_convert("America/New_York").tz_localize(None).isoformat(sep=" "),
                     "open": 1, "high": 2, "low": .5, "close": 1.5})
    pd.DataFrame(rows).to_csv(path, index=False)
    return stamps


def make_replay(tmp_path, **kwargs):
    path = tmp_path / "historical.csv"
    stamps = write_day(path, **kwargs)
    return HistoricalReplay(candidate=candidate(), candle_path=path), stamps


def test_date_selection_and_candle_selection_are_limited_to_permitted_cache(tmp_path):
    replay, stamps = make_replay(tmp_path)
    assert replay.dates() == ["2026-09-24"]
    day_candles = replay.candles("2026-09-24")
    assert len(day_candles) == 78
    selected = day_candles[24]
    assert selected["candle_time_utc"] == stamps[24].astimezone(UTC).isoformat().replace("+00:00", "Z")
    with pytest.raises(ValueError, match="permitted"):
        replay.candles("2026-09-25")


def test_prediction_uses_frozen_order_real_probabilities_and_only_selected_prefix(tmp_path):
    replay, stamps = make_replay(tmp_path)
    selected = stamps[24].astimezone(UTC).isoformat().replace("+00:00", "Z")
    result = replay.predict("2026-09-24", selected)
    assert result["prediction"] == "UP"
    assert list(result["probabilities"]) == CLASSES
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["horizon_minutes"] == 15
    assert result["reference_close"] == pytest.approx(100 + 24 * .015)
    model = replay.candidate.model
    assert list(model.last_input.columns) == FEATURES
    assert len(model.last_input) == 1
    assert "volume" not in model.last_input.columns

    changed, _ = make_replay(tmp_path / "changed", after_change=50.0)
    changed_result = changed.predict("2026-09-24", selected)
    assert changed_result["probabilities"] == result["probabilities"]
    assert changed_result["prediction"] == result["prediction"]


def test_exact_15_minute_target_and_actual_label_are_separate(tmp_path):
    replay, stamps = make_replay(tmp_path, after_change=.4)
    selected = stamps[24].astimezone(UTC).isoformat().replace("+00:00", "Z")
    result = replay.predict("2026-09-24", selected)
    assert result["outcome"]["status"] == "available"
    assert result["outcome"]["target_time_utc"] == (stamps[27].astimezone(UTC)).isoformat().replace("+00:00", "Z")
    assert result["outcome"]["actual"] == "UP"
    assert result["outcome"]["correct"] is True


def test_missing_intervening_or_endpoint_candle_makes_outcome_unavailable(tmp_path):
    replay, stamps = make_replay(tmp_path, missing={26})
    selected = stamps[24].astimezone(UTC).isoformat().replace("+00:00", "Z")
    result = replay.predict("2026-09-24", selected)
    assert result["outcome"]["status"] == "unavailable"
    assert result["outcome"]["reason"] == "missing_or_discontinuous_candles"


def test_session_boundary_outcome_is_unavailable(tmp_path):
    replay, stamps = make_replay(tmp_path)
    selected = stamps[75].astimezone(UTC).isoformat().replace("+00:00", "Z")
    result = replay.predict("2026-09-24", selected)
    assert result["outcome"]["status"] == "unavailable"
    assert result["outcome"]["reason"] == "session_boundary"


def test_insufficient_session_warmup_is_reported_without_prediction(tmp_path):
    replay, stamps = make_replay(tmp_path)
    selected = stamps[0].astimezone(UTC).isoformat().replace("+00:00", "Z")
    result = replay.predict("2026-09-24", selected)
    assert result["status"] == "unavailable"
    assert result["reason"] == "insufficient_warmup"


def test_september_24_1400_reports_exact_features_missing_after_1355_gap():
    replay = HistoricalReplay()
    result = replay.predict("2026-09-24", "2026-09-24T18:00:00Z")
    assert result["status"] == "unavailable"
    assert result["missing_features"] == [
        "return_1", "return_3", "return_6", "return_12",
        "sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
        "ema_ratio_5", "ema_ratio_10", "ema_ratio_20", "rsi_14",
        "volatility_5", "volatility_10", "volatility_20",
    ]
    assert result["feature_context"]["continuous_candles"] == 1
    assert result["feature_context"]["reset_reason"] == "missing_5_minute_candle"
    assert "resets rolling features" in result["feature_context"]["warmup_rule"]


def test_gap_resets_features_and_a_later_continuous_segment_can_warm_up(tmp_path):
    replay, stamps = make_replay(tmp_path, missing={40})
    before_window_warm = replay.predict("2026-09-24", stamps[60].astimezone(UTC).isoformat().replace("+00:00", "Z"))
    after_window_warm = replay.predict("2026-09-24", stamps[61].astimezone(UTC).isoformat().replace("+00:00", "Z"))
    assert before_window_warm["status"] == "unavailable"
    assert before_window_warm["feature_context"]["reset_reason"] == "missing_5_minute_candle"
    assert after_window_warm["status"] == "ok"
    assert after_window_warm["prediction"] in CLASSES


def test_prior_sessions_do_not_supply_warmup_and_future_candles_are_not_feature_input(tmp_path, monkeypatch):
    rows = []
    for day in (date(2026, 9, 23), date(2026, 9, 24)):
        opened, closed = session_for_date(day)
        current = opened
        close = 100.0
        while current < closed:
            rows.append({"timestamp_utc":current.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                         "datetime":current.replace(tzinfo=None).isoformat(sep=" "),
                         "open":close, "high":close+.03, "low":close-.03, "close":close+.01})
            close += .01
            current += timedelta(minutes=5)
    path = tmp_path / "two_days.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    replay = HistoricalReplay(candidate=candidate(), candle_path=path)
    first_selected = replay.predict("2026-09-24", "2026-09-24T13:30:00Z")
    assert first_selected["status"] == "unavailable"
    assert first_selected["feature_context"]["continuous_candles"] == 1
    assert first_selected["feature_context"]["reset_reason"] == "trading_session_boundary"

    captured = {}
    original = replay_module.calculate_b_features
    def capture(ohlc, feature_order):
        captured["ohlc"] = ohlc.copy()
        return original(ohlc, feature_order)
    monkeypatch.setattr(replay_module, "calculate_b_features", capture)
    selected = "2026-09-24T15:30:00Z"  # 11:30 ET, after sufficient same-session warm-up
    result = replay.predict("2026-09-24", selected)
    assert result["status"] == "ok"
    assert captured["ohlc"].datetime.dt.date.nunique() == 1
    assert captured["ohlc"].datetime.dt.date.iloc[0] == date(2026, 9, 24)
    assert captured["ohlc"].datetime.max() == pd.Timestamp("2026-09-24 11:30:00")
    assert list(result["probabilities"]) == CLASSES


def test_exact_frozen_24_feature_order_is_used_for_ready_prediction(tmp_path):
    replay, stamps = make_replay(tmp_path)
    selected = stamps[30].astimezone(UTC).isoformat().replace("+00:00", "Z")
    result = replay.predict("2026-09-24", selected)
    assert result["status"] == "ok"
    assert list(replay.feature_order) == FEATURES
    assert list(replay.candidate.model.last_input.columns) == FEATURES


def test_cache_rejects_any_observation_on_or_after_test_boundary(tmp_path):
    path = tmp_path / "contaminated.csv"
    write_day(path, include_out_of_range=True)
    with pytest.raises(ValueError, match="outside the permitted pre-test period"):
        HistoricalReplay(candidate=candidate(), candle_path=path)


def test_regular_session_open_and_last_normal_five_minute_bar_boundaries():
    bounds = session_for_date(date(2026, 9, 24))
    assert bounds is not None
    assert _is_regular_session_candle(pd.Timestamp("2026-09-24 09:30:00"), bounds)
    assert _is_regular_session_candle(pd.Timestamp("2026-09-24 15:55:00"), bounds)
    assert not _is_regular_session_candle(pd.Timestamp("2026-09-24 16:00:00"), bounds)
    assert not _is_regular_session_candle(pd.Timestamp("2026-09-24 09:25:00"), bounds)


def test_early_close_uses_calendar_close_as_exclusive_candle_end():
    # Friday after Thanksgiving, NYSE regular session closes at 13:00 ET.
    bounds = session_for_date(date(2026, 11, 27))
    assert bounds is not None
    assert bounds[1].hour == 13 and bounds[1].minute == 0
    assert _is_regular_session_candle(pd.Timestamp("2026-11-27 12:55:00"), bounds)
    assert not _is_regular_session_candle(pd.Timestamp("2026-11-27 13:00:00"), bounds)


def test_utc_and_new_york_timestamp_represent_same_valid_close_candle():
    utc_stamp = pd.Timestamp("2026-09-24T19:55:00Z")
    ny_stamp = utc_stamp.tz_convert("America/New_York")
    bounds = session_for_date(date(2026, 9, 24))
    assert ny_stamp.strftime("%Y-%m-%d %H:%M:%S %z") == "2026-09-24 15:55:00 -0400"
    assert _is_regular_session_candle(utc_stamp, bounds)
    assert _is_regular_session_candle(ny_stamp, bounds)


def test_actual_replay_cache_excludes_only_the_known_0925_extended_hours_row(tmp_path, monkeypatch):
    assert DEFAULT_CANDLE_PATH.is_absolute()
    assert DEFAULT_CANDLE_PATH.is_file()
    monkeypatch.chdir(tmp_path)
    replay = HistoricalReplay(candidate=candidate())
    assert replay._bars.attrs["excluded_out_of_session_rows"] == 1
    assert replay.dates()[0] == "2026-08-06"
    assert replay.dates()[-1] == "2026-09-24"
    assert len(replay.candles("2026-09-24")) > 0
    assert not any(row["candle_time_et"].startswith("2026-09-18 09:25:00") for row in replay.candles("2026-09-18"))
    assert any(row["candle_time_et"].startswith("2026-09-24 15:55:00") for row in replay.candles("2026-09-24"))


def test_actual_cache_full_api_replay_workflow(tmp_path):
    loaded = load_alpaca_candidate(MODEL_DIR / "candidate_pipeline.joblib", MODEL_DIR / "frozen_candidate_spec.json")
    replay = HistoricalReplay(candidate=loaded)
    store = AlpacaBarStore(tmp_path / "api_alpaca.sqlite3")
    engine = AlpacaPredictionEngine(candidate=loaded, store=store,
                                    now_fn=lambda: pd.Timestamp("2026-11-13T00:00:00Z").to_pydatetime())
    settings = get_alpaca_settings({"APCA_API_KEY_ID":"synthetic", "APCA_API_SECRET_KEY":"synthetic",
                                   "ALPACA_IEX_DISPLAY_AUTHORIZED":"true"})
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=settings, historical_replay=replay)
    with TestClient(app, client=("127.0.0.1", 9052)) as client:
        page = client.get("/replay", follow_redirects=False)
        dates_response = client.get("/api/replay/dates")
        assert page.status_code == 307 and page.headers["location"] == "/"
        dates = dates_response.json()["dates"]
        assert dates_response.status_code == 200 and dates[0] == "2026-08-06" and dates[-1] == "2026-09-24"
        candle_response = client.get("/api/replay/candles", params={"day":"2026-09-24"})
        candles = candle_response.json()
        assert candle_response.status_code == 200 and candles
        selected = next(row for row in candles if row["candle_time_et"].startswith("2026-09-24 11:30:00"))
        prediction_response = client.get("/api/replay/predict", params={
            "day":"2026-09-24", "candle_time":selected["candle_time_utc"]
        })
        assert prediction_response.status_code == 410
        assert store.prediction_count() == 0


def test_frozen_integrity_loader_and_feature_contract_are_used():
    loaded = load_alpaca_candidate(MODEL_DIR / "candidate_pipeline.joblib", MODEL_DIR / "frozen_candidate_spec.json")
    assert loaded.sha256 == SPEC["source_candidate_sha256"]
    assert list(loaded.model.feature_names_in_) == FEATURES
    assert [str(value) for value in loaded.model.classes_] == CLASSES


def test_integrity_loaded_frozen_model_generates_real_probabilities_on_synthetic_ohlc(tmp_path):
    loaded = load_alpaca_candidate(MODEL_DIR / "candidate_pipeline.joblib", MODEL_DIR / "frozen_candidate_spec.json")
    path = tmp_path / "synthetic_ohlc.csv"
    stamps = write_day(path)
    replay = HistoricalReplay(candidate=loaded, candle_path=path)
    selected = stamps[24].astimezone(UTC).isoformat().replace("+00:00", "Z")
    result = replay.predict("2026-09-24", selected)
    assert result["status"] == "ok"
    assert list(result["probabilities"]) == CLASSES
    assert all(np.isfinite(list(result["probabilities"].values())))
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_historical_replay_is_auth_gated_and_does_not_write_live_prediction_tables(tmp_path):
    loaded = candidate()
    store = AlpacaBarStore(tmp_path / "alpaca.sqlite3")
    engine = AlpacaPredictionEngine(candidate=loaded, store=store, now_fn=lambda: pd.Timestamp("2026-11-13T00:00:00Z").to_pydatetime())
    path = tmp_path / "history.csv"
    stamps = write_day(path)
    replay = HistoricalReplay(candidate=loaded, candle_path=path)
    settings = get_alpaca_settings({"APCA_API_KEY_ID":"synthetic", "APCA_API_SECRET_KEY":"synthetic",
                                   "ALPACA_IEX_DISPLAY_AUTHORIZED":"true"})
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=settings, historical_replay=replay)
    before = store.prediction_count()
    with TestClient(app, client=("127.0.0.1", 9048)) as client:
        assert client.get("/api/replay/dates").json()["dates"] == ["2026-09-24"]
        candles_response = client.get("/api/replay/candles?day=2026-09-24")
        assert candles_response.status_code == 200
        assert len(candles_response.json()) == 78
        stamp = stamps[24].astimezone(UTC).isoformat().replace("+00:00", "Z")
        assert client.get(f"/api/replay/predict?day=2026-09-24&candle_time={stamp}").status_code == 410
        page_response = client.get("/replay", follow_redirects=False)
        assert page_response.status_code == 307 and page_response.headers["location"] == "/"
        main_page = client.get("/").text
        assert "Market Timeline" in main_page and "Historical Replay" not in main_page
        assert "/static/alpaca.js" in main_page and "/static/alpaca.css" in main_page
    assert store.prediction_count() == before

    disabled = get_alpaca_settings({"ALPACA_IEX_DISPLAY_AUTHORIZED":"false"})
    blocked = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=disabled, historical_replay=replay)
    with TestClient(blocked, client=("127.0.0.1", 9049)) as client:
        assert client.get("/api/replay/dates").status_code == 403


def test_live_dashboard_route_remains_available_without_separate_replay_navigation(tmp_path):
    store = AlpacaBarStore(tmp_path / "alpaca.sqlite3")
    engine = AlpacaPredictionEngine(candidate=candidate(), store=store, now_fn=lambda: pd.Timestamp("2026-11-13T00:00:00Z").to_pydatetime())
    settings = get_alpaca_settings({"ALPACA_IEX_DISPLAY_AUTHORIZED":"true"})
    app = create_app(run_poller=False, alpaca_engine=engine, alpaca_settings=settings, historical_replay=object())
    with TestClient(app, client=("127.0.0.1", 9050)) as client:
        live_page = client.get("/").text
        assert "Market Timeline" in live_page
        assert "Historical Replay" not in live_page
        assert client.get("/api/status").status_code == 200
