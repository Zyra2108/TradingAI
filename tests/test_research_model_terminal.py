from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config import get_alpaca_settings
from live_app.app import create_app
from live_app.research_models import FEATURES, MODEL_SPECS, ResearchModelService

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def synthetic_candles(count=32):
    start = datetime(2026, 11, 13, 14, 30, tzinfo=UTC)
    rows = []
    for index in range(count):
        close = 276 + index * 0.025 + (0.04 if index % 4 == 0 else -0.015)
        rows.append({
            "time": int((start + timedelta(minutes=5 * index)).timestamp()),
            "candle_time_utc": (start + timedelta(minutes=5 * index)).isoformat().replace("+00:00", "Z"),
            "open": close - 0.025, "high": close + 0.08,
            "low": close - 0.09, "close": close, "volume": 1000 + index,
        })
    return rows


def test_catalog_uses_saved_oof_and_validation_reports_with_clear_test_disclosure():
    catalog = ResearchModelService().catalog()
    assert [model["id"] for model in catalog["models"]] == list(MODEL_SPECS)
    assert catalog["baselines"]["always_neutral_accuracy_all_rows"] == pytest.approx(623 / 1161)
    assert "not a pristine holdout" in catalog["evaluation_disclosure"]["validation"]
    assert "Previously exposed" in catalog["evaluation_disclosure"]["test_diagnostics"]
    assert catalog["models"][0]["evaluation"]["validation"]["confusion_matrix"] == [[69, 115, 93], [137, 314, 172], [75, 95, 91]]
    binary = catalog["models"][2]["evaluation"]["validation"]
    assert binary["n"] == 538
    assert binary["confusion_matrix"] == [[169, 108], [176, 85]]
    assert binary["up_recall"] == pytest.approx(85 / 261)


def test_all_three_existing_artifacts_pass_hash_and_feature_contract_checks():
    service = ResearchModelService()
    for model_id in MODEL_SPECS:
        model = service._model(model_id)
        assert list(model.feature_names_in_) == FEATURES


def test_real_saved_models_predict_from_completed_synthetic_candles_without_persistence():
    service = ResearchModelService()
    rows = synthetic_candles()
    for model_id in MODEL_SPECS:
        result = service.predict_latest(rows, model_id)
        assert result["status"] == "preview"
        assert result["persisted"] is False
        assert result["horizon_minutes"] == 15
        assert result["candle_time_epoch"] == rows[-1]["time"]
        assert result["reference_price"] == rows[-1]["close"]
        assert result["prediction"] in {"DOWN", "NEUTRAL", "UP"}
        assert result["signal"] in {"BUY", "SELL", "HOLD"}
        if result["probabilities"] is not None:
            assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
        if model_id == "ridge_return":
            assert result["probabilities"] is None
            assert result["predicted_return_percentage_points"] is not None
        if model_id == "xgboost_binary_v3":
            assert set(result["probabilities"]) == {"DOWN", "UP"}
            assert result["prediction"] != "NEUTRAL"


def test_preview_is_cached_for_same_completed_candle_and_recomputes_after_append():
    class FakeModel:
        feature_names_in_ = np.asarray(FEATURES, dtype=object)
        classes_ = np.asarray([0, 1, 2])
        def __init__(self): self.calls = 0
        def predict_proba(self, _frame):
            self.calls += 1
            return np.asarray([[0.2, 0.3, 0.5]])

    model = FakeModel()
    service = ResearchModelService(model_loader=lambda _path: model)
    # Use a feature builder with exact names/order and one available latest row.
    def feature_builder(ohlc, _feature_order):
        rows = [{"datetime": stamp, **{name: 1.0 for name in FEATURES}} for stamp in ohlc["datetime"]]
        return pd.DataFrame(rows)
    service.feature_builder = feature_builder
    service._model = lambda _model_id: model
    candles = synthetic_candles(1)
    first = service.predict_latest(candles, "xgboost_multiclass")
    again = service.predict_latest(candles, "xgboost_multiclass")
    assert first == again and model.calls == 1
    appended = candles + synthetic_candles(2)[-1:]
    next_result = service.predict_latest(appended, "xgboost_multiclass")
    assert next_result["candle_time_epoch"] > first["candle_time_epoch"]
    assert model.calls == 2


def test_three_model_previews_share_one_causal_feature_build_per_completed_candle():
    calls = []
    class Model:
        def __init__(self, kind):
            self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
            self.classes_ = np.asarray([0, 1, 2] if kind == "multiclass" else [0, 1])
            self.kind = kind
        def predict_proba(self, _frame):
            return np.asarray([[.2, .3, .5]]) if self.kind == "multiclass" else np.asarray([[.4, .6]])
        def predict(self, _frame): return np.asarray([-.02])
    models = {model_id: Model(spec["kind"]) for model_id, spec in MODEL_SPECS.items()}
    def builder(ohlc, _order):
        calls.append(len(ohlc))
        return pd.DataFrame([{"datetime": stamp, **{name: 1.0 for name in FEATURES}}
                             for stamp in ohlc["datetime"]])
    service = ResearchModelService(feature_builder=builder)
    service._model = lambda model_id: models[model_id]
    candles = synthetic_candles(24)
    outputs = [service.predict_latest(candles, model_id) for model_id in MODEL_SPECS]
    assert all(row["status"] == "preview" for row in outputs)
    assert all(row["persisted"] is False for row in outputs)
    assert len(calls) == 1
    service.predict_latest(candles, "xgboost_multiclass")
    assert len(calls) == 1
    service.predict_latest(candles + synthetic_candles(25)[-1:], "xgboost_multiclass")
    assert len(calls) == 2


def test_unwarmed_latest_candle_returns_missing_features_without_model_inference():
    class NeverModel:
        feature_names_in_ = np.asarray(FEATURES, dtype=object)
        classes_ = np.asarray([0, 1, 2])
    service = ResearchModelService(model_loader=lambda _path: pytest.fail("must not load an unneeded model"))
    service.feature_builder = lambda ohlc, _order: pd.DataFrame([{"datetime": ohlc.iloc[-1]["datetime"], **{name: np.nan for name in FEATURES}}])
    result = service.predict_latest(synthetic_candles(1), "xgboost_multiclass")
    assert result["status"] == "insufficient_features"
    assert set(result["missing_features"]) == set(FEATURES)


class FakeStore:
    def dashboard_candles(self, _limit): return []


class FakeEngine:
    feature_order = FEATURES
    model_sha256 = "production-hash-test-only"
    store = FakeStore()
    def recover_aggregations(self): return []
    def settle_due_predictions(self): return []
    def status(self):
        return {"model_ready": True, "feature_ready": True, "pending_settlements": 0,
                "late_updated_bar_count": 0, "open_stream_interruption_count": 0}


class FakeMarket:
    coverage = {"historical_candle_count": 1, "live_candle_count": 0}
    def candles(self): return synthetic_candles(1)


class FakeResearchService:
    def catalog(self): return {"models": [], "evaluation_disclosure": {}}
    def predict_latest(self, candles, model_id):
        return {"status": "preview", "model_id": model_id, "prediction": "UP",
                "signal": "BUY", "persisted": False, "horizon_minutes": 15,
                "input_candle_time": candles[-1]["time"]}


def test_research_model_api_is_authorized_and_uses_separate_non_persisting_service():
    settings = get_alpaca_settings({"ALPACA_IEX_DISPLAY_AUTHORIZED": "true"})
    app = create_app(run_poller=False, alpaca_engine=FakeEngine(), alpaca_settings=settings,
                     unified_historical_market=FakeMarket(), research_model_service=FakeResearchService())
    with TestClient(app, client=("127.0.0.1", 9047)) as client:
        catalog = client.get("/api/research-models")
        preview = client.get("/api/research-model-prediction?model_id=xgboost_multiclass")
    assert catalog.status_code == 200
    assert preview.status_code == 200
    assert preview.json()["persisted"] is False


def test_research_model_api_respects_display_authorization_gate():
    settings = get_alpaca_settings({"ALPACA_IEX_DISPLAY_AUTHORIZED": "false"})
    app = create_app(run_poller=False, alpaca_engine=FakeEngine(), alpaca_settings=settings,
                     unified_historical_market=FakeMarket(), research_model_service=FakeResearchService())
    with TestClient(app, client=("127.0.0.1", 9048)) as client:
        assert client.get("/api/research-models").status_code == 403
        assert client.get("/api/research-model-prediction?model_id=xgboost_multiclass").status_code == 403


def test_frontend_uses_saved_research_predictions_for_panel_and_chart_markers():
    html = (ROOT / "live_app/static/alpaca.html").read_text(encoding="utf-8")
    js = (ROOT / "live_app/static/alpaca.js").read_text(encoding="utf-8")
    assert all(model_id in html for model_id in MODEL_SPECS)
    assert "renderSelectedSavedResearchPreview" in js
    assert "const modelLabel" in js  # marker text identifies the saved model selected in the panel
    assert "api/research-model-prediction" not in js
    assert "savedResearchPreviews" in js and "research_preview" in js
    assert "No saved prediction for this model yet" in js
    assert "historical-model-signals" in js and "api/predictions" in js
    assert "setMarkers(markers)" in js
    assert "prediction-history-body" in html
    assert "api/report.csv" in html
    assert 'href="/api/research-model-preview-report.csv"' in html
    assert 'href="/api/report.csv"' in html
    assert "APCA_API_SECRET_KEY" not in html + js
    assert "LATEST WS EVENT" in html
    assert "WebSocket event delay" in js
    assert "Outcomes appear only after the 15-minute endpoint and required real candles" in html


def test_research_preview_export_contains_saved_rows_without_touching_production_ledger(monkeypatch):
    import live_app.research_preview_history as preview_module

    record = {
        "candle_time_utc": "2026-11-16T17:35:00Z", "generated_at_utc": "2026-11-16T17:36:00Z",
        "model_id": "xgboost_multiclass", "model_name": "XGBoost Multiclass",
        "prediction": "NEUTRAL", "signal": "HOLD",
        "probabilities": {"DOWN": 0.15, "NEUTRAL": 0.63, "UP": 0.22},
        "predicted_return_percentage_points": None, "reference_price": 308.41,
        "horizon_minutes": 15, "outcome_status": "PENDING",
        "endpoint_candle_time_utc": "2026-11-16T17:50:00Z", "actual_target": None,
        "actual_return_pct": None, "actual_close": None, "correct": None,
        "model_sha256": "model-hash", "source": "research_preview",
        "persisted_to_production_ledger": False,
    }
    calls = []
    records = [{**record, "candle_time_utc": f"2026-11-16T17:{index // 60:02d}:{index % 60:02d}Z"}
               for index in range(1001)]
    monkeypatch.setattr(preview_module, "all_preview_records", lambda **kwargs: (
        calls.append(kwargs) or records
    ))
    settings = get_alpaca_settings({"ALPACA_IEX_DISPLAY_AUTHORIZED": "true"})
    app = create_app(run_poller=False, alpaca_engine=FakeEngine(), alpaca_settings=settings,
                     unified_historical_market=FakeMarket())

    with TestClient(app, client=("127.0.0.1", 9049)) as client:
        response = client.get("/api/research-model-preview-report.csv")

    assert response.status_code == 200
    assert "ecl_alpaca_iex_research_preview_report.csv" in response.headers["content-disposition"]
    assert "probability_down,probability_neutral,probability_up" in response.text
    assert "0.15,0.63,0.22" in response.text
    assert "research_preview,False" in response.text
    assert calls
    assert response.text.count("\n") == 1002  # header plus every saved row, beyond the UI's 300-row view
    assert "production-hash-test-only" not in response.text


def test_empty_production_prediction_panel_is_hidden_without_hiding_research_preview():
    html = (ROOT / "live_app/static/alpaca.html").read_text(encoding="utf-8")
    script = (ROOT / "live_app/static/alpaca.js").read_text(encoding="utf-8")
    assert 'id="production-panel" class="panel production-panel hidden"' in html
    renderer = script.split("function renderLatestPrediction", 1)[1].split("function renderLatestPrediction", 1)[0]
    assert "productionPanel.classList.toggle('hidden', !prediction)" in renderer
    assert 'id="research-preview-history-body"' in html
