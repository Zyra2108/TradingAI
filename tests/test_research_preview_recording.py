from __future__ import annotations

import json
from datetime import datetime, timezone

from live_app.app import _eligible_research_preview_candles
from live_app.research_preview_history import MODEL_IDS, ResearchPreviewRecorder

UTC = timezone.utc


def _model_result(model_id: str) -> dict:
    prediction = "NEUTRAL" if model_id == "xgboost_multiclass" else "UP"
    return {
        "model_id": model_id, "status": "preview", "prediction": prediction,
        "signal": "HOLD" if prediction == "NEUTRAL" else "BUY",
        "probabilities": ({"DOWN": 0.2, "NEUTRAL": 0.6, "UP": 0.2}
                          if model_id != "ridge_return" else None),
        "predicted_return_percentage_points": 0.02 if model_id == "ridge_return" else None,
        "generated_at_utc": "2026-11-17T15:08:00Z", "reference_price": 300.0,
        "horizon_minutes": 15, "model_sha256": "a" * 64,
    }


def test_preview_recorder_appends_idempotently_across_restart(tmp_path):
    started = datetime(2026, 11, 17, 15, 8, tzinfo=UTC)
    stamp = "2026-11-17T15:00:00Z"
    record = {
        "checked_at_utc": "2026-11-17T15:08:00Z", "candle_time_utc": stamp,
        "reference_price": 300.0, "source": "verified Alpaca IEX finalized 5-minute candle",
        "models": [_model_result(model_id) for model_id in MODEL_IDS],
    }
    recorder = ResearchPreviewRecorder(tmp_path, started_at=started)
    assert recorder.append(record) is True
    assert recorder.append(record) is False
    restarted = ResearchPreviewRecorder(tmp_path, started_at=started)
    assert restarted.is_completed(stamp)
    lines = (tmp_path / "20261117T150800Z_app.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert [row["model_id"] for row in saved["models"]] == list(MODEL_IDS)
    assert saved["source"].startswith("verified Alpaca IEX")


def test_incomplete_model_error_can_retry_without_marking_candle_done(tmp_path):
    recorder = ResearchPreviewRecorder(tmp_path,
        started_at=datetime(2026, 11, 17, 15, 8, tzinfo=UTC))
    record = {
        "candle_time_utc": "2026-11-17T15:00:00Z",
        "models": [_model_result(model_id) for model_id in MODEL_IDS],
    }
    record["models"][-1] = {"model_id": MODEL_IDS[-1], "status": "model_error"}
    assert recorder.append(record) is True
    assert not recorder.is_completed(record["candle_time_utc"])


def test_eligibility_requires_recent_finalized_same_session_iex_candle(tmp_path):
    recorder = ResearchPreviewRecorder(tmp_path,
        started_at=datetime(2026, 11, 17, 15, 8, tzinfo=UTC))
    now = datetime(2026, 11, 17, 15, 8, tzinfo=UTC)
    candles = [
        {"time": int(datetime(2026, 11, 17, 15, 0, tzinfo=UTC).timestamp()),
         "candle_time_utc": "2026-11-17T15:00:00Z", "feed": "IEX"},
        {"time": int(datetime(2026, 11, 17, 14, 40, tzinfo=UTC).timestamp()),
         "candle_time_utc": "2026-11-17T14:40:00Z", "feed": "IEX"},  # stale
        {"time": int(datetime(2026, 11, 17, 15, 5, tzinfo=UTC).timestamp()),
         "candle_time_utc": "2026-11-17T15:05:00Z", "feed": "IEX"},  # buffer not elapsed
        {"time": int(datetime(2026, 11, 17, 20, 45, tzinfo=UTC).timestamp()),
         "candle_time_utc": "2026-11-17T20:45:00Z", "feed": "IEX"},  # horizon past close
        {"time": int(datetime(2026, 11, 17, 15, 0, tzinfo=UTC).timestamp()),
         "candle_time_utc": "2026-11-17T15:00:00Z", "feed": "NASDAQ"},
    ]
    eligible = _eligible_research_preview_candles(candles, now=now, recorder=recorder)
    assert [item[1]["candle_time_utc"] for item in eligible] == ["2026-11-17T15:00:00Z"]
    late = _eligible_research_preview_candles(
        [candles[3]], now=datetime(2026, 11, 17, 20, 53, tzinfo=UTC), recorder=recorder)
    assert late == []


def test_preview_recorder_waits_for_all_three_model_statuses(tmp_path):
    recorder = ResearchPreviewRecorder(tmp_path,
        started_at=datetime(2026, 11, 17, 15, 8, tzinfo=UTC))
    stamp = "2026-11-17T15:00:00Z"
    record = {"candle_time_utc": stamp,
              "models": [_model_result(model_id) for model_id in MODEL_IDS]}
    record["models"][0] = {"model_id": MODEL_IDS[0], "status": "insufficient_features"}
    assert recorder.append(record)
    assert recorder.is_completed(stamp)

