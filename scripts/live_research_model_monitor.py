"""Poll the local Alpaca IEX timeline and log separate research model previews.

This script never connects to Alpaca directly and never writes the production
prediction/outcome ledger. The FastAPI app owns bounded Alpaca IEX REST catch-up.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import pathlib
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
MAX_LIVE_CANDLE_AGE = dt.timedelta(minutes=15)
MODEL_IDS = ("xgboost_multiclass", "ridge_return", "xgboost_binary_v3")
PREVIEW_FIELDS = (
    "model_id", "status", "prediction", "signal", "probabilities",
    "predicted_return_percentage_points", "candle_time_utc", "generated_at_utc",
    "reference_price", "horizon_minutes", "persisted", "model_sha256",
    "missing_features", "message",
)


def _is_recent_current_session_candle(stamp: str, *, now: dt.datetime | None = None) -> bool:
    """Reject stale/previous-session bars so a fresh run cannot backfill previews."""
    try:
        candle_time = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return False
    if candle_time.tzinfo is None or candle_time.utcoffset() is None:
        return False
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        return False
    candle_time = candle_time.astimezone(NY)
    now = now.astimezone(NY)
    age = now - candle_time
    return candle_time.date() == now.date() and dt.timedelta(0) <= age <= MAX_LIVE_CANDLE_AGE


def _get_json(base: str, path: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("API response was not an object")
    return payload


def _read_completed_stamps(path: pathlib.Path) -> set[str]:
    completed = set()
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            models = record.get("models", [])
            statuses = {item.get("model_id"): item.get("status") for item in models if isinstance(item, dict)}
            if all(statuses.get(model_id) in {"preview", "insufficient_features", "features_unavailable"}
                   for model_id in MODEL_IDS):
                completed.add(record.get("candle_time_utc"))
        except (json.JSONDecodeError, OSError, AttributeError):
            continue
    return completed


def _request_preview(base: str, model_id: str) -> dict:
    try:
        result = _get_json(base, "/api/research-model-prediction?model_id=" + model_id, timeout=120)
        return {field: result.get(field) for field in PREVIEW_FIELDS}
    except Exception as exc:
        return {"model_id": model_id, "status": "request_error", "error_type": type(exc).__name__}


def run(*, base: str, log_path: pathlib.Path, poll_seconds: int = 20) -> None:
    try:
        app_status = _get_json(base, "/api/status", timeout=20)
    except Exception:
        print(json.dumps({"monitor": "not_started", "reason": "local_app_unavailable"}), flush=True)
        return
    if app_status.get("research_preview_recording_enabled"):
        print(json.dumps({"monitor": "not_started", "reason": "recording_managed_by_application"}), flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = _read_completed_stamps(log_path)
    retry_at: dict[str, float] = {}
    stop_at = dt.datetime.combine(dt.datetime.now(NY).date(), dt.time(16, 0), tzinfo=NY)
    print(json.dumps({"monitor": "started", "log": str(log_path), "completed_candles_loaded": len(done)}), flush=True)
    while dt.datetime.now(NY) < stop_at:
        try:
            timeline = _get_json(base, "/api/market-timeline", timeout=45)
            candles = timeline.get("candles") or []
            if candles:
                latest = candles[-1]
                stamp = latest.get("candle_time_utc")
                if (stamp and _is_recent_current_session_candle(stamp)
                        and stamp not in done and time.monotonic() >= retry_at.get(stamp, 0)):
                    status = _get_json(base, "/api/status", timeout=20)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                        results = list(pool.map(lambda model: _request_preview(base, model), MODEL_IDS))
                    record = {
                        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "candle_time_utc": stamp, "reference_price": latest.get("close"),
                        "provider_state": status.get("state"), "authenticated": status.get("authenticated"),
                        "subscribed": status.get("subscribed"), "source": "Alpaca IEX current timeline",
                        "models": results,
                    }
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    failed = any(item.get("status") == "request_error" for item in results)
                    if failed:
                        retry_at[stamp] = time.monotonic() + 60
                        print(json.dumps({"candle": stamp, "status": "retry_pending",
                                          "model_statuses": [item.get("status") for item in results]}), flush=True)
                    else:
                        done.add(stamp)
                        retry_at.pop(stamp, None)
                        print(json.dumps({
                            "candle": stamp, "provider": status.get("state"),
                            "models": [{"id": item.get("model_id"), "status": item.get("status"),
                                        "prediction": item.get("prediction"),
                                        "probabilities": item.get("probabilities"),
                                        "return": item.get("predicted_return_percentage_points"),
                                        "missing": item.get("missing_features")} for item in results],
                        }, separators=(",", ":")), flush=True)
        except Exception as exc:
            print(json.dumps({"poll_error_type": type(exc).__name__,
                              "at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}), flush=True)
        time.sleep(poll_seconds)
    print(json.dumps({"monitor": "stopped_at_ny_close", "time": dt.datetime.now(NY).isoformat(timespec="seconds")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--log", required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 5 and 60")
    run(base=args.base, log_path=pathlib.Path(args.log), poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
