"""Capture evidence from the integrated localhost Alpaca IEX application.

The script never starts/stops the server, makes provider requests, or writes to
SQLite. It attaches only to 127.0.0.1 after an explicit CLI confirmation and
only during a regular NYSE session. The running application remains responsible
for all real provider traffic and persistence.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Running this file as ``python scripts/...`` places only the scripts
    # directory on sys.path; import the project configuration from its root.
    sys.path.insert(0, str(ROOT))
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "live_market_verification_runs"
EXPECTED_MODEL_RELATIVE = Path("models/alpaca_final_freeze_v1/candidate_pipeline.joblib")
EXPECTED_FEATURES = None
EXPECTED_CLASSES = ["DOWN", "NEUTRAL", "UP"]
EXPECTED_HORIZON = 15


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def ensure_loopback_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port not in (None, 8000):
        raise ValueError("Verification accepts only http://127.0.0.1:8000")
    return base_url.rstrip("/")


def market_window(now: datetime) -> tuple[bool, str, datetime | None]:
    from live_app.calendar import NY, session_for_date
    local = now.astimezone(NY)
    session = session_for_date(local.date())
    if session is None:
        return False, "NYSE closed: holiday or weekend", None
    opened, closed = session
    if local < opened:
        return False, "NYSE closed: pre-open", closed
    if local >= closed:
        return False, "NYSE closed: after close", closed
    if closed - local < timedelta(minutes=25):
        return False, "Insufficient session time remains for a complete prediction and settlement", closed
    return True, "NYSE regular session open", closed


def _get_json(url: str, timeout: float = 6.0):
    request = __import__("urllib.request", fromlist=["Request"]).Request(
        url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_text(url: str, timeout: float = 6.0):
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _read_only_prediction(db_path: Path, candle_time_utc: str, feature_order: list[str],
                          capture_start: datetime) -> dict:
    """Verify an actual row using a read-only SQLite connection."""
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=3) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("""SELECT p.candle_time_utc,p.prediction_created_utc,p.prediction,
            p.probability_down,p.probability_neutral,p.probability_up,p.confidence,
            p.reference_close,p.horizon_minutes,p.outcome_candle_time_utc,p.model_sha256,
            p.input_features_json,p.provider_delay_seconds,
            o.actual_target,o.actual_return_pct,o.correct,o.observed_utc,
            o.endpoint_candle_time_utc
            FROM predictions p LEFT JOIN outcomes o
              ON o.prediction_candle_time_utc=p.candle_time_utc
            WHERE p.candle_time_utc=?""", (candle_time_utc,)).fetchone()
        if row is None:
            raise ValueError("New prediction was not found in the configured Alpaca database")
        record = dict(row)
        features = json.loads(record.pop("input_features_json"))
        event_source = db.execute("""SELECT source,feed,event_time_utc,received_at_utc
            FROM minute_bars WHERE source='Alpaca' AND feed='iex' AND received_at_utc>=?
            ORDER BY received_at_utc DESC LIMIT 1""", (capture_start.isoformat(),)).fetchone()
        final_candle = db.execute("""SELECT source,feed,aggregation_method,finalized_at_utc,
            candle_time_utc,open,high,low,close FROM five_minute_candles
            WHERE candle_time_utc=?""", (candle_time_utc,)).fetchone()
        path = db.execute("""SELECT candle_time_utc,source,feed FROM five_minute_candles
            WHERE candle_time_utc IN (?, ?, ?, ?) ORDER BY candle_time_utc""", (
                candle_time_utc,
                (parse_utc(candle_time_utc) + timedelta(minutes=5)).isoformat(),
                (parse_utc(candle_time_utc) + timedelta(minutes=10)).isoformat(),
                (parse_utc(candle_time_utc) + timedelta(minutes=15)).isoformat())).fetchall()
    probs = [record[k] for k in ("probability_down", "probability_neutral", "probability_up")]
    if record["prediction"] not in EXPECTED_CLASSES or not all(math.isfinite(float(p)) and 0 <= p <= 1 for p in probs):
        raise ValueError("Persisted prediction class/probability validation failed")
    if not math.isclose(sum(probs), 1.0, abs_tol=1e-8):
        raise ValueError("Persisted probability sum is invalid")
    if float(record["reference_close"]) <= 0 or record["horizon_minutes"] != EXPECTED_HORIZON:
        raise ValueError("Persisted reference price or horizon is invalid")
    if parse_utc(record["outcome_candle_time_utc"]) != parse_utc(record["candle_time_utc"]) + timedelta(minutes=EXPECTED_HORIZON):
        raise ValueError("Persisted outcome endpoint is not exactly 15 minutes later")
    if list(features.keys()) != sorted(feature_order):
        # Storage intentionally serializes feature JSON with sorted keys.
        raise ValueError("Persisted input feature names do not match the frozen feature set")
    if record["model_sha256"] == "" or not final_candle:
        raise ValueError("Prediction model identity or source finalized candle is missing")
    if final_candle["source"] != "Alpaca" or final_candle["feed"] != "iex":
        raise ValueError("Finalized candle provenance is not Alpaca IEX")
    return {
        "prediction_row_persisted": True,
        "feature_count_persisted": len(features),
        "feature_names_match_frozen_set": set(features) == set(feature_order),
        "candle_source": final_candle["source"], "candle_feed": final_candle["feed"],
        "aggregation_method": final_candle["aggregation_method"],
        "five_minute_path_candles_present": len(path),
        "five_minute_path_all_alpaca_iex": bool(path) and all(
            row["source"] == "Alpaca" and row["feed"] == "iex" for row in path),
        "minute_event_observed_during_capture": event_source is not None,
        "minute_event_source": event_source["source"] if event_source else None,
        "minute_event_feed": event_source["feed"] if event_source else None,
        "prediction": {k: record[k] for k in (
            "candle_time_utc", "prediction_created_utc", "prediction", "probability_down",
            "probability_neutral", "probability_up", "confidence", "reference_close",
            "horizon_minutes", "outcome_candle_time_utc", "model_sha256", "provider_delay_seconds",
            "actual_target", "actual_return_pct", "correct", "observed_utc", "endpoint_candle_time_utc")},
    }


def _read_only_capture_counts(db_path: Path, capture_start: datetime) -> dict:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=3) as db:
        minute = db.execute("""SELECT COUNT(*),COUNT(DISTINCT event_time_utc)
            FROM minute_bars WHERE source='Alpaca' AND feed='iex' AND received_at_utc>=?""",
            (capture_start.isoformat(),)).fetchone()
        candles = db.execute("""SELECT COUNT(*),COUNT(DISTINCT candle_time_utc)
            FROM five_minute_candles WHERE source='Alpaca' AND feed='iex' AND finalized_at_utc>=?""",
            (capture_start.isoformat(),)).fetchone()
    return {"minute_bars_received": int(minute[0]), "unique_minute_timestamps": int(minute[1]),
            "finalized_five_minute_candles": int(candles[0]),
            "unique_five_minute_timestamps": int(candles[1])}


def protocol_success_observed(status_snapshots: list[dict]) -> dict:
    """Use explicit protocol acknowledgements, never transport-connected as a proxy.

    Provider observed flags are sticky for the provider lifetime, and counts
    distinguish a new authentication/subscription acknowledgement after a
    capture's initial status snapshot. Current booleans also count when capture
    starts on an already authenticated/subscribed active session.
    """
    if not status_snapshots:
        return {"authentication_success_observed": False,
                "subscription_success_observed": False}
    initial = status_snapshots[0]
    observed = {}
    for name, current_key, sticky_key, count_key in (
        ("authentication", "authenticated", "authentication_success_observed", "authentication_success_count"),
        ("subscription", "subscribed", "subscription_success_observed", "subscription_success_count"),
    ):
        initial_count = int(initial.get(count_key) or 0)
        current_seen = any(snapshot.get(current_key) is True for snapshot in status_snapshots)
        new_ack_seen = any(int(snapshot.get(count_key) or 0) > initial_count
                           for snapshot in status_snapshots[1:])
        # Backward-compatible with an API status snapshot that predates counters.
        newly_sticky_seen = (initial.get(sticky_key) is not True and
                             any(snapshot.get(sticky_key) is True for snapshot in status_snapshots[1:]))
        observed[f"{name}_success_observed"] = bool(current_seen or new_ack_seen or newly_sticky_seen)
    return observed


def run_capture(base_url: str, duration_minutes: int, output_root: Path) -> Path:
    import config
    from live_app.model_loader import load_alpaca_candidate

    base_url = ensure_loopback_url(base_url)
    allowed, reason, close = market_window(utc_now())
    if not allowed:
        raise RuntimeError(reason)
    settings = config.get_alpaca_settings()
    if not settings.display_authorized:
        raise RuntimeError("Alpaca IEX display authorization is disabled; refusing capture")
    if not settings.credentials_configured:
        raise RuntimeError("Alpaca credentials are not configured; refusing capture")
    expected_path = (ROOT / EXPECTED_MODEL_RELATIVE).resolve()
    if settings.model_path.resolve() != expected_path:
        raise RuntimeError("Configured model path is not the frozen operational candidate")
    candidate = load_alpaca_candidate()
    feature_order = list(candidate.specification["features_exact_order"])
    if len(feature_order) != 24 or candidate.specification["class_order"] != EXPECTED_CLASSES:
        raise RuntimeError("Frozen feature/class contract failed preflight")
    if candidate.specification["target"]["horizon_minutes"] != EXPECTED_HORIZON:
        raise RuntimeError("Frozen prediction horizon failed preflight")

    capture_start = utc_now()
    deadline = time.monotonic() + duration_minutes * 60
    baseline_predictions = set()
    newly_seen = set()
    event_counts = {"new_minute_events": 0, "new_five_minute_candles": 0, "new_predictions": 0}
    reconnect_baseline = 0
    initial_status = None
    latest_status = None
    last_candle_epochs = set()
    errors = []

    def record(event: dict):
        event["observed_at_utc"] = utc_now().isoformat()
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")

    # Server health check, followed by safe JSON API snapshots. All URLs are loopback.
    health = _get_json(base_url + "/api/health")
    dashboard_html = _get_text(base_url + "/")
    dashboard_css = _get_text(base_url + "/static/alpaca.css")
    dashboard_js = _get_text(base_url + "/static/alpaca.js")
    initial_status = _get_json(base_url + "/api/status")
    if health.get("mode") != "alpaca_iex" or initial_status.get("mode") != "alpaca_iex":
        raise RuntimeError("The active localhost application is not in Alpaca IEX mode")
    if initial_status.get("display_authorized") is not True:
        raise RuntimeError("Active application reports display authorization disabled")
    if initial_status.get("model_ready") is not True or initial_status.get("model_sha256") != candidate.sha256:
        raise RuntimeError("Active application model readiness/hash does not match the frozen candidate")
    if initial_status.get("state") == "authentication_rejected":
        raise RuntimeError("Active application reports Alpaca authentication rejected; resolve locally before capture")
    run_dir = output_root / capture_start.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "observations.jsonl"
    summary_path = run_dir / "summary.json"
    reconnect_baseline = int(initial_status.get("reconnect_count") or 0)
    status_history = [initial_status]
    for row in _get_json(base_url + "/api/predictions?limit=100000"):
        baseline_predictions.add(row.get("candle_time_utc"))
    for row in _get_json(base_url + "/api/candles?limit=2000"):
        last_candle_epochs.add(row.get("time"))
    record({"event": "capture_started", "health_ok": health.get("ok"),
            "mode": health.get("mode"), "model_sha256": candidate.sha256,
            "dashboard_html_loaded": "ALPACA IEX" in dashboard_html,
            "dashboard_css_loaded": bool(dashboard_css), "dashboard_js_loaded": bool(dashboard_js),
            "market_state": initial_status.get("market_state"),
            "provider_state": initial_status.get("state"),
            "reconnect_count_start": reconnect_baseline,
            "baseline_prediction_count": len(baseline_predictions),
            "baseline_candle_count": len(last_candle_epochs)})

    while time.monotonic() < deadline:
        try:
            status = _get_json(base_url + "/api/status")
            status_history.append(status)
            candles = _get_json(base_url + "/api/candles?limit=2000")
            predictions = _get_json(base_url + "/api/predictions?limit=100000")
            latest_status = status
            new_candles = [row for row in candles if row.get("time") not in last_candle_epochs]
            for candle in new_candles:
                last_candle_epochs.add(candle.get("time"))
                event_counts["new_five_minute_candles"] += 1
                record({"event": "new_dashboard_candle", "candle": candle})
            for prediction in predictions:
                key = prediction.get("candle_time_utc")
                if not key or key in baseline_predictions or key in newly_seen:
                    continue
                created = parse_utc(prediction["prediction_created_utc"])
                if created < capture_start:
                    continue
                newly_seen.add(key)
                event_counts["new_predictions"] += 1
                evidence = _read_only_prediction(settings.database_path, key, feature_order, capture_start)
                if evidence["prediction"]["model_sha256"] != candidate.sha256:
                    raise ValueError("Persisted prediction model hash does not match frozen candidate")
                if evidence["feature_count_persisted"] != 24 or not evidence["feature_names_match_frozen_set"]:
                    raise ValueError("Persisted prediction feature set does not match frozen 24-feature contract")
                record({"event": "new_prediction", "persistence_evidence": evidence})
            provider_event = status.get("last_event_time_utc")
            if provider_event:
                # DB event count is authoritative; status exposes only the last event timestamp.
                event_counts.update(_read_only_capture_counts(settings.database_path, capture_start))
            if status.get("state") in ("authentication_rejected", "subscription_rejected", "connection_failed"):
                errors.append({"state": status.get("state"), "category": status.get("last_error_category"),
                               "transport": status.get("last_error_transport"),
                               "status_code": status.get("last_error_status_code")})
            record({"event": "status_snapshot", "status": {
                k: status.get(k) for k in ("state", "market_state", "connected", "authenticated", "subscribed",
                    "last_event_time_utc", "last_receive_time_utc", "receive_delay_seconds", "reconnect_count",
                    "last_completed_candle_time_et", "model_ready", "feature_ready", "pending_settlements",
                    "missing_data_warnings", "last_safe_error", "last_error_category", "last_error_transport",
                    "last_error_status_code")}})
            settled = []
            for prediction in predictions:
                key = prediction.get("candle_time_utc")
                if key in newly_seen and prediction.get("actual_target") is not None:
                    settled.append(prediction)
            if settled:
                settlement_evidence = []
                for row in settled:
                    evidence = _read_only_prediction(settings.database_path, row["candle_time_utc"], feature_order, capture_start)
                    if evidence["prediction"].get("actual_target") is None:
                        raise RuntimeError("API shows settlement but read-only SQLite does not")
                    if evidence["five_minute_path_candles_present"] != 4 or not evidence["five_minute_path_all_alpaca_iex"]:
                        raise RuntimeError("Settled prediction lacks exact decision/+5/+10/+15 candle path")
                    settlement_evidence.append(evidence)
                record({"event": "settlement_verified", "settled_predictions": settlement_evidence})
                break
        except Exception as exc:
            errors.append({"error_type": type(exc).__name__, "safe_message": str(exc)[:200]})
            record({"event": "local_api_or_persistence_error", "error_type": type(exc).__name__,
                    "safe_message": str(exc)[:200]})
        time.sleep(10)

    finished = utc_now()
    final_health = None
    final_api_ok = False
    try:
        final_health = _get_json(base_url + "/api/health")
        final_api_ok = bool(final_health.get("ok"))
    except Exception as exc:
        errors.append({"error_type": type(exc).__name__, "safe_message": str(exc)[:200]})
    new_rows = [p for p in _get_json(base_url + "/api/predictions?limit=100000")
                if p.get("candle_time_utc") in newly_seen]
    settled_rows = [p for p in new_rows if p.get("actual_target") is not None]
    persisted_counts = _read_only_capture_counts(settings.database_path, capture_start)
    event_counts["new_minute_events"] = persisted_counts["minute_bars_received"]
    event_counts["new_five_minute_candles"] = persisted_counts["finalized_five_minute_candles"]
    summary = {
        "capture_start_utc": capture_start.isoformat(), "capture_end_utc": finished.isoformat(),
        "capture_start_new_york": capture_start.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).isoformat(),
        "elapsed_seconds": (finished-capture_start).total_seconds(),
        "market_session_close_utc": close.astimezone(timezone.utc).isoformat() if close else None,
        "health_ok_at_end": final_api_ok, "final_health_mode": final_health.get("mode") if final_health else None,
        "model_sha256": candidate.sha256, "model_integrity": "PASS",
        "active_model_path": str(settings.model_path), "feature_count": len(feature_order),
        "class_order": EXPECTED_CLASSES, "prediction_horizon_minutes": EXPECTED_HORIZON,
        "initial_provider_state": initial_status.get("state"),
        "final_provider_state": latest_status.get("state") if latest_status else None,
        **protocol_success_observed(status_history),
        "event_counts": event_counts,
        "new_alpaca_iex_data_persisted": persisted_counts,
        "new_predictions": [{k:p.get(k) for k in ("candle_time_utc","prediction_created_utc","prediction",
            "probability_down","probability_neutral","probability_up","confidence","reference_close",
            "horizon_minutes","outcome_candle_time_utc","model_sha256","actual_target","actual_return_pct","correct")}
            for p in new_rows],
        "new_settled_predictions": len(settled_rows),
        "reconnect_count_start": reconnect_baseline,
        "reconnect_count_end": (latest_status or {}).get("reconnect_count"),
        "reconnect_exercised": bool(latest_status and int(latest_status.get("reconnect_count") or 0) > reconnect_baseline),
        "missing_data_warnings": (latest_status or {}).get("missing_data_warnings", []),
        "errors": errors,
        "full_prediction_and_settlement_cycle": bool(new_rows and settled_rows),
        "dashboard_visual_check": "manual: open http://127.0.0.1:8000 and verify actual candle/prediction/history display",
        "capture_output_directory": str(run_dir),
        "database_was_read_only_by_runner": True,
        "synthetic_data_inserted": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--duration-minutes", type=int, default=60)
    parser.add_argument("--confirm-live-capture", action="store_true",
                        help="explicitly authorize this one live localhost capture")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        ensure_loopback_url(args.base_url)
        if not args.confirm_live_capture:
            raise RuntimeError("Pass --confirm-live-capture to authorize a real live capture")
        if not 25 <= args.duration_minutes <= 180:
            raise ValueError("Duration must be between 25 and 180 minutes")
        result = run_capture(args.base_url, args.duration_minutes, args.output_root)
        print(json.dumps({"capture_result_file": str(result), "status": "capture_finished"}))
        return 0
    except Exception as exc:
        # Safe categories/messages only; never dump environment, headers, or traceback.
        print(json.dumps({"status": "not_captured_or_failed", "error_type": type(exc).__name__,
                          "safe_message": str(exc)[:200]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

