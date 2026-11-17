"""Read-only access to the separate JSONL log of experimental model previews.

These records are not production predictions and are never written to the
production prediction/outcome SQLite ledger.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from live_app.calendar import NY, UTC, session_for_date

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_RUN_DIR = ROOT / "reports" / "live_model_preview_runs"
PREVIEW_DB_PATH = ROOT / "data" / "live" / "ecl_research_previews.sqlite3"
MODEL_NAMES = {
    "xgboost_multiclass": "XGBoost Multiclass",
    "ridge_return": "Ridge Regression",
    "xgboost_binary_v3": "XGBoost Binary v3",
}
MODEL_IDS = tuple(MODEL_NAMES)
FINAL_PREVIEW_STATUSES = {"preview", "insufficient_features", "features_unavailable"}


def _utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _iter_records(run_dir: Path | None = None) -> list[dict[str, Any]]:
    """Read complete JSONL rows only; tolerate a partially written final line."""
    records: list[dict[str, Any]] = []
    run_dir = Path(run_dir) if run_dir is not None else PREVIEW_RUN_DIR
    if not run_dir.is_dir():
        return records
    for path in sorted(run_dir.glob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(row, dict) and _utc(row.get("candle_time_utc")) is not None:
                        records.append(row)
        except OSError:
            continue
    return records


class ResearchPreviewRecorder:
    """Append one separate JSONL record per current-session completed candle.

    This is deliberately independent from the immutable production ledger.
    It only stores outputs supplied by the research preview service; it never
    invokes a model itself.
    """

    def __init__(self, run_dir: Path = PREVIEW_RUN_DIR, *, started_at: datetime | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        started_at = started_at or datetime.now(UTC)
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("Preview recorder start time must be timezone-aware")
        stamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.run_dir / f"{stamp}_app.jsonl"
        self._lock = threading.Lock()
        self._completed = self._load_completed()

    def _load_completed(self) -> set[str]:
        completed: set[str] = set()
        for record in _iter_records(self.run_dir):
            statuses = {item.get("model_id"): item.get("status")
                        for item in record.get("models", []) if isinstance(item, dict)}
            if all(statuses.get(model_id) in FINAL_PREVIEW_STATUSES
                   for model_id in MODEL_IDS):
                stamp = _utc(record.get("candle_time_utc"))
                if stamp is not None:
                    completed.add(stamp.isoformat(timespec="seconds").replace("+00:00", "Z"))
        return completed

    def is_completed(self, candle_time_utc: str) -> bool:
        stamp = _utc(candle_time_utc)
        key = stamp.isoformat(timespec="seconds").replace("+00:00", "Z") if stamp else ""
        with self._lock:
            return key in self._completed

    @property
    def has_completed_records(self) -> bool:
        with self._lock:
            return bool(self._completed)

    def append(self, record: dict[str, Any]) -> bool:
        stamp = _utc(record.get("candle_time_utc"))
        models = record.get("models")
        if stamp is None or not isinstance(models, list):
            raise ValueError("A preview record needs a timezone-aware candle and model results")
        key = stamp.isoformat(timespec="seconds").replace("+00:00", "Z")
        statuses = {item.get("model_id"): item.get("status")
                    for item in models if isinstance(item, dict)}
        complete = all(statuses.get(model_id) in FINAL_PREVIEW_STATUSES
                       for model_id in MODEL_IDS)
        with self._lock:
            if key in self._completed:
                return False
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                import os
                os.fsync(stream.fileno())
            if complete:
                self._completed.add(key)
            return True


def _model_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    stamp = _utc(record.get("candle_time_utc"))
    if stamp is None:
        return []
    generated = _utc(record.get("checked_at_utc"))
    reference = record.get("reference_price")
    result = []
    models = record.get("models")
    if not isinstance(models, list):
        return result
    for model in models:
        if not isinstance(model, dict) or model.get("model_id") not in MODEL_NAMES:
            continue
        status = str(model.get("status", "unavailable"))
        probabilities = model.get("probabilities") if isinstance(model.get("probabilities"), dict) else None
        result.append({
            "model_id": model["model_id"],
            "model_name": MODEL_NAMES[model["model_id"]],
            "status": status,
            "prediction": model.get("prediction") if status == "preview" else None,
            "signal": model.get("signal") if status == "preview" else None,
            "probabilities": probabilities if status == "preview" else None,
            "predicted_return_percentage_points": (
                model.get("predicted_return_percentage_points") if status == "preview" else None
            ),
            "candle_time_utc": stamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "generated_at_utc": (_utc(model.get("generated_at_utc")) or generated),
            "reference_price": reference,
            "horizon_minutes": model.get("horizon_minutes", 15),
            "model_sha256": model.get("model_sha256"),
            "persisted_to_production_ledger": False,
            "outcome_status": "NOT_EVALUATED",
            "missing_features": model.get("missing_features") if status != "preview" else None,
            "message": model.get("message") if status != "preview" else None,
        })
    return result


class ResearchPreviewStore:
    """Separate immutable research-preview and outcome store (never production DB)."""

    def __init__(self, path: Path = PREVIEW_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS research_model_previews (
                    candle_time_utc TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    probabilities_json TEXT,
                    predicted_return_percentage_points REAL,
                    reference_price REAL NOT NULL,
                    horizon_minutes INTEGER NOT NULL,
                    generated_at_utc TEXT,
                    model_sha256 TEXT NOT NULL,
                    source_log TEXT NOT NULL,
                    PRIMARY KEY (candle_time_utc, model_id)
                );
                CREATE TABLE IF NOT EXISTS research_preview_outcomes (
                    candle_time_utc TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    endpoint_candle_time_utc TEXT NOT NULL,
                    actual_target TEXT NOT NULL,
                    actual_return_pct REAL NOT NULL,
                    actual_close REAL NOT NULL,
                    correct INTEGER NOT NULL CHECK (correct IN (0,1)),
                    settled_at_utc TEXT NOT NULL,
                    PRIMARY KEY (candle_time_utc, model_id),
                    FOREIGN KEY (candle_time_utc, model_id)
                      REFERENCES research_model_previews(candle_time_utc, model_id)
                );
                CREATE TABLE IF NOT EXISTS research_preview_conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candle_time_utc TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    existing_json TEXT NOT NULL,
                    incoming_json TEXT NOT NULL,
                    detected_at_utc TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS research_previews_no_update
                BEFORE UPDATE ON research_model_previews
                BEGIN SELECT RAISE(ABORT, 'research previews are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS research_previews_no_delete
                BEFORE DELETE ON research_model_previews
                BEGIN SELECT RAISE(ABORT, 'research previews are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS research_outcomes_no_update
                BEFORE UPDATE ON research_preview_outcomes
                BEGIN SELECT RAISE(ABORT, 'research outcomes are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS research_outcomes_no_delete
                BEFORE DELETE ON research_preview_outcomes
                BEGIN SELECT RAISE(ABORT, 'research outcomes are immutable'); END;
            """)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def import_log_records(self, records: list[dict[str, Any]], source_log: str) -> int:
        inserted = 0
        with self._connect() as db:
            for record in records:
                for row in _model_rows(record):
                    if row["status"] != "preview" or not row["prediction"] or not row["signal"]:
                        continue
                    try:
                        reference = float(row["reference_price"])
                        horizon = int(row["horizon_minutes"])
                        model_hash = str(row["model_sha256"] or "")
                        if not math.isfinite(reference) or reference <= 0 or horizon != 15 or len(model_hash) != 64:
                            continue
                        probabilities = row["probabilities"]
                        if probabilities is not None:
                            vals = [float(value) for value in probabilities.values()]
                            if (not vals or not all(math.isfinite(value) and 0 <= value <= 1 for value in vals)
                                    or not math.isclose(sum(vals), 1.0, abs_tol=1e-6)):
                                continue
                        estimate = row["predicted_return_percentage_points"]
                        if estimate is not None and not math.isfinite(float(estimate)):
                            continue
                        cursor = db.execute("""
                            INSERT OR IGNORE INTO research_model_previews
                            (candle_time_utc, model_id, model_name, prediction, signal, probabilities_json,
                             predicted_return_percentage_points, reference_price, horizon_minutes,
                             generated_at_utc, model_sha256, source_log)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row["candle_time_utc"], row["model_id"], row["model_name"],
                            row["prediction"], row["signal"],
                            json.dumps(probabilities, sort_keys=True) if probabilities is not None else None,
                            float(estimate) if estimate is not None else None, reference, horizon,
                            row["generated_at_utc"].isoformat(timespec="seconds") if row["generated_at_utc"] else None,
                            model_hash, source_log,
                        ))
                        inserted += cursor.rowcount
                        existing = db.execute("SELECT * FROM research_model_previews WHERE candle_time_utc=? AND model_id=?",
                                               (row["candle_time_utc"], row["model_id"])).fetchone()
                        if existing is not None and cursor.rowcount == 0:
                            expected = {"prediction": row["prediction"], "signal": row["signal"],
                                        "reference_price": reference, "model_sha256": model_hash}
                            actual = {key: existing[key] for key in expected}
                            if expected != actual:
                                db.execute("""
                                    INSERT INTO research_preview_conflicts
                                    (candle_time_utc, model_id, existing_json, incoming_json, detected_at_utc)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (row["candle_time_utc"], row["model_id"], json.dumps(actual, sort_keys=True),
                                      json.dumps(expected, sort_keys=True), datetime.now(UTC).isoformat(timespec="seconds")))
                    except (TypeError, ValueError, sqlite3.Error):
                        continue
        return inserted

    def settle_available(self, candles: list[dict[str, Any]], *, now: datetime | None = None) -> int:
        now = now or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Outcome clock must be timezone-aware")
        by_time: dict[datetime, dict[str, Any]] = {}
        for candle in candles:
            stamp = _utc(candle.get("candle_time_utc"))
            if stamp is not None:
                by_time[stamp] = candle
        with self._connect() as db:
            previews = db.execute("""
                SELECT p.* FROM research_model_previews p
                LEFT JOIN research_preview_outcomes o USING (candle_time_utc, model_id)
                WHERE o.candle_time_utc IS NULL
            """).fetchall()
            inserted = 0
            for preview in previews:
                decision = _utc(preview["candle_time_utc"])
                if decision is None:
                    continue
                local_day = decision.astimezone(NY).date()
                session = session_for_date(local_day)
                if session is None:
                    continue
                opened, closed = (bound.astimezone(UTC) for bound in session)
                endpoint = decision + pd.Timedelta(minutes=15).to_pytimedelta()
                if decision < opened or endpoint + pd.Timedelta(minutes=5).to_pytimedelta() > closed:
                    continue
                # Do not settle until the endpoint candle has closed and the
                # same two-minute finalization buffer used by the feed elapsed.
                if now.astimezone(UTC) < endpoint + pd.Timedelta(minutes=7).to_pytimedelta():
                    continue
                required = [decision + pd.Timedelta(minutes=offset).to_pytimedelta() for offset in (0, 5, 10, 15)]
                if not all(stamp in by_time for stamp in required):
                    continue
                closes = [float(by_time[stamp]["close"]) for stamp in required]
                reference = float(preview["reference_price"])
                if not math.isclose(closes[0], reference, rel_tol=0, abs_tol=1e-8):
                    continue
                ret_pct = (closes[-1] / reference - 1.0) * 100.0
                outcome = "UP" if ret_pct > 0.10 else "DOWN" if ret_pct < -0.10 else "NEUTRAL"
                correct = int(preview["prediction"] == outcome)
                cursor = db.execute("""
                    INSERT OR IGNORE INTO research_preview_outcomes
                    (candle_time_utc, model_id, endpoint_candle_time_utc, actual_target,
                     actual_return_pct, actual_close, correct, settled_at_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (preview["candle_time_utc"], preview["model_id"],
                      endpoint.isoformat(timespec="seconds").replace("+00:00", "Z"), outcome,
                      ret_pct, closes[-1], correct, now.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")))
                inserted += cursor.rowcount
        return inserted

    def rows(self, *, candle_time_utc: datetime | None = None, limit: int = 300,
             now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(UTC)
        query = """
            SELECT p.*, o.endpoint_candle_time_utc, o.actual_target, o.actual_return_pct,
                   o.actual_close, o.correct, o.settled_at_utc
            FROM research_model_previews p
            LEFT JOIN research_preview_outcomes o USING (candle_time_utc, model_id)
        """
        values: list[Any] = []
        if candle_time_utc is not None:
            query += " WHERE p.candle_time_utc=?"
            values.append(candle_time_utc.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"))
        query += " ORDER BY p.candle_time_utc DESC, p.model_id LIMIT ?"
        values.append(limit)
        with self._connect() as db:
            result = []
            for raw in db.execute(query, values).fetchall():
                row = dict(raw)
                row["probabilities"] = json.loads(row.pop("probabilities_json")) if row.get("probabilities_json") else None
                row["persisted_to_production_ledger"] = False
                row["status"] = "preview"
                row["outcome_status"] = "SETTLED" if row.get("actual_target") is not None else "PENDING"
                if row.get("actual_target") is None:
                    decision = _utc(row["candle_time_utc"])
                    session = session_for_date(decision.astimezone(NY).date()) if decision else None
                    endpoint = decision + pd.Timedelta(minutes=15).to_pytimedelta() if decision else None
                    if (session is None or (endpoint is not None and
                            endpoint + pd.Timedelta(minutes=5).to_pytimedelta() > session[1].astimezone(UTC))
                            or now.astimezone(UTC) >= session[1].astimezone(UTC)):
                        row["outcome_status"] = "UNAVAILABLE"
                row["source"] = "separate_research_preview_store"
                result.append(row)
            return result


def _synced_store(timeline: list[dict[str, Any]], *, now: datetime | None = None,
                  path: Path | None = None) -> ResearchPreviewStore:
    store = ResearchPreviewStore(path or PREVIEW_DB_PATH)
    store.import_log_records(_iter_records(), str(PREVIEW_RUN_DIR))
    store.settle_available(timeline, now=now)
    return store


def previews_for_candle(candle_time: datetime, *, timeline: list[dict[str, Any]] | None = None,
                        now: datetime | None = None, path: Path | None = None) -> dict[str, Any]:
    """Return saved preview and settled outcome rows without running inference."""
    if candle_time.tzinfo is None or candle_time.utcoffset() is None:
        raise ValueError("Candle timestamp must be timezone-aware")
    target = candle_time.astimezone(timezone.utc)
    if timeline is not None:
        store = _synced_store(timeline, now=now, path=path)
        rows = store.rows(candle_time_utc=target, limit=10, now=now)
        return {
            "source": "separate_research_preview_store",
            "candle_time_utc": target.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "models": rows,
        }
    by_model: dict[str, dict[str, Any]] = {}
    for record in _iter_records():
        if _utc(record.get("candle_time_utc")) != target:
            continue
        for row in _model_rows(record):
            old = by_model.get(row["model_id"])
            if old is None or (row.get("generated_at_utc") or datetime.min.replace(tzinfo=timezone.utc)) >= (
                    old.get("generated_at_utc") or datetime.min.replace(tzinfo=timezone.utc)):
                by_model[row["model_id"]] = row
    return {
        "source": "separate_experimental_preview_log",
        "candle_time_utc": target.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "models": sorted(by_model.values(), key=lambda row: row["model_id"]),
    }


def preview_history(limit: int = 300, *, timeline: list[dict[str, Any]] | None = None,
                    now: datetime | None = None, path: Path | None = None) -> dict[str, Any]:
    """Return latest model outputs and actual outcomes from the separate store."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if timeline is not None:
        store = _synced_store(timeline, now=now, path=path)
        rows = store.rows(limit=limit, now=now)
        return {"source": "separate_research_preview_store", "count": len(rows), "records": rows}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in _iter_records():
        for row in _model_rows(record):
            if row["status"] != "preview":
                continue
            key = (row["candle_time_utc"], row["model_id"])
            old = by_key.get(key)
            if old is None or (row.get("generated_at_utc") or datetime.min.replace(tzinfo=timezone.utc)) >= (
                    old.get("generated_at_utc") or datetime.min.replace(tzinfo=timezone.utc)):
                by_key[key] = row
    rows = sorted(by_key.values(), key=lambda row: (row["candle_time_utc"], row["model_id"]), reverse=True)
    return {"source": "separate_experimental_preview_log", "count": min(len(rows), limit), "records": rows[:limit]}


def all_preview_records(*, timeline: list[dict[str, Any]], now: datetime | None = None,
                        path: Path | None = None) -> list[dict[str, Any]]:
    """Return every saved preview and settled outcome for a complete export."""
    store = _synced_store(timeline, now=now, path=path)
    return store.rows(limit=1_000_000, now=now)
