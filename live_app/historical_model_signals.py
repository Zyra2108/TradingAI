"""Offline, immutable retrospective signals kept separate from live records."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from threadpoolctl import threadpool_limits

from config import ALPACA_MODEL_SPEC_PATH
from live_app.calendar import NY, UTC
from live_app.features import calculate_b_features
from live_app.model_loader import EXPECTED_CLASSES, load_alpaca_candidate
from live_app.unified_market import EXPECTED_SHA256 as PERMITTED_CACHE_SHA256, UnifiedHistoricalMarket

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORICAL_SIGNAL_DB = ROOT / "data" / "historical" / "ecl_model_indicator_backtest.sqlite3"
HORIZON_MINUTES = 15
SOURCE = "historical_backtest"


def _utc_text(stamp: datetime) -> str:
    return stamp.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class HistoricalSignalStore:
    """Separate append-only SQLite ledger for retrospective model signals."""

    def __init__(self, path: Path | str = DEFAULT_HISTORICAL_SIGNAL_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS historical_model_signals (
                    candle_time_utc TEXT PRIMARY KEY,
                    prediction_created_utc TEXT NOT NULL,
                    prediction TEXT NOT NULL CHECK(prediction IN ('DOWN','NEUTRAL','UP')),
                    probability_down REAL NOT NULL,
                    probability_neutral REAL NOT NULL,
                    probability_up REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reference_close REAL NOT NULL,
                    horizon_minutes INTEGER NOT NULL CHECK(horizon_minutes = 15),
                    outcome_candle_time_utc TEXT NOT NULL,
                    model_identifier TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    source TEXT NOT NULL CHECK(source = 'historical_backtest'),
                    input_features_json TEXT NOT NULL,
                    outcome_status TEXT NOT NULL CHECK(outcome_status IN ('SETTLED','UNAVAILABLE')),
                    actual_target TEXT,
                    actual_return_pct REAL,
                    actual_close REAL,
                    correct INTEGER,
                    unavailable_reason TEXT
                );
                CREATE TRIGGER IF NOT EXISTS historical_signals_no_update
                BEFORE UPDATE ON historical_model_signals BEGIN
                    SELECT RAISE(ABORT, 'historical model signals are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS historical_signals_no_delete
                BEFORE DELETE ON historical_model_signals BEGIN
                    SELECT RAISE(ABORT, 'historical model signals are immutable');
                END;
                CREATE TABLE IF NOT EXISTS generation_manifest (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def insert_many(self, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> int:
        inserted = 0
        with self._connect() as connection:
            for row in rows:
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO historical_model_signals (
                    candle_time_utc,prediction_created_utc,prediction,
                    probability_down,probability_neutral,probability_up,confidence,
                    reference_close,horizon_minutes,outcome_candle_time_utc,
                    model_identifier,model_sha256,source,input_features_json,
                    outcome_status,actual_target,actual_return_pct,actual_close,
                    correct,unavailable_reason)
                    VALUES (:candle_time_utc,:prediction_created_utc,:prediction,
                    :probability_down,:probability_neutral,:probability_up,:confidence,
                    :reference_close,:horizon_minutes,:outcome_candle_time_utc,
                    :model_identifier,:model_sha256,:source,:input_features_json,
                    :outcome_status,:actual_target,:actual_return_pct,:actual_close,
                    :correct,:unavailable_reason)""", row)
                inserted += cursor.rowcount
            for key, value in manifest.items():
                connection.execute(
                    "INSERT OR IGNORE INTO generation_manifest(key,value) VALUES (?,?)",
                    (key, json.dumps(value, sort_keys=True)),
                )
        return inserted

    def rows(self, limit: int = 100000) -> list[dict[str, Any]]:
        with self._connect() as connection:
            records = connection.execute(
                "SELECT * FROM historical_model_signals ORDER BY candle_time_utc DESC LIMIT ?",
                (max(1, min(int(limit), 100000)),),
            ).fetchall()
        return [dict(row) for row in records]

    def at(self, candle_time_utc: datetime) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM historical_model_signals WHERE candle_time_utc=?",
                (_utc_text(candle_time_utc),),
            ).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM historical_model_signals").fetchone()[0])


def _feature_frame(candles: list[dict[str, Any]], feature_order: list[str]) -> pd.DataFrame:
    rows = []
    for candle in candles:
        local = datetime.fromtimestamp(int(candle["time"]), UTC).astimezone(NY).replace(tzinfo=None)
        rows.append({"datetime": local, **{name: float(candle[name]) for name in ("open", "high", "low", "close")}})
    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(columns=["datetime", *feature_order])
    return calculate_b_features(raw, feature_order)


def _outcome_for(candle: dict[str, Any], candles_by_time: dict[int, dict[str, Any]],
                 sessions: dict[Any, tuple[datetime, datetime]]) -> tuple[str, str | None, str | None, float | None, float | None, str | None]:
    decision = datetime.fromtimestamp(int(candle["time"]), UTC)
    endpoint = decision + timedelta(minutes=HORIZON_MINUTES)
    local = decision.astimezone(NY)
    session = sessions.get(local.date())
    if session is None:
        return "UNAVAILABLE", None, None, None, None, "no_valid_nyse_session"
    opened, closed = session
    if (decision < opened.astimezone(UTC) or decision + timedelta(minutes=5) > closed.astimezone(UTC)
            or endpoint + timedelta(minutes=5) > closed.astimezone(UTC)):
        return "UNAVAILABLE", _utc_text(endpoint), None, None, None, "endpoint_outside_same_session"
    path_times = [int((decision + timedelta(minutes=step)).timestamp()) for step in (0, 5, 10, 15)]
    path = [candles_by_time.get(stamp) for stamp in path_times]
    if any(row is None for row in path):
        return "UNAVAILABLE", _utc_text(endpoint), None, None, None, "incomplete_or_missing_outcome_path"
    reference, target = float(path[0]["close"]), float(path[-1]["close"])
    if not np.isfinite([reference, target]).all() or reference <= 0:
        return "UNAVAILABLE", _utc_text(endpoint), None, None, None, "invalid_outcome_price"
    return_pct = (target / reference - 1.0) * 100.0
    actual = "UP" if return_pct > 0.10 else "DOWN" if return_pct < -0.10 else "NEUTRAL"
    return "SETTLED", _utc_text(endpoint), actual, return_pct, target, None


def generate_historical_signals(*, store: HistoricalSignalStore | None = None,
                                candles: list[dict[str, Any]] | None = None,
                                candidate=None) -> dict[str, Any]:
    """Generate deterministic retrospective signals from the permitted cache only.

    No test-period source is opened. Predictions are saved only to the separate
    historical ledger; live prediction and outcome tables are never referenced.
    """
    store = store or HistoricalSignalStore()
    candidate = candidate or load_alpaca_candidate()
    if candles is None:
        candles, _excluded = UnifiedHistoricalMarket.load_permitted_pretest_candles()
    feature_order = candidate.specification["features_exact_order"]
    if len(feature_order) != 24 or list(candidate.model.feature_names_in_) != feature_order:
        raise ValueError("Frozen feature order does not match the historical generator")
    if candidate.specification["class_order"] != EXPECTED_CLASSES:
        raise ValueError("Frozen class order is not DOWN, NEUTRAL, UP")
    if any(datetime.fromtimestamp(int(row["time"]), UTC).astimezone(NY).date().isoformat() > "2026-09-24"
           for row in candles):
        raise ValueError("Historical signal input extends beyond the permitted pre-test cutoff")
    candles = sorted(candles, key=lambda row: int(row["time"]))
    local_dates = [datetime.fromtimestamp(int(row["time"]), UTC).astimezone(NY).date() for row in candles]
    schedule = mcal.get_calendar("NYSE").schedule(start_date=min(local_dates), end_date=max(local_dates))
    sessions = {
        index.date(): (row.market_open.tz_convert(NY).to_pydatetime(), row.market_close.tz_convert(NY).to_pydatetime())
        for index, row in schedule.iterrows()
    }
    feature_frame = _feature_frame(candles, feature_order)
    features = feature_frame[feature_order].to_numpy(dtype=float)
    valid_mask = np.isfinite(features).all(axis=1)
    feature_missing_counts = {
        name: int((~np.isfinite(feature_frame[name].to_numpy(dtype=float))).sum())
        for name in feature_order
    }
    segment_positions = np.zeros(len(candles), dtype=np.int32)
    segment_count = 0
    previous_day = previous_stamp = None
    for index, candle in enumerate(candles):
        stamp = int(candle["time"])
        day = datetime.fromtimestamp(stamp, UTC).astimezone(NY).date()
        if previous_stamp is None or stamp - previous_stamp != 300 or day != previous_day:
            segment_count += 1
            segment_positions[index] = 0
        else:
            segment_positions[index] = segment_positions[index - 1] + 1
        previous_stamp, previous_day = stamp, day
    warmup_mask = (~valid_mask) & (segment_positions < 20)
    unexpected_missing_mask = (~valid_mask) & (segment_positions >= 20)
    valid_indices = np.flatnonzero(valid_mask)
    valid_frame = feature_frame.loc[valid_mask, feature_order]
    probability_blocks = []
    # The frozen estimator was trained with n_jobs=-1. Bound only the runtime
    # thread pool for this offline generation; the artifact and model settings
    # are not changed.
    with threadpool_limits(limits=4):
        for start in range(0, len(valid_frame), 2048):
            batch = valid_frame.iloc[start:start + 2048]
            block = np.asarray(candidate.model.predict_proba(batch), dtype=float)
            if block.shape != (len(batch), 3) or not np.isfinite(block).all() or (block < 0).any():
                raise ValueError("Frozen model returned invalid class probabilities")
            if not np.allclose(block.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("Frozen model probabilities do not sum to one")
            probability_blocks.append(block)
    probabilities = np.vstack(probability_blocks) if probability_blocks else np.empty((0, 3))
    candle_by_time = {int(row["time"]): row for row in candles}
    records: list[dict[str, Any]] = []
    settled = unavailable = 0
    for index, probability in zip(valid_indices, probabilities):
        candle = candles[int(index)]
        candle_time = datetime.fromtimestamp(int(candle["time"]), UTC)
        prediction = EXPECTED_CLASSES[int(np.argmax(probability))]
        outcome_status, endpoint_text, actual, return_pct, actual_close, reason = _outcome_for(candle, candle_by_time, sessions)
        settled += outcome_status == "SETTLED"
        unavailable += outcome_status == "UNAVAILABLE"
        feature_values = {name: float(feature_frame.iloc[int(index)][name]) for name in feature_order}
        records.append({
            "candle_time_utc": _utc_text(candle_time),
            "prediction_created_utc": _utc_text(candle_time + timedelta(minutes=5)),
            "prediction": prediction,
            "probability_down": float(probability[0]),
            "probability_neutral": float(probability[1]),
            "probability_up": float(probability[2]),
            "confidence": float(probability.max()),
            "reference_close": float(candle["close"]),
            "horizon_minutes": HORIZON_MINUTES,
            "outcome_candle_time_utc": endpoint_text or _utc_text(candle_time + timedelta(minutes=HORIZON_MINUTES)),
            "model_identifier": candidate.specification.get("candidate", "frozen_alpaca_candidate"),
            "model_sha256": candidate.sha256,
            "source": SOURCE,
            "input_features_json": json.dumps(feature_values, sort_keys=False, separators=(",", ":")),
            "outcome_status": outcome_status,
            "actual_target": actual,
            "actual_return_pct": return_pct,
            "actual_close": actual_close,
            "correct": None if actual is None else int(prediction == actual),
            "unavailable_reason": reason,
        })
    manifest = {
        "source": "Alpaca IEX permitted 1-minute train/validation cache aggregated by verified 5-minute buckets",
        "source_first_date_et": "2025-11-13",
        "source_last_date_et": "2026-09-24",
        "source_cache_sha256": PERMITTED_CACHE_SHA256,
        "model_sha256": candidate.sha256,
        "feature_order": feature_order,
        "class_order": EXPECTED_CLASSES,
        "horizon_minutes": HORIZON_MINUTES,
        "prediction_policy": "raw three-class argmax",
        "feature_generation": "live_app.features.calculate_b_features / src/11_create_features.py and src/17_add_time_features.py",
        "outcome_path": "decision candle and exact +5/+10/+15 minute candles, same NYSE session",
        "retrospective": True,
    }
    inserted = store.insert_many(records, manifest)
    return {
        "source_candles": len(candles),
        "feature_ready_candles": len(valid_indices),
        "skipped_feature_warmup_or_gap": len(candles) - len(valid_indices),
        "continuous_session_or_gap_segments": segment_count,
        "warmup_skipped_candles_first_20_per_segment": int(warmup_mask.sum()),
        "unexpected_missing_rows_after_warmup": int(unexpected_missing_mask.sum()),
        "missing_required_values_by_feature": feature_missing_counts,
        "historical_predictions_generated": len(records),
        "records_inserted_this_run": inserted,
        "records_existing_or_idempotently_skipped": len(records) - inserted,
        "records_in_store": store.count(),
        "settled_outcomes": int(settled),
        "outcome_unavailable": int(unavailable),
        "model_sha256": candidate.sha256,
        "source_cache_sha256": PERMITTED_CACHE_SHA256,
        "first_candle_utc": records[0]["candle_time_utc"] if records else None,
        "last_candle_utc": records[-1]["candle_time_utc"] if records else None,
        "source_date_limit_et": "2026-09-24",
        "store": str(store.path),
    }
