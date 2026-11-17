"""Durable, Alpaca-only storage for source minute bars and derived candles.

The store only opens the supplied path (defaulting to the configured Alpaca DB)
and never migrates or reads another project database.
"""
from __future__ import annotations

import json
import hashlib
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from config import ALPACA_DB_PATH


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_stamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _validate_bar(bar: Any) -> None:
    if getattr(bar, "symbol", None) != "ECL":
        raise ValueError("Only ECL bars can be persisted")
    event_time = _utc(bar.event_time_utc, "event_time_utc")
    _utc(bar.received_at_utc, "received_at_utc")
    if event_time.second != 0 or event_time.microsecond != 0:
        raise ValueError("Minute-bar event timestamp must be aligned to a minute")
    values = (bar.open, bar.high, bar.low, bar.close)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or v <= 0 for v in values):
        raise ValueError("OHLC values must be finite positive numbers")
    if bar.high < max(bar.open, bar.low, bar.close) or bar.low > min(bar.open, bar.high, bar.close):
        raise ValueError("Invalid OHLC range")
    if bar.volume is not None and (not math.isfinite(float(bar.volume)) or bar.volume < 0):
        raise ValueError("Volume must be finite and nonnegative")
    if bar.trade_count is not None and (isinstance(bar.trade_count, bool) or not isinstance(bar.trade_count, int) or bar.trade_count < 0):
        raise ValueError("Trade count must be a nonnegative integer")


class AlpacaBarStore:
    """Versioned SQLite store that only touches its Alpaca-specific path."""

    SCHEMA_VERSION = 4

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path if path is not None else ALPACA_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _migrate(self) -> None:
        with self.connect() as db:
            version = int(db.execute("PRAGMA user_version").fetchone()[0])
            if version > self.SCHEMA_VERSION:
                raise RuntimeError("Alpaca database schema is newer than this application")
            if version == 0:
                db.executescript("""
                    CREATE TABLE schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at_utc TEXT NOT NULL,
                        description TEXT NOT NULL
                    );
                    CREATE TABLE minute_bars (
                        event_time_utc TEXT PRIMARY KEY,
                        received_at_utc TEXT NOT NULL,
                        symbol TEXT NOT NULL CHECK(symbol = 'ECL'),
                        open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
                        volume REAL,
                        trade_count INTEGER,
                        source TEXT NOT NULL,
                        feed TEXT NOT NULL,
                        original_event_timestamp TEXT NOT NULL
                    );
                    CREATE TABLE minute_bar_conflicts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_time_utc TEXT NOT NULL,
                        observed_at_utc TEXT NOT NULL,
                        conflict_kind TEXT NOT NULL,
                        observed_json TEXT NOT NULL,
                        source TEXT NOT NULL,
                        feed TEXT NOT NULL
                    );
                    CREATE INDEX minute_bar_conflicts_time_idx ON minute_bar_conflicts(event_time_utc);
                    CREATE TABLE five_minute_candles (
                        candle_time_utc TEXT PRIMARY KEY,
                        finalized_at_utc TEXT NOT NULL,
                        open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
                        volume REAL,
                        trade_count INTEGER,
                        source TEXT NOT NULL,
                        feed TEXT NOT NULL,
                        aggregation_method TEXT NOT NULL,
                        aggregation_metadata_json TEXT NOT NULL
                    );
                    CREATE TABLE five_minute_conflicts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        candle_time_utc TEXT NOT NULL,
                        observed_at_utc TEXT NOT NULL,
                        observed_json TEXT NOT NULL,
                        source TEXT NOT NULL,
                        feed TEXT NOT NULL
                    );
                    CREATE TABLE aggregation_gaps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bucket_start_utc TEXT NOT NULL,
                        detected_at_utc TEXT NOT NULL,
                        missing_timestamps_json TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        UNIQUE(bucket_start_utc, missing_timestamps_json)
                    );
                    INSERT INTO schema_migrations(version, applied_at_utc, description)
                    VALUES(1, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        'Create Alpaca minute-bar and finalized five-minute storage');
                    PRAGMA user_version=1;
                """)
                version = 1
            if version == 1:
                db.executescript("""
                    CREATE TABLE predictions (
                        candle_time_utc TEXT PRIMARY KEY,
                        prediction_created_utc TEXT NOT NULL,
                        prediction TEXT NOT NULL CHECK(prediction IN ('DOWN','NEUTRAL','UP')),
                        probability_down REAL NOT NULL,
                        probability_neutral REAL NOT NULL,
                        probability_up REAL NOT NULL,
                        confidence REAL NOT NULL,
                        reference_close REAL NOT NULL,
                        horizon_minutes INTEGER NOT NULL CHECK(horizon_minutes=15),
                        outcome_candle_time_utc TEXT NOT NULL,
                        model_sha256 TEXT NOT NULL,
                        input_features_json TEXT NOT NULL,
                        provider_delay_seconds REAL NOT NULL,
                        FOREIGN KEY(candle_time_utc) REFERENCES five_minute_candles(candle_time_utc)
                    );
                    CREATE TABLE outcomes (
                        prediction_candle_time_utc TEXT PRIMARY KEY,
                        observed_utc TEXT NOT NULL,
                        endpoint_candle_time_utc TEXT NOT NULL,
                        actual_target TEXT NOT NULL CHECK(actual_target IN ('DOWN','NEUTRAL','UP')),
                        actual_return_pct REAL NOT NULL,
                        correct INTEGER NOT NULL CHECK(correct IN (0,1)),
                        FOREIGN KEY(prediction_candle_time_utc) REFERENCES predictions(candle_time_utc),
                        FOREIGN KEY(endpoint_candle_time_utc) REFERENCES five_minute_candles(candle_time_utc)
                    );
                    CREATE INDEX predictions_due_idx ON predictions(outcome_candle_time_utc);
                    CREATE TRIGGER predictions_no_update
                    BEFORE UPDATE ON predictions BEGIN SELECT RAISE(ABORT, 'predictions are immutable'); END;
                    CREATE TRIGGER predictions_no_delete
                    BEFORE DELETE ON predictions BEGIN SELECT RAISE(ABORT, 'predictions are immutable'); END;
                    CREATE TRIGGER outcomes_no_update
                    BEFORE UPDATE ON outcomes BEGIN SELECT RAISE(ABORT, 'outcomes are immutable'); END;
                    CREATE TRIGGER outcomes_no_delete
                    BEFORE DELETE ON outcomes BEGIN SELECT RAISE(ABORT, 'outcomes are immutable'); END;
                    INSERT INTO schema_migrations(version, applied_at_utc, description)
                    VALUES(2, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        'Add immutable Alpaca predictions and verified outcomes');
                    PRAGMA user_version=2;
                """)
                version = 2
            if version == 2:
                db.executescript("""
                    CREATE TABLE minute_bar_updates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_time_utc TEXT NOT NULL,
                        observed_at_utc TEXT NOT NULL,
                        open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
                        volume REAL,
                        trade_count INTEGER,
                        original_event_timestamp TEXT NOT NULL,
                        payload_hash TEXT NOT NULL UNIQUE,
                        late INTEGER NOT NULL CHECK(late IN (0,1))
                    );
                    CREATE INDEX minute_bar_updates_time_idx
                        ON minute_bar_updates(event_time_utc, observed_at_utc);
                    CREATE TABLE stream_interruptions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        started_at_utc TEXT NOT NULL,
                        ended_at_utc TEXT,
                        reason TEXT NOT NULL
                    );
                    CREATE INDEX stream_interruptions_open_idx
                        ON stream_interruptions(ended_at_utc, started_at_utc);
                    CREATE UNIQUE INDEX one_open_stream_interruption
                        ON stream_interruptions((1)) WHERE ended_at_utc IS NULL;
                    CREATE TABLE aggregation_invalidations (
                        bucket_start_utc TEXT NOT NULL,
                        detected_at_utc TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        PRIMARY KEY(bucket_start_utc, reason)
                    );
                    INSERT INTO schema_migrations(version, applied_at_utc, description)
                    VALUES(3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        'Add updated minute-bar observations and stream interruption diagnostics');
                    PRAGMA user_version=3;
                """)
                version = 3
            if version == 3:
                db.executescript("""
                    CREATE TABLE late_minute_updates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_time_utc TEXT NOT NULL,
                        observed_at_utc TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        UNIQUE(event_time_utc, observed_at_utc, payload_hash)
                    );
                    INSERT INTO schema_migrations(version, applied_at_utc, description)
                    VALUES(4, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        'Add immutable audit records for late updated minute bars');
                    PRAGMA user_version=4;
                """)

    @staticmethod
    def _bar_payload(bar: Any) -> dict[str, Any]:
        return {
            "event_time_utc": _stamp(bar.event_time_utc),
            "received_at_utc": _stamp(bar.received_at_utc),
            "symbol": bar.symbol,
            "open": float(bar.open), "high": float(bar.high),
            "low": float(bar.low), "close": float(bar.close),
            "volume": None if bar.volume is None else float(bar.volume),
            "trade_count": bar.trade_count,
            "source": "Alpaca", "feed": "iex",
            "original_event_timestamp": bar.original_event_timestamp,
        }

    def insert_minute_bar(self, bar: Any) -> tuple[bool, bool]:
        """Insert first observation; return (inserted, conflicting_revision)."""
        _validate_bar(bar)
        item = self._bar_payload(bar)
        with self.connect() as db:
            existing = db.execute(
                "SELECT event_time_utc, received_at_utc, symbol, open, high, low, close, volume, trade_count, source, feed, original_event_timestamp "
                "FROM minute_bars WHERE event_time_utc=?", (item["event_time_utc"],)
            ).fetchone()
            if existing is None:
                db.execute("""INSERT INTO minute_bars (
                    event_time_utc, received_at_utc, symbol, open, high, low, close,
                    volume, trade_count, source, feed, original_event_timestamp
                ) VALUES (:event_time_utc, :received_at_utc, :symbol, :open, :high, :low, :close,
                    :volume, :trade_count, :source, :feed, :original_event_timestamp)""", item)
                return True, False
            # Receive time and raw timestamp spelling can differ across duplicate
            # deliveries; only market content determines whether a bar was revised.
            market_keys = ("symbol", "open", "high", "low", "close", "volume", "trade_count", "source", "feed")
            same = all(existing[key] == item[key] for key in market_keys)
            if same:
                return False, False
            db.execute("""INSERT INTO minute_bar_conflicts (
                event_time_utc, observed_at_utc, conflict_kind, observed_json, source, feed
            ) VALUES (?, ?, ?, ?, ?, ?)""", (
                item["event_time_utc"], item["received_at_utc"], "revised_or_conflicting_observation",
                json.dumps(item, sort_keys=True, separators=(",", ":")), item["source"], item["feed"],
            ))
            return False, True

    def get_minute_bars(self, start_utc: datetime, end_utc: datetime,
                        *, update_cutoff_utc: datetime | None = None) -> list[dict[str, Any]]:
        start, end = _stamp(start_utc), _stamp(end_utc)
        with self.connect() as db:
            rows = db.execute("""SELECT event_time_utc, received_at_utc, symbol, open, high, low, close,
                volume, trade_count, source, feed, original_event_timestamp
                FROM minute_bars WHERE event_time_utc>=? AND event_time_utc<? ORDER BY event_time_utc""", (start, end)).fetchall()
            result = {row["event_time_utc"]: dict(row) for row in rows}
            if update_cutoff_utc is not None:
                updates = db.execute("""SELECT event_time_utc, observed_at_utc, open, high, low, close,
                    volume, trade_count, original_event_timestamp FROM minute_bar_updates
                    WHERE event_time_utc>=? AND event_time_utc<? AND observed_at_utc<=? AND late=0
                    ORDER BY observed_at_utc, id""", (start, end, _stamp(update_cutoff_utc))).fetchall()
                for update in updates:
                    key = update["event_time_utc"]
                    if key in result:
                        item = result[key]
                        for field in ("open", "high", "low", "close", "volume", "trade_count", "original_event_timestamp"):
                            item[field] = update[field]
                        item["received_at_utc"] = update["observed_at_utc"]
        return [result[key] for key in sorted(result)]

    def insert_updated_minute_bar(self, bar: Any, *, bucket_start_utc: datetime,
                                  bucket_end_utc: datetime, update_cutoff_utc: datetime) -> tuple[bool, bool]:
        """Record a full-replacement updatedBar; never rewrite source or finalized data."""
        _validate_bar(bar)
        item = self._bar_payload(bar)
        cutoff = _stamp(update_cutoff_utc)
        with self.connect() as db:
            finalized = db.execute("SELECT 1 FROM five_minute_candles WHERE candle_time_utc=?",
                                   (_stamp(bucket_start_utc),)).fetchone() is not None
            late = finalized or item["received_at_utc"] > cutoff
            # An updatedBar may be the first event observed after a reconnect.
            existing = db.execute("SELECT 1 FROM minute_bars WHERE event_time_utc=?",
                                  (item["event_time_utc"],)).fetchone()
            inserted_base = False
            if existing is None and not finalized and item["received_at_utc"] <= cutoff:
                db.execute("""INSERT INTO minute_bars (
                    event_time_utc, received_at_utc, symbol, open, high, low, close,
                    volume, trade_count, source, feed, original_event_timestamp
                ) VALUES (:event_time_utc, :received_at_utc, :symbol, :open, :high, :low, :close,
                    :volume, :trade_count, :source, :feed, :original_event_timestamp)""", item)
                inserted_base = True
            canonical = {key: item[key] for key in ("event_time_utc", "open", "high", "low", "close", "volume", "trade_count")}
            payload_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if late:
                reason = "after_five_minute_finalization" if finalized else "after_finalization_buffer"
                db.execute("""INSERT OR IGNORE INTO late_minute_updates(
                    event_time_utc, observed_at_utc, payload_hash, reason
                ) VALUES (?, ?, ?, ?)""", (item["event_time_utc"], item["received_at_utc"], payload_hash, reason))
            cursor = db.execute("""INSERT OR IGNORE INTO minute_bar_updates (
                event_time_utc, observed_at_utc, open, high, low, close, volume, trade_count,
                original_event_timestamp, payload_hash, late
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                item["event_time_utc"], item["received_at_utc"], item["open"], item["high"],
                item["low"], item["close"], item["volume"], item["trade_count"],
                item["original_event_timestamp"], payload_hash, int(late)))
        return inserted_base or cursor.rowcount > 0, late

    def begin_stream_interruption(self, at_utc: datetime, reason: str = "websocket_disconnected") -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            open_row = db.execute("SELECT id FROM stream_interruptions WHERE ended_at_utc IS NULL ORDER BY id DESC LIMIT 1").fetchone()
            if open_row is None:
                db.execute("INSERT INTO stream_interruptions(started_at_utc, ended_at_utc, reason) VALUES (?, NULL, ?)",
                           (_stamp(at_utc), reason))

    def end_stream_interruption(self, at_utc: datetime) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id, started_at_utc FROM stream_interruptions WHERE ended_at_utc IS NULL ORDER BY id DESC LIMIT 1").fetchone()
            if row is not None:
                end = max(_stamp(at_utc), row["started_at_utc"])
                db.execute("UPDATE stream_interruptions SET ended_at_utc=? WHERE id=?", (end, row["id"]))

    def close_open_stream_interruptions(self, at_utc: datetime) -> None:
        self.end_stream_interruption(at_utc)

    def record_aggregation_invalidation(self, bucket_start_utc: datetime, detected_at_utc: datetime,
                                        reason: str = "minute_processing_failure") -> None:
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO aggregation_invalidations(bucket_start_utc, detected_at_utc, reason) VALUES (?, ?, ?)",
                       (_stamp(bucket_start_utc), _stamp(detected_at_utc), reason))

    def bucket_rejection_reason(self, bucket_start_utc: datetime, bucket_end_utc: datetime,
                                now_utc: datetime) -> str | None:
        start, end, now = _stamp(bucket_start_utc), _stamp(bucket_end_utc), _stamp(now_utc)
        with self.connect() as db:
            invalid = db.execute("SELECT reason FROM aggregation_invalidations WHERE bucket_start_utc=? LIMIT 1", (start,)).fetchone()
            if invalid:
                return invalid[0]
            interruption = db.execute("""SELECT 1 FROM stream_interruptions
                WHERE started_at_utc < ? AND COALESCE(ended_at_utc, ?) > ? LIMIT 1""",
                (end, now, start)).fetchone()
        return "known_stream_interruption" if interruption else None

    def late_update_count(self) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM late_minute_updates").fetchone()[0])

    def latest_late_update(self) -> dict[str, str] | None:
        with self.connect() as db:
            row = db.execute("""SELECT event_time_utc, observed_at_utc, reason
                FROM late_minute_updates ORDER BY observed_at_utc DESC, id DESC LIMIT 1""").fetchone()
        return None if row is None else dict(row)

    def open_stream_interruption_count(self) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM stream_interruptions WHERE ended_at_utc IS NULL").fetchone()[0])

    def minute_bar_timestamps(self) -> list[datetime]:
        with self.connect() as db:
            rows = db.execute("SELECT event_time_utc FROM minute_bars ORDER BY event_time_utc").fetchall()
        return [_parse_stamp(row[0]) for row in rows]

    @staticmethod
    def _candle_payload(candle: Any) -> dict[str, Any]:
        return {
            "candle_time_utc": _stamp(candle.candle_time_utc),
            "finalized_at_utc": _stamp(candle.finalized_at_utc),
            "open": float(candle.open), "high": float(candle.high),
            "low": float(candle.low), "close": float(candle.close),
            "volume": candle.volume, "trade_count": candle.trade_count,
            "source": candle.source, "feed": candle.feed,
            "aggregation_method": candle.aggregation_method,
            "aggregation_metadata_json": json.dumps(candle.aggregation_metadata, sort_keys=True, separators=(",", ":")),
        }

    def insert_five_minute_candle(self, candle: Any) -> tuple[bool, bool]:
        """Persist a finalized candle immutably; return (inserted, conflicting)."""
        item = self._candle_payload(candle)
        with self.connect() as db:
            # Serialize the read-then-insert decision across SQLite clients.
            # The primary key remains the final constraint, while BEGIN
            # IMMEDIATE prevents two workers from both observing no row.
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("""SELECT open, high, low, close, volume, trade_count, source, feed,
                aggregation_method, aggregation_metadata_json FROM five_minute_candles WHERE candle_time_utc=?""",
                (item["candle_time_utc"],)).fetchone()
            if existing is None:
                db.execute("""INSERT INTO five_minute_candles (
                    candle_time_utc, finalized_at_utc, open, high, low, close, volume, trade_count,
                    source, feed, aggregation_method, aggregation_metadata_json
                ) VALUES (:candle_time_utc, :finalized_at_utc, :open, :high, :low, :close, :volume,
                    :trade_count, :source, :feed, :aggregation_method, :aggregation_metadata_json)""", item)
                return True, False
            comparable = {k: item[k] for k in ("open", "high", "low", "close", "volume", "trade_count", "source", "feed", "aggregation_method", "aggregation_metadata_json")}
            if all(existing[k] == v for k, v in comparable.items()):
                return False, False
            db.execute("""INSERT INTO five_minute_conflicts (
                candle_time_utc, observed_at_utc, observed_json, source, feed
            ) VALUES (?, ?, ?, ?, ?)""", (item["candle_time_utc"], item["finalized_at_utc"],
                json.dumps(item, sort_keys=True, separators=(",", ":")), item["source"], item["feed"]))
            return False, True

    def record_gap(self, bucket_start_utc: datetime, detected_at_utc: datetime,
                   missing_timestamps: list[datetime], reason: str = "missing_expected_minute") -> None:
        missing_json = json.dumps([_stamp(x) for x in missing_timestamps], separators=(",", ":"))
        with self.connect() as db:
            db.execute("""INSERT OR IGNORE INTO aggregation_gaps(
                bucket_start_utc, detected_at_utc, missing_timestamps_json, reason
            ) VALUES (?, ?, ?, ?)""", (_stamp(bucket_start_utc), _stamp(detected_at_utc), missing_json, reason))

    def five_minute_count(self) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM five_minute_candles").fetchone()[0])

    def five_minute_candles_for_session(self, session_open_utc: datetime,
                                        session_close_utc: datetime) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("""SELECT candle_time_utc, finalized_at_utc, open, high, low, close,
                volume, trade_count, source, feed, aggregation_method, aggregation_metadata_json
                FROM five_minute_candles WHERE candle_time_utc>=? AND candle_time_utc<?
                ORDER BY candle_time_utc""", (_stamp(session_open_utc), _stamp(session_close_utc))).fetchall()
        return [dict(row) for row in rows]

    def five_minute_candle_at(self, candle_time_utc: datetime) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("""SELECT candle_time_utc, finalized_at_utc, open, high, low, close,
                volume, trade_count, source, feed, aggregation_method, aggregation_metadata_json
                FROM five_minute_candles WHERE candle_time_utc=?""", (_stamp(candle_time_utc),)).fetchone()
        return None if row is None else dict(row)

    def insert_prediction(self, prediction: dict[str, Any]) -> bool:
        """Append one immutable prediction, returning False for an existing candle."""
        values = {
            "candle_time_utc": _stamp(prediction["candle_time_utc"]),
            "prediction_created_utc": _stamp(prediction["prediction_created_utc"]),
            "prediction": prediction["prediction"],
            "probability_down": float(prediction["probability_down"]),
            "probability_neutral": float(prediction["probability_neutral"]),
            "probability_up": float(prediction["probability_up"]),
            "confidence": float(prediction["confidence"]),
            "reference_close": float(prediction["reference_close"]),
            "horizon_minutes": int(prediction["horizon_minutes"]),
            "outcome_candle_time_utc": _stamp(prediction["outcome_candle_time_utc"]),
            "model_sha256": prediction["model_sha256"],
            "input_features_json": json.dumps(prediction["input_features"], sort_keys=True, separators=(",", ":")),
            "provider_delay_seconds": float(prediction["provider_delay_seconds"]),
        }
        if values["outcome_candle_time_utc"] != _stamp(prediction["candle_time_utc"] + timedelta(minutes=15)):
            raise ValueError("Prediction endpoint must be exactly 15 minutes after candle time")
        with self.connect() as db:
            cursor = db.execute("""INSERT OR IGNORE INTO predictions (
                candle_time_utc, prediction_created_utc, prediction, probability_down,
                probability_neutral, probability_up, confidence, reference_close,
                horizon_minutes, outcome_candle_time_utc, model_sha256, input_features_json,
                provider_delay_seconds
            ) VALUES (:candle_time_utc, :prediction_created_utc, :prediction, :probability_down,
                :probability_neutral, :probability_up, :confidence, :reference_close,
                :horizon_minutes, :outcome_candle_time_utc, :model_sha256, :input_features_json,
                :provider_delay_seconds)""", values)
            return cursor.rowcount == 1

    def pending_predictions(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("""SELECT p.* FROM predictions p LEFT JOIN outcomes o
                ON o.prediction_candle_time_utc=p.candle_time_utc
                WHERE o.prediction_candle_time_utc IS NULL ORDER BY p.candle_time_utc""").fetchall()
        return [dict(row) for row in rows]

    def prediction_at(self, candle_time_utc: datetime) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM predictions WHERE candle_time_utc=?", (_stamp(candle_time_utc),)).fetchone()
        return None if row is None else dict(row)

    def latest_prediction(self) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM predictions ORDER BY candle_time_utc DESC LIMIT 1").fetchone()
        return None if row is None else dict(row)

    def insert_outcome(self, outcome: dict[str, Any]) -> bool:
        with self.connect() as db:
            cursor = db.execute("""INSERT OR IGNORE INTO outcomes (
                prediction_candle_time_utc, observed_utc, endpoint_candle_time_utc,
                actual_target, actual_return_pct, correct
            ) VALUES (?, ?, ?, ?, ?, ?)""", (
                _stamp(outcome["prediction_candle_time_utc"]), _stamp(outcome["observed_utc"]),
                _stamp(outcome["endpoint_candle_time_utc"]), outcome["actual_target"],
                float(outcome["actual_return_pct"]), int(bool(outcome["correct"])),
            ))
            return cursor.rowcount == 1

    def prediction_count(self) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def outcome_count(self) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0])

    def dashboard_candles(self, limit: int = 500) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit), 100000))
        with self.connect() as db:
            rows = db.execute("""SELECT candle_time_utc, open, high, low, close, volume, source, feed
                FROM five_minute_candles ORDER BY candle_time_utc DESC LIMIT ?""", (capped,)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def prediction_detail_at(self, candle_time_utc: datetime) -> dict[str, Any] | None:
        """Read one immutable prediction and its genuinely persisted outcome."""
        with self.connect() as db:
            row = db.execute("""SELECT p.candle_time_utc, p.prediction_created_utc, p.prediction,
                p.probability_down, p.probability_neutral, p.probability_up, p.confidence,
                p.reference_close, p.horizon_minutes, p.outcome_candle_time_utc, p.model_sha256,
                o.observed_utc, o.endpoint_candle_time_utc, o.actual_target,
                o.actual_return_pct, o.correct, c.close AS actual_close
                FROM predictions p LEFT JOIN outcomes o
                ON o.prediction_candle_time_utc=p.candle_time_utc
                LEFT JOIN five_minute_candles c ON c.candle_time_utc=o.endpoint_candle_time_utc
                WHERE p.candle_time_utc=?""", (_stamp(candle_time_utc),)).fetchone()
        return None if row is None else dict(row)

    def prediction_history(self, limit: int = 200) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit), 100000))
        with self.connect() as db:
            rows = db.execute("""SELECT p.candle_time_utc, p.prediction_created_utc, p.prediction,
                p.probability_down, p.probability_neutral, p.probability_up, p.confidence,
                p.reference_close, p.horizon_minutes, p.outcome_candle_time_utc, p.model_sha256,
                o.observed_utc, o.actual_target, o.actual_return_pct, o.correct
                FROM predictions p LEFT JOIN outcomes o
                ON o.prediction_candle_time_utc=p.candle_time_utc
                ORDER BY p.candle_time_utc DESC LIMIT ?""", (capped,)).fetchall()
        return [dict(row) for row in rows]

    def performance_report(self) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("""SELECT COUNT(*) AS predictions,
                SUM(CASE WHEN o.prediction_candle_time_utc IS NOT NULL THEN 1 ELSE 0 END) AS settled,
                SUM(CASE WHEN o.prediction_candle_time_utc IS NULL THEN 1 ELSE 0 END) AS pending,
                AVG(CASE WHEN o.prediction_candle_time_utc IS NOT NULL THEN o.correct END) AS accuracy_settled
                FROM predictions p LEFT JOIN outcomes o
                ON o.prediction_candle_time_utc=p.candle_time_utc""").fetchone()
        return {key: row[key] for key in row.keys()}

    def unresolved_gaps(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("""SELECT g.bucket_start_utc, g.detected_at_utc,
                g.missing_timestamps_json, g.reason FROM aggregation_gaps g
                WHERE NOT EXISTS (SELECT 1 FROM five_minute_candles c
                    WHERE c.candle_time_utc=g.bucket_start_utc)
                ORDER BY g.detected_at_utc DESC LIMIT ?""", (max(1, min(limit, 50)),)).fetchall()
        return [dict(row) for row in rows]

    def unresolved_gap_count(self) -> int:
        """Count all unresolved bucket warnings without changing validation state."""
        with self.connect() as db:
            row = db.execute("""SELECT COUNT(*) FROM aggregation_gaps g
                WHERE NOT EXISTS (SELECT 1 FROM five_minute_candles c
                    WHERE c.candle_time_utc=g.bucket_start_utc)""").fetchone()
        return int(row[0])

    def unresolved_gaps_all(self) -> list[dict[str, Any]]:
        """Return all unresolved warning details for explicit user expansion."""
        with self.connect() as db:
            rows = db.execute("""SELECT g.bucket_start_utc, g.detected_at_utc,
                g.missing_timestamps_json, g.reason FROM aggregation_gaps g
                WHERE NOT EXISTS (SELECT 1 FROM five_minute_candles c
                    WHERE c.candle_time_utc=g.bucket_start_utc)
                ORDER BY g.detected_at_utc DESC""").fetchall()
        return [dict(row) for row in rows]
