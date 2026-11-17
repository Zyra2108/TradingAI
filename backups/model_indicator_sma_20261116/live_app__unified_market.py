"""Read-only verified Alpaca IEX candles for the unified dashboard timeline.

The original pre-test train/validation cache and a separately staged native
five-minute backfill are integrity-checked. Empty buckets are never created.
No model artifact or test dataset is loaded by this module.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from live_app.calendar import NY, session_for_date

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = ROOT / "models/alpaca_1min_research_v1/raw_ecl_iex_1min_train_validation.csv"
DEFAULT_METADATA_PATH = DEFAULT_RAW_PATH.with_name("fetch_metadata.json")
FIRST_ALLOWED_DATE = date(2025, 11, 13)
LAST_ALLOWED_DATE = date(2026, 9, 24)
BACKFILL_START_DATE = date(2026, 9, 25)
BACKFILL_END_DATE = date(2026, 11, 12)
BACKFILL_DIR = ROOT / "data/alpaca_backfill_v1"
BACKFILL_CSV = BACKFILL_DIR / "ecl_iex_5min_20260925_20261112.csv"
BACKFILL_MANIFEST = BACKFILL_DIR / "ecl_iex_5min_20260925_20261112_manifest.json"
EXPECTED_SHA256 = "7d5bce7c8925b9cb1f27aba122f9ce97b0459a1bdce826bdee1f70f3d109880c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UnifiedHistoricalMarket:
    """Load the known permitted cache and expose real regular-session candles."""

    def __init__(self, raw_path: Path | str = DEFAULT_RAW_PATH,
                 metadata_path: Path | str | None = None):
        self.raw_path = Path(raw_path).resolve()
        self.metadata_path = Path(metadata_path).resolve() if metadata_path else self.raw_path.with_name("fetch_metadata.json")
        base_candles, self.excluded_out_of_session = self._load()
        backfill_candles, overlap_count = self._load_verified_backfill(base_candles)
        by_time = {int(row["time"]): row for row in base_candles}
        exact_duplicates = 0
        for row in backfill_candles:
            stamp = int(row["time"])
            old = by_time.get(stamp)
            if old is not None:
                if any(not np.isclose(float(old[key]), float(row[key]), rtol=0, atol=1e-8)
                       for key in ("open", "high", "low", "close", "volume")):
                    raise ValueError("Verified backfill conflicts with an existing timeline candle")
                exact_duplicates += 1
                continue
            by_time[stamp] = row
        self._candles = [by_time[key] for key in sorted(by_time)]
        self.backfill_overlap_timestamps = overlap_count
        self.backfill_exact_duplicate_count = exact_duplicates
        self.backfill_candle_count = len(backfill_candles) - exact_duplicates
        self.coverage = {
            "source": "Alpaca IEX historical 1-minute cache" + (" plus verified 5-minute backfill" if backfill_candles else ""),
            "feed": "IEX",
            "first_allowed_date_et": FIRST_ALLOWED_DATE.isoformat(),
            "last_allowed_date_et": BACKFILL_END_DATE.isoformat() if backfill_candles else LAST_ALLOWED_DATE.isoformat(),
            "first_candle_utc": self._candles[0]["candle_time_utc"] if self._candles else None,
            "last_candle_utc": self._candles[-1]["candle_time_utc"] if self._candles else None,
            "candle_count": len(self._candles),
            "excluded_out_of_session_minutes": self.excluded_out_of_session,
            "backfill_candle_count": self.backfill_candle_count,
            "backfill_overlap_timestamps": self.backfill_overlap_timestamps,
            "backfill_exact_duplicates": self.backfill_exact_duplicate_count,
            "backfill_manifest_status": "PASS" if backfill_candles else "not_present",
            "note": "Available verified Alpaca IEX history only; this is not all-time ECL history.",
        }

    @staticmethod
    def _load_verified_backfill(existing: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        if not BACKFILL_CSV.exists() and not BACKFILL_MANIFEST.exists():
            return [], 0
        if not BACKFILL_CSV.is_file() or not BACKFILL_MANIFEST.is_file():
            raise ValueError("Historical backfill is incomplete; both CSV and manifest are required")
        manifest = json.loads(BACKFILL_MANIFEST.read_text(encoding="utf-8"))
        if (manifest.get("validation_status") != "PASS" or manifest.get("symbol") != "ECL"
                or manifest.get("feed") != "iex" or manifest.get("timeframe") != "5Min"
                or manifest.get("requested_date_range_et_inclusive") != [BACKFILL_START_DATE.isoformat(), BACKFILL_END_DATE.isoformat()]
                or manifest.get("staging_csv") != BACKFILL_CSV.name
                or manifest.get("staging_csv_sha256") != _sha256(BACKFILL_CSV)):
            raise ValueError("Historical backfill provenance or integrity check failed")
        frame = pd.read_csv(BACKFILL_CSV)
        required = {"timestamp_utc", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            raise ValueError("Historical backfill has an invalid schema")
        times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
        if times.duplicated().any() or not times.is_monotonic_increasing:
            raise ValueError("Historical backfill timestamps must be unique and chronological")
        local = times.dt.tz_convert("America/New_York")
        dates = local.dt.date
        if ((dates < BACKFILL_START_DATE) | (dates > BACKFILL_END_DATE)).any():
            raise ValueError("Historical backfill contains dates outside its approved range")
        prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="raise")
        values = prices.to_numpy(dtype=float)
        volume = pd.to_numeric(frame["volume"], errors="raise").to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values <= 0).any()
                or (prices.high < prices[["open", "close", "low"]].max(axis=1)).any()
                or (prices.low > prices[["open", "high", "close"]].min(axis=1)).any()
                or not np.isfinite(volume).all() or (volume < 0).any()):
            raise ValueError("Historical backfill contains invalid OHLCV")
        schedule = mcal.get_calendar("NYSE").schedule(start_date=BACKFILL_START_DATE, end_date=BACKFILL_END_DATE)
        sessions = {day.date(): (row.market_open.tz_convert("UTC"), row.market_close.tz_convert("UTC"))
                    for day, row in schedule.iterrows()}
        for stamp, day in zip(times, dates):
            bounds = sessions.get(day)
            if bounds is None:
                raise ValueError("Historical backfill contains a non-session date")
            opened, closed = bounds
            if stamp < opened or stamp + pd.Timedelta(minutes=5) > closed:
                raise ValueError("Historical backfill contains an out-of-session candle")
            if (stamp - opened).total_seconds() % 300 != 0 or stamp.second != 0 or stamp.microsecond != 0:
                raise ValueError("Historical backfill candle is not aligned to the NYSE session")
        existing_by_time = {int(row["time"]): row for row in existing}
        overlaps = 0
        candles: list[dict[str, Any]] = []
        for row, stamp in zip(frame.itertuples(index=False), times):
            utc = stamp.to_pydatetime().astimezone(timezone.utc)
            epoch = int(utc.timestamp())
            old = existing_by_time.get(epoch)
            if old is not None:
                overlaps += 1
                if any(not np.isclose(float(old[key]), float(getattr(row, key)), rtol=0, atol=1e-8)
                       for key in ("open", "high", "low", "close", "volume")):
                    raise ValueError("Historical backfill overlap has conflicting OHLCV values")
            trade_count = getattr(row, "trade_count", None)
            if pd.isna(trade_count):
                trade_count = None
            candles.append({"time": epoch, "candle_time_utc": utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "open": float(row.open), "high": float(row.high), "low": float(row.low), "close": float(row.close),
                "volume": float(row.volume), "trade_count": None if trade_count is None else int(trade_count),
                "source": "historical", "feed": "IEX", "historical_source": "verified_5min_backfill"})
        if manifest.get("retrieved_unique_candles") != len(candles):
            raise ValueError("Historical backfill manifest candle count does not match its CSV")
        return candles, overlaps

    def _load(self) -> tuple[list[dict[str, Any]], int]:
        if not self.raw_path.is_file() or not self.metadata_path.is_file():
            raise FileNotFoundError("Permitted historical ECL cache or provenance metadata is missing")
        if self.raw_path != DEFAULT_RAW_PATH.resolve():
            raise ValueError("Historical timeline may only read the approved train/validation cache")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        permitted = metadata.get("permitted_new_york_date_range")
        if (metadata.get("symbol") != "ECL" or metadata.get("feed") != "iex"
                or metadata.get("timeframe") != "1Min"
                or permitted != [FIRST_ALLOWED_DATE.isoformat(), LAST_ALLOWED_DATE.isoformat()]
                or metadata.get("raw_csv_sha256") != EXPECTED_SHA256
                or _sha256(self.raw_path) != EXPECTED_SHA256):
            raise ValueError("Historical cache provenance or integrity check failed")

        frame = pd.read_csv(self.raw_path)
        required = {"timestamp_utc", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            raise ValueError("Historical ECL cache has an invalid schema")
        times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
        if times.duplicated().any() or not times.is_monotonic_increasing:
            raise ValueError("Historical ECL timestamps must be unique and chronological")
        local = times.dt.tz_convert("America/New_York")
        dates = local.dt.date
        if ((dates < FIRST_ALLOWED_DATE) | (dates > LAST_ALLOWED_DATE)).any():
            raise ValueError("Historical ECL cache contains data outside the permitted pre-test range")
        prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="raise")
        values = prices.to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values <= 0).any()
                or (prices.high < prices[["open", "close", "low"]].max(axis=1)).any()
                or (prices.low > prices[["open", "close", "high"]].min(axis=1)).any()):
            raise ValueError("Historical ECL cache contains invalid OHLC")
        volume = pd.to_numeric(frame["volume"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(volume).all() or (volume < 0).any():
            raise ValueError("Historical ECL cache contains invalid volume")
        if (times.dt.second.ne(0) | times.dt.microsecond.ne(0)).any():
            raise ValueError("Historical ECL minute timestamps are not minute-aligned")

        days = list(set(dates))
        schedule = mcal.get_calendar("NYSE").schedule(start_date=min(days), end_date=max(days))
        sessions = {}
        for session_day, session_row in schedule.iterrows():
            sessions[session_day.date()] = (
                session_row.market_open.tz_convert("America/New_York").to_pydatetime(),
                session_row.market_close.tz_convert("America/New_York").to_pydatetime(),
            )
        keep_parts: list[np.ndarray] = []
        bucket_parts: list[pd.DatetimeIndex] = []
        excluded = 0
        for day in days:
            indices = np.flatnonzero(dates.to_numpy() == day)
            bounds = sessions.get(day)
            if bounds is None:
                excluded += len(indices)
                continue
            opened, closed = bounds
            opened_utc = pd.Timestamp(opened.astimezone(timezone.utc))
            closed_utc = pd.Timestamp(closed.astimezone(timezone.utc))
            day_times = pd.DatetimeIndex(times.iloc[indices])
            valid = (day_times >= opened_utc) & (day_times < closed_utc)
            valid_offsets = np.flatnonzero(valid)
            excluded += len(indices) - len(valid_offsets)
            if not len(valid_offsets):
                continue
            valid_indices = indices[valid_offsets]
            valid_times = day_times[valid_offsets]
            elapsed_seconds = (valid_times - opened_utc).total_seconds()
            bucket_offsets = np.floor(elapsed_seconds / 300).astype(np.int64) * 300
            bucket_times = opened_utc + pd.to_timedelta(bucket_offsets, unit="s")
            inside = bucket_times + pd.Timedelta(minutes=5) <= closed_utc
            excluded += int((~inside).sum())
            keep_parts.append(valid_indices[inside])
            bucket_parts.append(pd.DatetimeIndex(bucket_times[inside]))

        if not keep_parts:
            raise ValueError("Historical cache contains no valid regular-session candles")
        keep = np.concatenate(keep_parts)
        bucket_stamps = pd.DatetimeIndex(np.concatenate([part.to_numpy() for part in bucket_parts]))
        chronological = np.argsort(keep, kind="stable")
        keep = keep[chronological]
        bucket_stamps = bucket_stamps.take(chronological)
        selected = frame.iloc[keep].copy().reset_index(drop=True)
        selected["minute_utc"] = pd.DatetimeIndex(times.iloc[keep])
        selected["bucket_utc"] = bucket_stamps
        candles: list[dict[str, Any]] = []
        aggregated = selected.groupby("bucket_utc", sort=True, as_index=False).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"), observed_minute_count=("minute_utc", "count"),
        )
        for row in aggregated.itertuples(index=False):
            bucket = row.bucket_utc
            stamp = bucket.to_pydatetime().astimezone(timezone.utc)
            candles.append({
                "time": int(stamp.timestamp()),
                "candle_time_utc": stamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "open": float(row.open), "high": float(row.high), "low": float(row.low),
                "close": float(row.close), "volume": float(row.volume),
                "source": "historical", "feed": "IEX",
                "observed_minute_count": int(row.observed_minute_count),
            })
        return candles, excluded

    def candles(self) -> list[dict[str, Any]]:
        return [dict(candle) for candle in self._candles]
