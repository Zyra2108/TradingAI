"""Bounded Alpaca IEX REST catch-up for the live chart's separate candle cache.

This cache is never used as training data and never writes prediction records.
Only regular-session, finalized native five-minute bars are accepted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from live_app.calendar import NY, session_for_date

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
CURRENT_CACHE_DIR = ROOT / "data/live/alpaca_iex_current_market"
CURRENT_CACHE_CSV = CURRENT_CACHE_DIR / "ecl_iex_5min_current.csv"
CURRENT_CACHE_MANIFEST = CURRENT_CACHE_DIR / "ecl_iex_5min_current_manifest.json"
CURRENT_CACHE_CONFLICTS = CURRENT_CACHE_DIR / "conflicts.csv"
ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
SYMBOL = "ECL"
FEED = "iex"
TIMEFRAME = "5Min"
FINALIZATION_BUFFER_SECONDS = 120
PAGE_LIMIT = 10000
MAX_PAGES = 5
MAX_RETRIES = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _latest_finalizable_bar_open(now: datetime) -> datetime | None:
    """Return the latest NYSE-aligned bar open whose close plus buffer elapsed."""
    now_utc = now.astimezone(UTC)
    latest_safe_end = now_utc - timedelta(seconds=FINALIZATION_BUFFER_SECONDS)
    for offset in range(0, 16):
        day = now_utc.astimezone(NY).date() - timedelta(days=offset)
        session = session_for_date(day)
        if not session:
            continue
        opened = session[0].astimezone(UTC)
        closed = session[1].astimezone(UTC)
        last_open = min(latest_safe_end - timedelta(minutes=5), closed - timedelta(minutes=5))
        if last_open < opened:
            continue
        elapsed = int((last_open - opened).total_seconds() // 300) * 300
        candidate = opened + timedelta(seconds=elapsed)
        if candidate + timedelta(minutes=5, seconds=FINALIZATION_BUFFER_SECONDS) <= now_utc:
            return candidate
    return None


def _read_existing_cache(csv_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists() and not manifest_path.exists():
        return []
    if not csv_path.is_file() or not manifest_path.is_file():
        raise ValueError("Current Alpaca cache is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("validation_status") != "PASS" or manifest.get("symbol") != SYMBOL
            or manifest.get("feed") != "IEX" or manifest.get("timeframe") != TIMEFRAME
            or manifest.get("timestamp_convention") != "UTC candle-open timestamps"
            or manifest.get("csv_sha256") != _sha256(csv_path)):
        raise ValueError("Current Alpaca cache provenance or integrity check failed")
    frame = pd.read_csv(csv_path)
    required = {"timestamp_utc", "open", "high", "low", "close", "volume", "trade_count"}
    if not required.issubset(frame.columns):
        raise ValueError("Current Alpaca cache has an invalid schema")
    times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Current Alpaca cache timestamps are not unique and chronological")
    result = []
    for row, stamp in zip(frame.to_dict("records"), times):
        parsed = {key: float(row[key]) for key in ("open", "high", "low", "close", "volume")}
        if (not all(math.isfinite(value) for value in parsed.values())
                or min(parsed[key] for key in ("open", "high", "low", "close")) <= 0
                or parsed["volume"] < 0
                or parsed["high"] < max(parsed["open"], parsed["low"], parsed["close"])
                or parsed["low"] > min(parsed["open"], parsed["high"], parsed["close"])):
            raise ValueError("Current Alpaca cache contains invalid OHLCV")
        result.append({"timestamp_utc": _utc_text(stamp.to_pydatetime()), **parsed,
                       "trade_count": None if pd.isna(row["trade_count"]) else int(row["trade_count"])})
    if manifest.get("row_count") != len(result):
        raise ValueError("Current Alpaca cache manifest row count does not match")
    fetched_at = pd.Timestamp(manifest.get("last_fetch_utc"))
    if fetched_at.tzinfo is None:
        raise ValueError("Current Alpaca cache fetch timestamp must be timezone-aware")
    fetched_at_utc = fetched_at.tz_convert("UTC").to_pydatetime()
    for row in result:
        stamp = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")).astimezone(UTC)
        session = session_for_date(stamp.astimezone(NY).date())
        if not session:
            raise ValueError("Current Alpaca cache contains an out-of-session timestamp")
        opened, closed = (bound.astimezone(UTC) for bound in session)
        if (stamp < opened or stamp + timedelta(minutes=5) > closed
                or (stamp - opened).total_seconds() % 300 != 0
                or stamp + timedelta(minutes=5, seconds=FINALIZATION_BUFFER_SECONDS) > fetched_at_utc):
            raise ValueError("Current Alpaca cache contains an incomplete or misaligned candle")
    return result


class AlpacaCurrentBarsRefresher:
    """Fetch finalized IEX 5Min bars with bounded pagination and append-only writes."""

    def __init__(self, settings: Any, *, session: Any | None = None,
                 csv_path: Path = CURRENT_CACHE_CSV,
                 manifest_path: Path = CURRENT_CACHE_MANIFEST,
                 conflict_path: Path = CURRENT_CACHE_CONFLICTS,
                 sleep: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.session = session or requests.Session()
        self.csv_path, self.manifest_path = Path(csv_path), Path(manifest_path)
        self.conflict_path = Path(conflict_path)
        self.sleep = sleep
        self._status: dict[str, Any] = {"state": "not_started", "last_success_utc": None,
                                        "latest_bar_utc": None, "bars_added": 0,
                                        "last_safe_error": None}

    @property
    def status(self) -> dict[str, Any]:
        return dict(self._status)

    def refresh(self, existing_candles: list[dict[str, Any]], *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Refresh clock must be timezone-aware")
        if not self.settings.credentials_configured:
            self._status.update(state="credentials_missing", last_safe_error="credentials_missing")
            raise RuntimeError("Alpaca credentials are not configured")
        latest_allowed = _latest_finalizable_bar_open(now)
        if latest_allowed is None:
            self._status.update(state="no_completed_session_bar", bars_added=0, last_safe_error=None)
            return self.status
        latest_existing = max((int(row["time"]) for row in existing_candles), default=None)
        if latest_existing is not None:
            # Re-query the preceding finalized bucket too. This reconciles a
            # recent stream gap and detects late revisions without rewriting
            # an already accepted observation.
            latest_dt = datetime.fromtimestamp(latest_existing, UTC)
            session = session_for_date(latest_dt.astimezone(NY).date())
            session_open = session[0].astimezone(UTC) if session else latest_dt
            start = max(session_open, latest_dt - timedelta(minutes=5))
        else:
            session = session_for_date(latest_allowed.astimezone(NY).date())
            start = session[0].astimezone(UTC) if session else latest_allowed
        if start > latest_allowed:
            self._status.update(state="current", bars_added=0, last_safe_error=None)
            return self.status

        headers = {"APCA-API-KEY-ID": self.settings.api_key_id,
                   "APCA-API-SECRET-KEY": self.settings.api_secret_key}
        params: dict[str, Any] = {"symbols": SYMBOL, "timeframe": TIMEFRAME, "feed": FEED,
            "start": _utc_text(start), "end": _utc_text(latest_allowed), "limit": PAGE_LIMIT, "sort": "asc"}
        raw: list[dict[str, Any]] = []
        token = None
        seen_tokens: set[str] = set()
        page_count = 0
        for _ in range(MAX_PAGES):
            page_params = dict(params)
            if token:
                page_params["page_token"] = token
            response = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.session.get(ENDPOINT, headers=headers, params=page_params, timeout=(10, 45))
                except requests.RequestException:
                    if attempt + 1 == MAX_RETRIES:
                        self._status.update(state="request_failed", last_safe_error="network_error")
                        raise RuntimeError("Alpaca IEX historical request failed") from None
                    self.sleep(2 ** attempt)
                    continue
                if response.status_code == 429 or response.status_code in {500, 502, 503, 504}:
                    if attempt + 1 == MAX_RETRIES:
                        category = "rate_limited" if response.status_code == 429 else "provider_unavailable"
                        self._status.update(state="request_failed", last_safe_error=category)
                        raise RuntimeError(f"Alpaca IEX historical request stopped after HTTP {response.status_code}")
                    delay = response.headers.get("Retry-After")
                    try:
                        wait = max(float(delay), 2 ** attempt) if delay else 2 ** attempt
                    except ValueError:
                        wait = 2 ** attempt
                    self.sleep(min(wait, 30))
                    continue
                break
            if response is None:
                raise RuntimeError("Alpaca IEX historical request returned no response")
            if response.status_code != 200:
                category = {401: "authentication_failed", 403: "feed_or_permission_denied"}.get(
                    response.status_code, "provider_request_rejected")
                self._status.update(state="request_failed", last_safe_error=category)
                raise RuntimeError(f"Alpaca IEX historical request failed with HTTP {response.status_code}")
            try:
                payload = response.json()
            except ValueError:
                self._status.update(state="request_failed", last_safe_error="invalid_json")
                raise RuntimeError("Alpaca IEX historical response was not valid JSON") from None
            bars = payload.get("bars", {}).get(SYMBOL, [])
            if not isinstance(bars, list):
                self._status.update(state="request_failed", last_safe_error="invalid_response")
                raise RuntimeError("Alpaca IEX response had an invalid bar list")
            raw.extend(bars)
            page_count += 1
            next_token = payload.get("next_page_token")
            if not next_token:
                break
            if not isinstance(next_token, str) or next_token in seen_tokens:
                raise RuntimeError("Alpaca IEX pagination token repeated or was invalid")
            seen_tokens.add(next_token)
            token = next_token
        else:
            raise RuntimeError("Alpaca IEX pagination exceeded its bounded page limit")

        parsed = self._validate_rows(raw, start, latest_allowed, now)
        prior = _read_existing_cache(self.csv_path, self.manifest_path)
        by_stamp = {row["timestamp_utc"]: row for row in prior}
        conflicts = []
        added = 0
        for row in parsed:
            old = by_stamp.get(row["timestamp_utc"])
            if old is not None:
                if any(not math.isclose(float(old[k]), float(row[k]), rel_tol=0, abs_tol=1e-8)
                       for k in ("open", "high", "low", "close", "volume")):
                    conflicts.append({"timestamp_utc": row["timestamp_utc"], "existing": old, "incoming": row})
                continue
            by_stamp[row["timestamp_utc"]] = row
            added += 1
        baseline = {int(row["time"]): row for row in existing_candles}
        for row in parsed:
            stamp = int(pd.Timestamp(row["timestamp_utc"]).timestamp())
            old = baseline.get(stamp)
            if old is not None and any(not math.isclose(float(old[key]), float(row[key]), rel_tol=0, abs_tol=1e-8)
                                       for key in ("open", "high", "low", "close", "volume")):
                conflicts.append({"timestamp_utc": row["timestamp_utc"], "existing": old, "incoming": row})
        if conflicts:
            self._write_conflicts(conflicts)
            self._status.update(state="conflict", bars_added=0, last_safe_error="historical_bar_conflict")
            raise RuntimeError("Alpaca IEX returned a conflicting existing candle; cache was not changed")

        merged = [by_stamp[key] for key in sorted(by_stamp)]
        if added:
            self._write_cache(merged, now, start, latest_allowed, page_count)
        latest = merged[-1]["timestamp_utc"] if merged else None
        self._status.update(state="current", last_success_utc=_utc_text(now), latest_bar_utc=latest,
                            bars_added=added, last_safe_error=None)
        return {**self.status, "requested_start_utc": _utc_text(start),
                "eligible_through_utc": _utc_text(latest_allowed), "pages": page_count,
                "returned_rows": len(raw), "validated_rows": len(parsed)}

    def _validate_rows(self, raw: list[dict[str, Any]], start: datetime, end: datetime,
                       now: datetime) -> list[dict[str, Any]]:
        by_stamp: dict[str, dict[str, Any]] = {}
        for item in raw:
            try:
                stamp = pd.Timestamp(item["t"])
                if stamp.tzinfo is None:
                    raise ValueError
                stamp = stamp.tz_convert("UTC")
                dt = stamp.to_pydatetime()
                o, h, low, close, volume = (float(item[key]) for key in ("o", "h", "l", "c", "v"))
            except (KeyError, TypeError, ValueError, OverflowError):
                raise RuntimeError("Alpaca IEX returned a malformed bar") from None
            if dt < start or dt > end:
                raise RuntimeError("Alpaca IEX returned a bar outside the requested completed interval")
            session = session_for_date(dt.astimezone(NY).date())
            if not session:
                raise RuntimeError("Alpaca IEX returned a bar outside an NYSE session")
            opened, closed = (bound.astimezone(UTC) for bound in session)
            if (dt < opened or dt + timedelta(minutes=5) > closed
                    or (dt - opened).total_seconds() % 300 != 0):
                raise RuntimeError("Alpaca IEX returned a misaligned or out-of-session bar")
            if (not all(math.isfinite(v) for v in (o, h, low, close, volume))
                    or min(o, h, low, close) <= 0 or volume < 0
                    or h < max(o, low, close) or low > min(o, h, close)):
                raise RuntimeError("Alpaca IEX returned invalid OHLCV")
            if dt + timedelta(minutes=5, seconds=FINALIZATION_BUFFER_SECONDS) > now.astimezone(UTC):
                raise RuntimeError("Alpaca IEX returned a candle before the finalization buffer elapsed")
            trade_count = item.get("n")
            trade_count = None if trade_count is None else int(trade_count)
            if trade_count is not None and trade_count < 0:
                raise RuntimeError("Alpaca IEX returned an invalid trade count")
            key = _utc_text(dt)
            row = {"timestamp_utc": key, "open": o, "high": h, "low": low,
                   "close": close, "volume": volume, "trade_count": trade_count}
            old = by_stamp.get(key)
            if old is not None and any(old[k] != row[k] for k in row if k != "timestamp_utc"):
                raise RuntimeError("Alpaca IEX returned conflicting duplicate timestamps")
            by_stamp[key] = row
        return [by_stamp[key] for key in sorted(by_stamp)]

    def _write_conflicts(self, conflicts: list[dict[str, Any]]) -> None:
        import csv
        self.conflict_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.conflict_path.exists()
        with self.conflict_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["timestamp_utc", "existing_json", "incoming_json"])
            if not exists:
                writer.writeheader()
            for row in conflicts:
                writer.writerow({"timestamp_utc": row["timestamp_utc"],
                    "existing_json": json.dumps(row["existing"], sort_keys=True),
                    "incoming_json": json.dumps(row["incoming"], sort_keys=True)})

    def _write_cache(self, rows: list[dict[str, Any]], now: datetime, start: datetime,
                     end: datetime, page_count: int) -> None:
        import csv
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        temp_csv = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        columns = ["timestamp_utc", "open", "high", "low", "close", "volume", "trade_count"]
        with temp_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_csv, self.csv_path)
        old_manifest = {}
        if self.manifest_path.is_file():
            old_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest = {"schema_version": 1, "validation_status": "PASS",
            "source": "Alpaca historical stock bars API", "symbol": SYMBOL, "feed": "IEX",
            "timeframe": TIMEFRAME, "timestamp_convention": "UTC candle-open timestamps",
            "cache_csv": self.csv_path.name, "csv_sha256": _sha256(self.csv_path),
            "row_count": len(rows), "first_timestamp_utc": rows[0]["timestamp_utc"] if rows else None,
            "last_timestamp_utc": rows[-1]["timestamp_utc"] if rows else None,
            "first_fetch_utc": old_manifest.get("first_fetch_utc", _utc_text(now)),
            "last_fetch_utc": _utc_text(now), "requested_start_utc": _utc_text(start),
            "eligible_through_utc": _utc_text(end), "pagination_pages": page_count,
            "fetch_count": int(old_manifest.get("fetch_count", 0)) + 1,
            "missing_intervals": "Preserved as absent observations; no candles are synthesized."}
        temp_manifest = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_manifest, self.manifest_path)


def load_verified_current_cache(csv_path: Path = CURRENT_CACHE_CSV,
                                manifest_path: Path = CURRENT_CACHE_MANIFEST) -> list[dict[str, Any]]:
    """Read the independently manifested current cache for the unified chart."""
    rows = _read_existing_cache(Path(csv_path), Path(manifest_path))
    result = []
    for row in rows:
        stamp = pd.Timestamp(row["timestamp_utc"])
        epoch = int(stamp.timestamp())
        result.append({"time": epoch, "candle_time_utc": row["timestamp_utc"],
            "open": row["open"], "high": row["high"], "low": row["low"],
            "close": row["close"], "volume": row["volume"], "trade_count": row["trade_count"],
            "source": "historical_current_cache", "feed": "IEX"})
    return result
