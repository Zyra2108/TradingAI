"""Fetch and validate only the missing ECL IEX five-minute timeline interval.

Credentials are read in memory from the existing config. This script never
prints credentials, headers, response bodies, or pagination tokens.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pandas_market_calendars as mcal
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ALPACA_SETTINGS
from live_app.calendar import NY
from live_app.unified_market import UnifiedHistoricalMarket

ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
SYMBOL = "ECL"
FEED = "iex"
TIMEFRAME = "5Min"
START_DATE_ET = date(2026, 9, 25)
END_DATE_ET = date(2026, 11, 12)
START_UTC = "2026-09-25T04:00:00Z"  # midnight EDT at the beginning of the request
END_UTC = "2026-11-13T04:59:59Z"    # inclusive end of Nov 12 EST
OUTPUT_DIR = ROOT / "data/alpaca_backfill_v1"
STAGING_CSV = OUTPUT_DIR / "ecl_iex_5min_20260925_20261112.csv"
MANIFEST_JSON = OUTPUT_DIR / "ecl_iex_5min_20260925_20261112_manifest.json"
REPORT_JSON = OUTPUT_DIR / "ecl_iex_5min_20260925_20261112_quality_report.json"
GAPS_CSV = OUTPUT_DIR / "ecl_iex_5min_20260925_20261112_missing_intervals.csv"
MAX_PAGES = 20
PAGE_LIMIT = 10000
MAX_RETRIES = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("bar timestamp is not timezone-aware")
    return stamp.tz_convert("UTC")


def _expected_session_bars() -> tuple[dict[date, tuple[datetime, datetime]], dict[date, list[pd.Timestamp]]]:
    schedule = mcal.get_calendar("NYSE").schedule(start_date=START_DATE_ET, end_date=END_DATE_ET)
    bounds: dict[date, tuple[datetime, datetime]] = {}
    expected: dict[date, list[pd.Timestamp]] = {}
    for session_day, row in schedule.iterrows():
        day = session_day.date()
        opened_utc = row.market_open.tz_convert("UTC").to_pydatetime()
        closed_utc = row.market_close.tz_convert("UTC").to_pydatetime()
        bounds[day] = (opened_utc, closed_utc)
        points = []
        cursor = opened_utc
        while cursor + timedelta(minutes=5) <= closed_utc:
            points.append(pd.Timestamp(cursor))
            cursor += timedelta(minutes=5)
        expected[day] = points
    return bounds, expected


def _parse_api_bars(raw_bars: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    bounds, expected = _expected_session_bars()
    parsed: list[dict[str, Any]] = []
    counts = {"invalid_timestamps": 0, "out_of_range": 0, "out_of_session": 0,
              "misaligned_timestamps": 0, "invalid_ohlc": 0, "invalid_volume": 0,
              "duplicate_rows": 0, "conflicting_duplicates": 0, "out_of_order_rows": 0}
    seen: dict[pd.Timestamp, dict[str, Any]] = {}
    previous: pd.Timestamp | None = None
    for raw in raw_bars:
        try:
            stamp = _parse_timestamp(str(raw["t"]))
        except Exception:
            counts["invalid_timestamps"] += 1
            continue
        if previous is not None and stamp < previous:
            counts["out_of_order_rows"] += 1
        previous = stamp
        local_day = stamp.tz_convert("America/New_York").date()
        if not START_DATE_ET <= local_day <= END_DATE_ET:
            counts["out_of_range"] += 1
            continue
        session = bounds.get(local_day)
        if session is None:
            counts["out_of_session"] += 1
            continue
        opened, closed = session
        opened_ts, closed_ts = pd.Timestamp(opened), pd.Timestamp(closed)
        if stamp < opened_ts or stamp + pd.Timedelta(minutes=5) > closed_ts:
            counts["out_of_session"] += 1
            continue
        if (stamp - opened_ts).total_seconds() % 300 != 0 or stamp.second != 0 or stamp.microsecond != 0:
            counts["misaligned_timestamps"] += 1
            continue
        try:
            o, h, low, c = (float(raw[k]) for k in ("o", "h", "l", "c"))
            v = float(raw["v"])
            n = raw.get("n")
            trade_count = None if n is None else int(n)
            valid_ohlc = all(math.isfinite(x) and x > 0 for x in (o, h, low, c)) and h >= max(o, low, c) and low <= min(o, h, c)
            valid_volume = math.isfinite(v) and v >= 0 and (trade_count is None or trade_count >= 0)
        except (TypeError, ValueError, OverflowError):
            valid_ohlc, valid_volume = False, False
            o = h = low = c = v = float("nan")
            trade_count = None
        if not valid_ohlc:
            counts["invalid_ohlc"] += 1
            continue
        if not valid_volume:
            counts["invalid_volume"] += 1
            continue
        row = {"timestamp_utc": stamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
               "open": o, "high": h, "low": low, "close": c, "volume": v,
               "trade_count": trade_count}
        if stamp in seen:
            counts["duplicate_rows"] += 1
            old = seen[stamp]
            if any(old[key] != row[key] for key in ("open", "high", "low", "close", "volume", "trade_count")):
                counts["conflicting_duplicates"] += 1
            continue
        seen[stamp] = row
        parsed.append(row)
    parsed.sort(key=lambda row: row["timestamp_utc"])
    if counts["conflicting_duplicates"]:
        raise ValueError("API returned conflicting duplicate timestamps")
    invalid_total = sum(counts[key] for key in ("invalid_timestamps", "out_of_range", "out_of_session",
        "misaligned_timestamps", "invalid_ohlc", "invalid_volume"))
    if invalid_total:
        raise ValueError("API data failed timestamp, session, OHLC, or volume validation")
    return parsed, counts


def _quality_report(rows: list[dict[str, Any]], raw_count: int, counts: dict[str, int], statuses: list[int], retries: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    bounds, expected = _expected_session_bars()
    present = {pd.Timestamp(row["timestamp_utc"]) for row in rows}
    missing = []
    for day, stamps in expected.items():
        for stamp in stamps:
            if stamp not in present:
                missing.append({"trading_date_et": day.isoformat(), "missing_timestamp_utc": stamp.isoformat().replace("+00:00", "Z"),
                                "missing_timestamp_et": stamp.tz_convert("America/New_York").isoformat(timespec="seconds")})
    covered_dates = sorted({pd.Timestamp(row["timestamp_utc"]).tz_convert("America/New_York").date().isoformat() for row in rows})
    all_dates = [day.isoformat() for day in expected]
    unavailable_dates = [day for day in all_dates if day not in set(covered_dates)]
    by_date = {day: sum(1 for row in rows if pd.Timestamp(row["timestamp_utc"]).tz_convert("America/New_York").date().isoformat() == day)
               for day in all_dates}
    report = {
        "validation_status": "PASS",
        "requested_date_range_et_inclusive": [START_DATE_ET.isoformat(), END_DATE_ET.isoformat()],
        "requested_timestamp_range_utc_inclusive": [START_UTC, END_UTC],
        "actual_returned_date_range_et": [covered_dates[0], covered_dates[-1]] if covered_dates else None,
        "source": "Alpaca historical stock bars API", "symbol": SYMBOL, "feed": FEED.upper(), "timeframe": TIMEFRAME,
        "endpoint": ENDPOINT, "http_statuses": statuses, "page_count": len(statuses), "retry_count": retries,
        "api_rows_received": raw_count, "unique_valid_candles": len(rows), "trading_dates_covered": len(covered_dates),
        "trading_dates": covered_dates, "trading_dates_without_bars": unavailable_dates,
        "duplicate_rows": counts["duplicate_rows"], "conflicting_duplicates": counts["conflicting_duplicates"],
        "out_of_order_rows": counts["out_of_order_rows"],
        "invalid_timestamps": counts["invalid_timestamps"], "out_of_range_rows": counts["out_of_range"],
        "out_of_session_rows": counts["out_of_session"], "misaligned_timestamps": counts["misaligned_timestamps"],
        "invalid_ohlc_rows": counts["invalid_ohlc"], "invalid_volume_rows": counts["invalid_volume"],
        "expected_regular_session_buckets": sum(map(len, expected.values())),
        "missing_expected_iex_buckets": len(missing),
        "missing_intervals_note": "Missing IEX buckets are reported as absent feed observations; they are not assumed to be stream interruptions or market-wide inactivity.",
        "candles_per_trading_date": by_date,
        "first_timestamp_utc": rows[0]["timestamp_utc"] if rows else None,
        "last_timestamp_utc": rows[-1]["timestamp_utc"] if rows else None,
        "first_timestamp_et": pd.Timestamp(rows[0]["timestamp_utc"]).tz_convert("America/New_York").isoformat() if rows else None,
        "last_timestamp_et": pd.Timestamp(rows[-1]["timestamp_utc"]).tz_convert("America/New_York").isoformat() if rows else None,
    }
    return report, missing


def fetch_pages(session: Any, settings: Any, *, sleep: Callable[[float], None] = time.sleep) -> tuple[list[dict[str, Any]], list[int], int]:
    if not settings.credentials_configured:
        raise RuntimeError("Alpaca credentials are not configured")
    headers = {"APCA-API-KEY-ID": settings.api_key_id, "APCA-API-SECRET-KEY": settings.api_secret_key}
    params: dict[str, Any] = {"symbols": SYMBOL, "timeframe": TIMEFRAME, "feed": FEED,
        "start": START_UTC, "end": END_UTC, "limit": PAGE_LIMIT, "sort": "asc"}
    page_token = None
    seen_tokens: set[str] = set()
    raw_rows: list[dict[str, Any]] = []
    statuses: list[int] = []
    retry_count = 0
    for _ in range(MAX_PAGES):
        page_params = dict(params)
        if page_token:
            page_params["page_token"] = page_token
        response = None
        for attempt in range(MAX_RETRIES):
            try:
                response = session.get(ENDPOINT, headers=headers, params=page_params, timeout=(10, 60))
            except requests.RequestException:
                if attempt + 1 >= MAX_RETRIES:
                    raise RuntimeError("Alpaca historical request failed after bounded retries") from None
                retry_count += 1
                sleep(min(2 ** attempt, 30))
                continue
            if response.status_code == 429 or response.status_code in {500, 502, 503, 504}:
                if attempt + 1 >= MAX_RETRIES:
                    raise RuntimeError(f"Alpaca historical request stopped after HTTP {response.status_code}")
                retry_count += 1
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = max(float(retry_after), 2 ** attempt) if retry_after else float(2 ** attempt)
                except ValueError:
                    delay = float(2 ** attempt)
                sleep(min(delay, 60))
                continue
            break
        if response is None:
            raise RuntimeError("Alpaca historical request returned no response")
        statuses.append(int(response.status_code))
        if response.status_code != 200:
            raise RuntimeError(f"Alpaca historical request failed with HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError("Alpaca historical response was not valid JSON") from None
        bars_by_symbol = payload.get("bars")
        if not isinstance(bars_by_symbol, dict):
            raise RuntimeError("Alpaca historical response had an unexpected bar structure")
        bars = bars_by_symbol.get(SYMBOL, []) or []
        if not isinstance(bars, list):
            raise RuntimeError("Alpaca historical ECL bars were not a list")
        raw_rows.extend(bars)
        next_token = payload.get("next_page_token")
        if not next_token:
            return raw_rows, statuses, retry_count
        if not isinstance(next_token, str) or next_token in seen_tokens:
            raise RuntimeError("Alpaca pagination token repeated or was invalid")
        seen_tokens.add(next_token)
        page_token = next_token
    raise RuntimeError(f"Alpaca pagination exceeded the bounded {MAX_PAGES}-page limit")


def run_backfill(*, session: Any | None = None, settings: Any = ALPACA_SETTINGS,
                 sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    if OUTPUT_DIR.exists():
        raise FileExistsError("Backfill staging directory already exists; refusing to overwrite it")
    if session is None:
        session = requests.Session()
    raw_rows, statuses, retries = fetch_pages(session, settings, sleep=sleep)
    if not raw_rows:
        raise RuntimeError("Alpaca IEX returned no ECL five-minute bars for the requested interval")
    rows, counts = _parse_api_bars(raw_rows)
    if not rows:
        raise RuntimeError("No validated regular-session ECL bars remain")

    # Compare every returned candle with the existing permitted timeline before staging.
    existing = UnifiedHistoricalMarket().candles()
    existing_by_time = {row["candle_time_utc"]: row for row in existing}
    overlap_checks = 0
    overlap_conflicts = 0
    for row in rows:
        old = existing_by_time.get(row["timestamp_utc"])
        if old is None:
            continue
        overlap_checks += 1
        for key in ("open", "high", "low", "close", "volume"):
            if not math.isclose(float(old[key]), float(row[key]), rel_tol=0, abs_tol=1e-8):
                overlap_conflicts += 1
                break
    if overlap_conflicts:
        raise RuntimeError("Backfill has conflicting OHLCV values at existing timeline timestamps")
    report, missing = _quality_report(rows, len(raw_rows), counts, statuses, retries)
    report["existing_timeline_overlap_timestamps"] = overlap_checks
    report["existing_timeline_overlap_conflicts"] = overlap_conflicts
    report["new_unique_timestamps_to_merge"] = len(rows) - overlap_checks
    if counts["conflicting_duplicates"] or any(report[key] for key in (
            "invalid_timestamps", "out_of_range_rows", "out_of_session_rows", "misaligned_timestamps",
            "invalid_ohlc_rows", "invalid_volume_rows", "existing_timeline_overlap_conflicts")):
        report["validation_status"] = "FAIL"
        raise RuntimeError("Backfill failed one or more validation checks")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows, columns=["timestamp_utc", "open", "high", "low", "close", "volume", "trade_count"]).to_csv(STAGING_CSV, index=False)
    pd.DataFrame(missing, columns=["trading_date_et", "missing_timestamp_utc", "missing_timestamp_et"]).to_csv(GAPS_CSV, index=False)
    file_hash = _sha256(STAGING_CSV)
    report["staging_csv_sha256"] = file_hash
    report["staging_csv_path"] = str(STAGING_CSV.relative_to(ROOT))
    manifest = {
        "validation_status": "PASS", "source": "Alpaca historical stock bars API", "symbol": SYMBOL,
        "feed": FEED, "timeframe": TIMEFRAME, "endpoint": ENDPOINT,
        "requested_date_range_et_inclusive": [START_DATE_ET.isoformat(), END_DATE_ET.isoformat()],
        "requested_timestamp_range_utc_inclusive": [START_UTC, END_UTC],
        "retrieved_unique_candles": len(rows), "staging_csv_sha256": file_hash,
        "staging_csv": STAGING_CSV.name, "http_statuses": statuses,
        "credential_values_saved": False, "request_headers_saved": False,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    try:
        result = run_backfill()
    except Exception as exc:
        # Only this explicit class of messages is constructed from a numeric
        # HTTP status; never print arbitrary request or parsing exceptions.
        safe_message = str(exc)
        if safe_message.startswith(("Alpaca historical request failed with HTTP ",
                                   "Alpaca historical request stopped after HTTP ")):
            print(f"BACKFILL_FAILED: {safe_message}")
        elif safe_message in {
            "Alpaca historical request failed after bounded retries",
            "Alpaca historical response was not valid JSON",
            "Alpaca historical request returned no response",
            "Alpaca IEX returned no ECL five-minute bars for the requested interval",
            "No validated regular-session ECL bars remain",
            "Backfill has conflicting OHLCV values at existing timeline timestamps",
            "Backfill failed one or more validation checks",
        }:
            print(f"BACKFILL_FAILED: {safe_message}")
        else:
            print(f"BACKFILL_FAILED: {type(exc).__name__}")
        raise SystemExit(1)
    print(json.dumps({key: result[key] for key in (
        "validation_status", "requested_date_range_et_inclusive", "actual_returned_date_range_et",
        "unique_valid_candles", "trading_dates_covered", "missing_expected_iex_buckets",
        "duplicate_rows", "existing_timeline_overlap_timestamps", "staging_csv_path")}, indent=2))
