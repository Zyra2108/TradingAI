"""Fetch only the established Alpaca IEX train/validation date window.

This script reads timestamps from train.csv and validation.csv only to derive
the permitted date limits. It never opens the test split or any broader source.
Credentials are obtained through config.py and are never printed or persisted.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "alpaca_rebuild_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
MAX_PAGES = 40
MAX_RETRIES = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def allowed_bounds() -> tuple[str, str, dict[str, str]]:
    """Read only date metadata from the permitted train and validation files."""
    train_dt = pd.to_datetime(pd.read_csv(DATA / "train.csv", usecols=["datetime"])["datetime"], errors="raise")
    validation_dt = pd.to_datetime(pd.read_csv(DATA / "validation.csv", usecols=["datetime"])["datetime"], errors="raise")
    first = min(train_dt.min().date(), validation_dt.min().date())
    last = max(train_dt.max().date(), validation_dt.max().date())
    if first > last:
        raise ValueError("Training and validation boundaries are inconsistent")
    return first.isoformat(), last.isoformat(), {
        "train_start": train_dt.min().date().isoformat(),
        "train_end": train_dt.max().date().isoformat(),
        "validation_start": validation_dt.min().date().isoformat(),
        "validation_end": validation_dt.max().date().isoformat(),
    }


def _request_page(params: dict, headers: dict) -> tuple[dict, int, dict]:
    for attempt in range(MAX_RETRIES):
        try:
            request = Request(f"{ENDPOINT}?{urlencode(params)}", headers=headers, method="GET")
            with urlopen(request, timeout=60) as response:
                status_code = int(response.status)
                response_headers = response.headers
                response_body = response.read()
        except HTTPError as response:
            status_code = int(response.code)
            response_headers = response.headers
            response_body = b""
        except URLError:
            if attempt + 1 == MAX_RETRIES:
                raise RuntimeError("Alpaca request failed at the network/transport layer") from None
            time.sleep(min(2 ** attempt, 30))
            continue
        if status_code == 429 or status_code >= 500:
            if attempt + 1 == MAX_RETRIES:
                raise RuntimeError(f"Alpaca request failed with HTTP {status_code}")
            retry_after = response_headers.get("Retry-After", "")
            try:
                delay = min(max(float(retry_after), 1.0), 60.0)
            except ValueError:
                delay = min(2 ** attempt, 30)
            time.sleep(delay)
            continue
        if status_code in (401, 403):
            raise RuntimeError(f"Alpaca rejected historical-data access (HTTP {status_code})")
        if status_code != 200:
            raise RuntimeError(f"Alpaca historical-data request failed (HTTP {status_code})")
        try:
            body = json.loads(response_body)
        except (ValueError, UnicodeDecodeError):
            raise RuntimeError("Alpaca returned a non-JSON response") from None
        if not isinstance(body, dict) or not isinstance(body.get("bars"), dict):
            raise RuntimeError("Alpaca response did not contain the expected bars object")
        rate = {key: response_headers.get(key) for key in
                ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset")
                if response_headers.get(key) is not None}
        return body, status_code, rate
    raise RuntimeError("Alpaca request retry limit reached")


def main() -> int:
    from config import get_alpaca_settings

    lower_date, upper_date, split_ranges = allowed_bounds()
    settings = get_alpaca_settings()
    if not settings.credentials_configured:
        raise RuntimeError("Alpaca credentials are not configured")
    # Bounds are explicit UTC instants covering only the permitted New York
    # dates. End immediately before midnight following the final permitted day.
    first_day = pd.Timestamp(lower_date, tz="America/New_York")
    start_utc = (first_day + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
    end_utc = (pd.Timestamp(upper_date, tz="America/New_York") + pd.Timedelta(hours=23, minutes=59, seconds=59)).tz_convert("UTC")
    start_text, end_text = start_utc.isoformat().replace("+00:00", "Z"), end_utc.isoformat().replace("+00:00", "Z")
    headers = {"APCA-API-KEY-ID": settings.api_key_id,
               "APCA-API-SECRET-KEY": settings.api_secret_key}
    params = {"symbols": "ECL", "timeframe": "1Min", "feed": "iex",
              "start": start_text, "end": end_text, "limit": 10000,
              "sort": "asc", "adjustment": "raw"}
    run = {"started_utc": datetime.now(timezone.utc).isoformat(),
           "endpoint": ENDPOINT, "symbol": "ECL", "feed": "iex", "timeframe": "1Min",
           "start_utc": start_text, "end_inclusive_utc": end_text,
           "permitted_new_york_date_range": [lower_date, upper_date],
           "permitted_split_date_ranges": split_ranges,
           "pagination_limit": 10000, "max_pages": MAX_PAGES,
           "credential_values_saved": False, "request_headers_saved": False,
           "page_statuses": [], "rate_limit_headers": [], "request_success": False}
    records = []
    token = None
    try:
        for page in range(1, MAX_PAGES + 1):
            page_params = dict(params)
            if token:
                page_params["page_token"] = token
            body, status, rate = _request_page(page_params, headers)
            bars = body["bars"].get("ECL", [])
            if not isinstance(bars, list):
                raise RuntimeError("Alpaca returned an invalid ECL bars collection")
            records.extend(bars)
            run["page_statuses"].append({"page": page, "http_status": status,
                                          "bars_returned": len(bars)})
            run["rate_limit_headers"].append({"page": page, **rate})
            print(f"page={page} http_status={status} bars={len(bars)}")
            token = body.get("next_page_token")
            if not token:
                break
            time.sleep(0.35)
        else:
            raise RuntimeError("Alpaca pagination exceeded the safety page limit")
        run["request_success"] = True
    except Exception as exc:
        # Never print exception strings from HTTP libraries; they can include
        # request details. Only our own fixed safe descriptions are surfaced.
        run["safe_error"] = str(exc) if isinstance(exc, RuntimeError) else "unexpected_fetch_failure"
        run["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (OUT / "fetch_metadata.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        print(f"fetch_failed={run['safe_error']}")
        return 2

    fields = {"t": "timestamp_utc", "o": "open", "h": "high", "l": "low",
              "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap"}
    frame = pd.DataFrame([{new: bar.get(old) for old, new in fields.items()} for bar in records],
                         columns=list(fields.values()))
    if frame.empty:
        run["finished_utc"] = datetime.now(timezone.utc).isoformat()
        run["returned_bars"] = 0
        (OUT / "fetch_metadata.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        print("fetch_success=true bars=0; no usable historical data; stopping")
        return 3
    parsed = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    local_days = parsed.dt.tz_convert("America/New_York").dt.date
    if (local_days < pd.Timestamp(lower_date).date()).any() or (local_days > pd.Timestamp(upper_date).date()).any():
        raise RuntimeError("Provider response included a timestamp outside the permitted date window")
    frame["timestamp_utc"] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    frame = frame.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    raw_path = OUT / "raw_ecl_iex_1min_train_validation.csv"
    frame.to_csv(raw_path, index=False)
    run.update({"finished_utc": datetime.now(timezone.utc).isoformat(),
                "returned_bars": int(len(frame)),
                "first_timestamp_utc": frame.timestamp_utc.iloc[0],
                "last_timestamp_utc": frame.timestamp_utc.iloc[-1],
                "raw_csv": raw_path.name, "raw_csv_sha256": sha256(raw_path),
                "raw_csv_bytes": raw_path.stat().st_size})
    (OUT / "fetch_metadata.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(f"fetch_success=true bars={len(frame)}")
    print(f"raw_data={raw_path}")
    print(f"fetch_metadata={OUT / 'fetch_metadata.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        # Safe validation errors only. Do not stringify arbitrary exceptions.
        safe_type = type(exc).__name__
        print(f"fetch_failed=unexpected_{safe_type}")
        sys.exit(2)
