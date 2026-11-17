"""Fetch the fair 5-minute IEX comparator over only train/validation dates."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FETCH_SCRIPT = OUT / "fetch_iex_1min.py"
spec = importlib.util.spec_from_file_location("safe_alpaca_fetch_helpers", FETCH_SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def main():
    from config import get_alpaca_settings
    lower, upper, split_ranges = fetch.allowed_bounds()
    settings = get_alpaca_settings()
    if not settings.credentials_configured:
        raise RuntimeError("Alpaca credentials are not configured")
    first = pd.Timestamp(lower, tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
    end = pd.Timestamp(upper, tz="America/New_York") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    params = {"symbols": "ECL", "timeframe": "5Min", "feed": "iex",
              "start": first.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
              "end": end.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
              "limit": 10000, "sort": "asc", "adjustment": "raw"}
    headers = {"APCA-API-KEY-ID": settings.api_key_id, "APCA-API-SECRET-KEY": settings.api_secret_key}
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "endpoint": fetch.ENDPOINT, "symbol": "ECL", "feed": "iex", "timeframe": "5Min",
                "permitted_new_york_date_range": [lower, upper], "permitted_split_date_ranges": split_ranges,
                "credential_values_saved": False, "request_headers_saved": False,
                "page_statuses": [], "rate_limit_headers": []}
    records, token = [], None
    for page in range(1, fetch.MAX_PAGES + 1):
        page_params = dict(params)
        if token:
            page_params["page_token"] = token
        body, status, rate = fetch._request_page(page_params, headers)
        batch = body["bars"].get("ECL", [])
        if not isinstance(batch, list):
            raise RuntimeError("Alpaca returned an invalid ECL 5-minute bars collection")
        records.extend(batch)
        metadata["page_statuses"].append({"page": page, "http_status": status, "bars_returned": len(batch)})
        metadata["rate_limit_headers"].append({"page": page, **rate})
        print(f"5min_page={page} http_status={status} bars={len(batch)}")
        token = body.get("next_page_token")
        if not token:
            break
        import time
        time.sleep(.35)
    else:
        raise RuntimeError("Alpaca 5-minute pagination exceeded the safety page limit")
    if not records:
        metadata.update({"request_success": True, "bars_returned": 0,
                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        (OUT / "fetch_5min_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print("5min_fetch_success=true bars=0")
        return
    columns = {"t": "timestamp_utc", "o": "open", "h": "high", "l": "low",
               "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap"}
    data = pd.DataFrame([{new: row.get(old) for old, new in columns.items()} for row in records])
    stamp = pd.to_datetime(data.timestamp_utc, utc=True, errors="raise")
    nydate = stamp.dt.tz_convert("America/New_York").dt.date
    if ((nydate < pd.Timestamp(lower).date()) | (nydate > pd.Timestamp(upper).date())).any():
        raise RuntimeError("Alpaca 5-minute response crossed the permitted date boundary")
    data["timestamp_utc"] = stamp.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    data = data.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    path = OUT / "raw_ecl_iex_5min_train_validation.csv"
    data.to_csv(path, index=False)
    metadata.update({"request_success": True, "bars_returned": int(len(data)),
                     "first_timestamp_utc": data.timestamp_utc.iloc[0],
                     "last_timestamp_utc": data.timestamp_utc.iloc[-1],
                     "raw_csv": path.name, "raw_csv_sha256": fetch.sha256(path),
                     "raw_csv_bytes": path.stat().st_size,
                     "finished_utc": datetime.now(timezone.utc).isoformat()})
    (OUT / "fetch_5min_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"5min_fetch_success=true bars={len(data)}")
    print(f"5min_raw_data={path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"5min_fetch_failed=unexpected_{type(exc).__name__}")
        raise SystemExit(2)
