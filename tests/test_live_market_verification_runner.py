from __future__ import annotations
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "live_market_verification.py"
spec = importlib.util.spec_from_file_location("live_market_verification", PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_only_loopback_http_verification_urls_are_accepted():
    assert runner.ensure_loopback_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    for url in ("https://127.0.0.1:8000", "http://localhost:8000", "http://0.0.0.0:8000",
                "http://192.168.1.2:8000", "http://127.0.0.1:9000"):
        with pytest.raises(ValueError):
            runner.ensure_loopback_url(url)


def test_capture_refuses_closed_market_and_allows_regular_session():
    saturday = datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)
    allowed, reason, _ = runner.market_window(saturday)
    assert not allowed and "closed" in reason.lower()
    open_session = datetime(2026, 11, 13, 15, 0, tzinfo=timezone.utc)  # 10:00 New York
    allowed, reason, close = runner.market_window(open_session)
    assert allowed and "open" in reason.lower() and close is not None


def test_capture_refuses_when_too_little_session_time_remains():
    near_close = datetime(2026, 11, 13, 20, 40, tzinfo=timezone.utc)  # 15:40 New York
    allowed, reason, _ = runner.market_window(near_close)
    assert not allowed and "insufficient" in reason.lower()


def test_protocol_success_reporting_uses_acknowledgements_not_final_transport_state():
    snapshots = [
        {"connected": True, "authenticated": False, "subscribed": False,
         "authentication_success_count": 0, "subscription_success_count": 0},
        {"connected": True, "authenticated": True, "subscribed": True,
         "authentication_success_observed": True, "subscription_success_observed": True,
         "authentication_success_count": 1, "subscription_success_count": 1},
        {"connected": False, "authenticated": False, "subscribed": False,
         "authentication_success_observed": True, "subscription_success_observed": True,
         "authentication_success_count": 1, "subscription_success_count": 1},
    ]
    assert runner.protocol_success_observed(snapshots) == {
        "authentication_success_observed": True,
        "subscription_success_observed": True,
    }


def test_transport_connection_alone_is_not_authentication_or_subscription_success():
    assert runner.protocol_success_observed([{
        "connected": True, "authenticated": False, "subscribed": False,
        "authentication_success_count": 0, "subscription_success_count": 0,
    }]) == {"authentication_success_observed": False,
            "subscription_success_observed": False}


def test_read_only_capture_counts_use_alpaca_iex_rows(tmp_path):
    import sqlite3
    db_path = tmp_path / "verify.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.executescript("""
            CREATE TABLE minute_bars(event_time_utc TEXT,received_at_utc TEXT,source TEXT,feed TEXT);
            CREATE TABLE five_minute_candles(candle_time_utc TEXT,finalized_at_utc TEXT,source TEXT,feed TEXT);
            INSERT INTO minute_bars VALUES('t1','2026-11-13T15:00:00+00:00','Alpaca','iex');
            INSERT INTO minute_bars VALUES('t2','2026-11-13T15:01:00+00:00','Alpaca','iex');
            INSERT INTO minute_bars VALUES('t3','2026-11-13T15:01:00+00:00','Other','iex');
            INSERT INTO five_minute_candles VALUES('c1','2026-11-13T15:05:00+00:00','Alpaca','iex');
        """)
    before = db_path.stat().st_mtime_ns
    result = runner._read_only_capture_counts(db_path, datetime(2026, 11, 13, 14, 59, tzinfo=timezone.utc))
    assert result == {"minute_bars_received": 2, "unique_minute_timestamps": 2,
                      "finalized_five_minute_candles": 1, "unique_five_minute_timestamps": 1}
    assert db_path.stat().st_mtime_ns == before
