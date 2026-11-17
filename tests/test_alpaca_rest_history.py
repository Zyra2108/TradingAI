from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from config import AlpacaSettings
from live_app.alpaca_rest_history import AlpacaCurrentBarsRefresher, load_verified_current_cache

UTC = timezone.utc


class FakeResponse:
    status_code = 200
    headers = {}
    def __init__(self, payload): self.payload = payload
    def json(self): return self.payload


class FakeSession:
    def __init__(self, pages): self.pages = list(pages); self.calls = []
    def get(self, url, *, headers, params, timeout):
        self.calls.append({"url": url, "headers": headers, "params": dict(params), "timeout": timeout})
        return FakeResponse(self.pages.pop(0))


def _bar(timestamp, *, close=100.5):
    return {"t": timestamp, "o": 100.0, "h": 101.0, "l": 99.0,
            "c": close, "v": 50, "n": 5}


def _settings(configured=True):
    return AlpacaSettings(api_key_id="private-id" if configured else None,
        api_secret_key="private-secret" if configured else None, display_authorized=True)


def test_rest_refresh_fetches_only_finalized_iex_five_minute_bars_and_keeps_cache_separate(tmp_path):
    session = FakeSession([{"bars": {"ECL": [_bar("2026-11-16T15:20:00Z")]}, "next_page_token": None}])
    csv_path, manifest_path = tmp_path / "current.csv", tmp_path / "manifest.json"
    refresher = AlpacaCurrentBarsRefresher(_settings(), session=session, csv_path=csv_path,
        manifest_path=manifest_path, conflict_path=tmp_path / "conflicts.csv", sleep=lambda _: None)
    result = refresher.refresh([], now=datetime(2026, 11, 16, 15, 30, tzinfo=UTC))

    assert result["state"] == "current"
    assert result["bars_added"] == 1
    assert result["latest_bar_utc"] == "2026-11-16T15:20:00Z"
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://data.alpaca.markets/v2/stocks/bars"
    assert call["params"]["symbols"] == "ECL"
    assert call["params"]["feed"] == "iex"
    assert call["params"]["timeframe"] == "5Min"
    assert "private-id" not in json.dumps(result)
    assert "private-secret" not in json.dumps(result)
    loaded = load_verified_current_cache(csv_path, manifest_path)
    assert len(loaded) == 1
    assert loaded[0]["candle_time_utc"] == "2026-11-16T15:20:00Z"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["feed"] == "IEX" and manifest["row_count"] == 1


def test_missing_iex_five_minute_buckets_are_not_synthesized(tmp_path):
    session = FakeSession([{"bars": {"ECL": [_bar("2026-11-16T15:20:00Z")]}, "next_page_token": None}])
    refresher = AlpacaCurrentBarsRefresher(_settings(), session=session,
        csv_path=tmp_path / "current.csv", manifest_path=tmp_path / "manifest.json",
        conflict_path=tmp_path / "conflicts.csv", sleep=lambda _: None)
    existing = [{"time": int(datetime(2026, 11, 16, 15, 10, tzinfo=UTC).timestamp())}]
    result = refresher.refresh(existing, now=datetime(2026, 11, 16, 15, 30, tzinfo=UTC))
    assert result["bars_added"] == 1
    assert len(load_verified_current_cache(tmp_path / "current.csv", tmp_path / "manifest.json")) == 1


def test_refresh_rejects_invalid_session_or_unfinalized_bars_without_writing(tmp_path):
    session = FakeSession([{"bars": {"ECL": [_bar("2026-11-16T14:25:00Z")]}, "next_page_token": None}])
    csv_path, manifest_path = tmp_path / "current.csv", tmp_path / "manifest.json"
    refresher = AlpacaCurrentBarsRefresher(_settings(), session=session, csv_path=csv_path,
        manifest_path=manifest_path, conflict_path=tmp_path / "conflicts.csv", sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="out-of-session"):
        refresher._validate_rows([_bar("2026-11-16T14:25:00Z")],
            datetime(2026, 11, 16, 14, 25, tzinfo=UTC),
            datetime(2026, 11, 16, 15, 20, tzinfo=UTC),
            datetime(2026, 11, 16, 15, 30, tzinfo=UTC))
    assert not csv_path.exists() and not manifest_path.exists()


def test_current_cache_conflict_does_not_overwrite_first_observation(tmp_path):
    session = FakeSession([{"bars": {"ECL": [_bar("2026-11-16T15:20:00Z", close=101.0)]}, "next_page_token": None}])
    csv_path, manifest_path = tmp_path / "current.csv", tmp_path / "manifest.json"
    refresher = AlpacaCurrentBarsRefresher(_settings(), session=session, csv_path=csv_path,
        manifest_path=manifest_path, conflict_path=tmp_path / "conflicts.csv", sleep=lambda _: None)
    original = {"time": int(datetime(2026, 11, 16, 15, 20, tzinfo=UTC).timestamp()),
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 50}
    with pytest.raises(RuntimeError, match="conflicting existing candle"):
        refresher.refresh([original], now=datetime(2026, 11, 16, 15, 30, tzinfo=UTC))
    assert not csv_path.exists()
    assert (tmp_path / "conflicts.csv").exists()


def test_missing_credentials_fail_without_request(tmp_path):
    session = FakeSession([])
    refresher = AlpacaCurrentBarsRefresher(_settings(False), session=session, csv_path=tmp_path / "current.csv",
        manifest_path=tmp_path / "manifest.json", conflict_path=tmp_path / "conflicts.csv")
    with pytest.raises(RuntimeError, match="credentials are not configured"):
        refresher.refresh([], now=datetime(2026, 11, 16, 15, 30, tzinfo=UTC))
    assert session.calls == []


def test_paginated_refresh_appends_rows_in_timestamp_order(tmp_path):
    session = FakeSession([
        {"bars": {"ECL": [_bar("2026-11-16T15:20:00Z")]}, "next_page_token": "page-2"},
        {"bars": {"ECL": [_bar("2026-11-16T15:25:00Z", close=100.8)]}, "next_page_token": None},
    ])
    refresher = AlpacaCurrentBarsRefresher(_settings(), session=session, csv_path=tmp_path / "current.csv",
        manifest_path=tmp_path / "manifest.json", conflict_path=tmp_path / "conflicts.csv", sleep=lambda _: None)
    result = refresher.refresh([], now=datetime(2026, 11, 16, 15, 35, tzinfo=UTC))
    assert result["pages"] == 2 and result["bars_added"] == 2
    rows = load_verified_current_cache(tmp_path / "current.csv", tmp_path / "manifest.json")
    assert [row["candle_time_utc"] for row in rows] == ["2026-11-16T15:20:00Z", "2026-11-16T15:25:00Z"]
