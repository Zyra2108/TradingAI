from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
import hashlib
import json
import pandas as pd

from live_app.unified_market import LAST_ALLOWED_DATE, UnifiedHistoricalMarket
import live_app.unified_market as unified_module


def test_unified_history_merges_only_provenance_verified_historical_sources(tmp_path, monkeypatch):
    # Keep this historical-source contract test independent of a user's
    # separately maintained current-market cache.
    monkeypatch.setattr(unified_module, "CURRENT_CACHE_CSV", tmp_path / "no-current-cache.csv")
    monkeypatch.setattr(unified_module, "CURRENT_CACHE_MANIFEST", tmp_path / "no-current-cache.json")
    import live_app.alpaca_rest_history as rest_history
    monkeypatch.setattr(rest_history, "CURRENT_CACHE_CSV", tmp_path / "no-current-cache.csv")
    monkeypatch.setattr(rest_history, "CURRENT_CACHE_MANIFEST", tmp_path / "no-current-cache.json")
    monkeypatch.setattr(unified_module, "load_verified_current_cache", lambda: [])
    market = UnifiedHistoricalMarket()
    candles = market.candles()
    assert len(candles) > 10000
    assert candles == sorted(candles, key=lambda row: row["time"])
    assert len({row["time"] for row in candles}) == len(candles)
    assert candles[0]["candle_time_utc"].startswith("2025-11-13T14:30:00")
    assert candles[-1]["candle_time_utc"].startswith("2026-11-12T20:55:00")
    assert all(row["volume"] is not None and row["volume"] >= 0 for row in candles)
    assert all(row["source"] == "historical" and row["feed"] == "IEX" for row in candles)
    assert market.coverage["last_allowed_date_et"] == "2026-11-12"
    assert market.coverage["backfill_manifest_status"] == "PASS"
    assert market.coverage["backfill_candle_count"] == 2682
    assert market.coverage["backfill_overlap_timestamps"] == 0
    assert market.coverage["candle_count"] == 16059 + 2682
    assert market.coverage["note"].endswith("not all-time ECL history.")
    last_local_date = datetime.fromisoformat(candles[-1]["candle_time_utc"].replace("Z", "+00:00")).astimezone(
        ZoneInfo("America/New_York")
    ).date()
    assert last_local_date.isoformat() == "2026-11-12"


def test_unified_history_refuses_any_alternate_source_path(tmp_path):
    alternate = tmp_path / "not-approved.csv"
    alternate.write_text("timestamp_utc,open,high,low,close,volume\n", encoding="utf-8")
    metadata = tmp_path / "fetch_metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="only read the approved"):
        UnifiedHistoricalMarket(alternate, metadata)


def write_backfill(tmp_path, monkeypatch, row):
    csv_path = tmp_path / "staged.csv"
    manifest_path = tmp_path / "manifest.json"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps({
        "validation_status": "PASS", "symbol": "ECL", "feed": "iex", "timeframe": "5Min",
        "requested_date_range_et_inclusive": ["2026-09-25", "2026-11-12"],
        "staging_csv": csv_path.name, "staging_csv_sha256": digest, "retrieved_unique_candles": 1,
    }), encoding="utf-8")
    monkeypatch.setattr(unified_module, "BACKFILL_CSV", csv_path)
    monkeypatch.setattr(unified_module, "BACKFILL_MANIFEST", manifest_path)
    return csv_path, manifest_path


def test_verified_backfill_loader_merges_without_overwriting_and_rejects_conflicts(tmp_path, monkeypatch):
    row = {"timestamp_utc": "2026-09-25T13:30:00Z", "open": 300, "high": 301, "low": 299,
           "close": 300.5, "volume": 400, "trade_count": 7}
    write_backfill(tmp_path, monkeypatch, row)
    epoch = int(pd.Timestamp(row["timestamp_utc"]).timestamp())
    same = {"time": epoch, "open": 300, "high": 301, "low": 299, "close": 300.5, "volume": 400}
    loaded, overlap = UnifiedHistoricalMarket._load_verified_backfill([same])
    assert overlap == 1
    assert loaded[0]["feed"] == "IEX" and loaded[0]["historical_source"] == "verified_5min_backfill"
    conflict = {**same, "close": 300.75}
    with pytest.raises(ValueError, match="conflicting OHLCV"):
        UnifiedHistoricalMarket._load_verified_backfill([conflict])


def test_verified_backfill_loader_rejects_hash_mismatch_and_out_of_session_rows(tmp_path, monkeypatch):
    row = {"timestamp_utc": "2026-09-25T13:25:00Z", "open": 300, "high": 301, "low": 299,
           "close": 300.5, "volume": 400}
    csv_path, _ = write_backfill(tmp_path, monkeypatch, row)
    # Alter after manifest creation to prove the loader detects changed staging bytes.
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check"):
        UnifiedHistoricalMarket._load_verified_backfill([])
    # Rebuild a passing manifest around a real but out-of-session 09:25 ET bar.
    write_backfill(tmp_path, monkeypatch, row)
    with pytest.raises(ValueError, match="out-of-session"):
        UnifiedHistoricalMarket._load_verified_backfill([])
    row["timestamp_utc"] = "2026-09-25T13:30:00Z"
    write_backfill(tmp_path, monkeypatch, row)
    loaded, _ = UnifiedHistoricalMarket._load_verified_backfill([])
    assert len(loaded) == 1
