from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from scripts import live_research_model_monitor
from scripts.live_research_model_monitor import _is_recent_current_session_candle


NY = ZoneInfo("America/New_York")


def test_monitor_accepts_only_recent_candles_from_today_ny_session():
    now = datetime(2026, 11, 16, 12, 20, tzinfo=NY)
    current = (now - timedelta(minutes=10)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    previous = (now - timedelta(days=1)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    stale = (now - timedelta(minutes=16)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    future = (now + timedelta(minutes=1)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    assert _is_recent_current_session_candle(current, now=now)
    assert not _is_recent_current_session_candle(previous, now=now)
    assert not _is_recent_current_session_candle(stale, now=now)
    assert not _is_recent_current_session_candle(future, now=now)
    assert not _is_recent_current_session_candle("2026-11-16T17:10:00", now=now)


def test_legacy_monitor_does_not_duplicate_application_recording(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(live_research_model_monitor, "_get_json", lambda *_args, **_kwargs: {
        "research_preview_recording_enabled": True,
    })
    log = tmp_path / "should-not-be-created.jsonl"
    live_research_model_monitor.run(base="http://127.0.0.1:8000", log_path=log)
    assert not log.exists()
    assert "recording_managed_by_application" in capsys.readouterr().out
