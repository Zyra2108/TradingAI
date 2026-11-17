from __future__ import annotations

from scripts.backfill_ecl_iex_timeline import _parse_api_bars, _quality_report, fetch_pages


class FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, *, headers, params, timeout):
        self.calls.append((url, dict(params), timeout))
        if len(self.calls) == 1:
            return FakeResponse({"bars": {"ECL": [{"t": "2026-09-25T13:30:00Z", "o": 300,
                "h": 301, "l": 299, "c": 300.5, "v": 5, "n": 2}]}, "next_page_token": "opaque-token"})
        return FakeResponse({"bars": {"ECL": [{"t": "2026-09-25T13:35:00Z", "o": 300.5,
            "h": 302, "l": 300, "c": 301, "v": 7, "n": 3}]}, "next_page_token": None})


def test_rest_fetch_uses_iex_5min_pagination_and_bounded_request_shape():
    session = FakeSession()
    settings = type("Settings", (), {"credentials_configured": True, "api_key_id": "hidden", "api_secret_key": "hidden"})()
    rows, statuses, retries = fetch_pages(session, settings, sleep=lambda _: None)
    assert len(rows) == 2 and statuses == [200, 200] and retries == 0
    first = session.calls[0][1]
    second = session.calls[1][1]
    assert first["symbols"] == "ECL" and first["feed"] == "iex" and first["timeframe"] == "5Min"
    assert first["start"] == "2026-09-25T04:00:00Z" and first["end"] == "2026-11-13T04:59:59Z"
    assert "page_token" not in first and second["page_token"] == "opaque-token"
    assert "hidden" not in repr(session.calls)


def test_native_bars_validate_and_missing_iex_buckets_are_reported_not_filled():
    raw = [{"t": "2026-09-25T13:30:00Z", "o": 300, "h": 301, "l": 299, "c": 300.5, "v": 5, "n": 2}]
    rows, counts = _parse_api_bars(raw)
    report, gaps = _quality_report(rows, len(raw), counts, [200], 0)
    assert report["validation_status"] == "PASS"
    assert report["unique_valid_candles"] == 1
    assert report["missing_expected_iex_buckets"] == len(gaps) > 0
    assert len(gaps) == report["expected_regular_session_buckets"] - 1
    assert rows[0]["timestamp_utc"] == "2026-09-25T13:30:00Z"


def test_bad_ohlc_and_extended_hours_rows_are_rejected():
    bad_ohlc = [{"t": "2026-09-25T13:30:00Z", "o": 300, "h": 299, "l": 301, "c": 300.5, "v": 5, "n": 2}]
    try:
        _parse_api_bars(bad_ohlc)
    except ValueError as exc:
        assert "validation" in str(exc)
    else:
        raise AssertionError("invalid OHLC was accepted")
    extended = [{"t": "2026-09-25T13:25:00Z", "o": 300, "h": 301, "l": 299, "c": 300.5, "v": 5, "n": 2}]
    try:
        _parse_api_bars(extended)
    except ValueError as exc:
        assert "validation" in str(exc)
    else:
        raise AssertionError("out-of-session candle was accepted")
