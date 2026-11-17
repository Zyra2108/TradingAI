from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from live_app.alpaca_aggregation import AlpacaFiveMinuteAggregator
from live_app.alpaca_storage import AlpacaBarStore
from live_app.alpaca_provider import MinuteBarEvent
from live_app.calendar import session_for_date

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
REGULAR_DAY = date(2026, 11, 12)


def utc_open(day=REGULAR_DAY):
    session = session_for_date(day)
    assert session is not None
    return session[0].astimezone(UTC)


def make_event(stamp: datetime, index: int, *, received: datetime | None = None, **overrides):
    values = dict(
        symbol="ECL", original_event_timestamp=stamp.isoformat().replace("+00:00", "Z"),
        event_time_utc=stamp, received_at_utc=received or stamp + timedelta(minutes=1),
        open=100.0 + index, high=102.0 + index, low=99.0 + index,
        close=101.0 + index, volume=10.0 + index, trade_count=2 + index,
    )
    values.update(overrides)
    return MinuteBarEvent(**values)


def setup(tmp_path, *, day=REGULAR_DAY, buffer=120):
    store = AlpacaBarStore(tmp_path / "alpaca.sqlite3")
    start = utc_open(day)
    now = [start]
    aggregator = AlpacaFiveMinuteAggregator(store, finalization_buffer_seconds=buffer, clock=lambda: now[0])
    return store, aggregator, start, now


def send_bucket(aggregator, start, *, order=(0, 1, 2, 3, 4), missing=(), now=None):
    result = None
    for i in order:
        if i not in missing:
            event = make_event(start + i * timedelta(minutes=1), i)
            result = aggregator.ingest(event, now=event.received_at_utc)
    if now is not None:
        return aggregator.finalize_bucket(start, now=now)
    return result


def test_correct_ohlcv_aggregation_and_named_schema(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    now = start + timedelta(minutes=7)
    result = send_bucket(agg, start, now=now)
    assert result.status == "finalized"
    assert result.candle.candle_time_utc == start
    assert (result.candle.open, result.candle.high, result.candle.low, result.candle.close) == (100, 106, 99, 105)
    assert result.candle.volume == 60
    assert result.candle.trade_count == 20
    assert result.candle.feed == "iex"
    with store.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 4
        assert db.execute("SELECT COUNT(*) FROM minute_bars").fetchone()[0] == 5
        assert db.execute("SELECT COUNT(*) FROM five_minute_candles").fetchone()[0] == 1


@pytest.mark.parametrize("missing_index", range(5))
def test_each_missing_minute_is_recorded_without_fabrication(tmp_path, missing_index):
    store, agg, start, _ = setup(tmp_path)
    now = start + timedelta(minutes=7)
    result = send_bucket(agg, start, missing=(missing_index,), now=now)
    assert result.status == "finalized"
    assert result.missing_timestamps_utc == (start + missing_index * timedelta(minutes=1),)
    assert store.five_minute_count() == 1
    import json
    metadata = json.loads(store.five_minute_candle_at(start)["aggregation_metadata_json"])
    assert metadata["observed_minute_count"] == 4
    assert len(metadata["missing_minute_timestamps_utc"]) == 1
    with store.connect() as db:
        rows = db.execute("SELECT missing_timestamps_json FROM aggregation_gaps WHERE bucket_start_utc=?",
                          (start.isoformat(timespec="microseconds").replace("+00:00", "Z"),)).fetchall()
        # Gap diagnostics are append-only snapshots as late bars arrive; the most
        # recent snapshot must still identify the unfilled expected minute.
        assert rows
        assert str(result.missing_timestamps_utc[0].isoformat(timespec="microseconds").replace("+00:00", "Z")) in rows[-1][0]


def test_identical_duplicate_is_idempotent_and_revised_message_is_conflict(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    first = make_event(start, 0)
    assert store.insert_minute_bar(first) == (True, False)
    replayed = replace(first, received_at_utc=first.received_at_utc + timedelta(seconds=9))
    assert store.insert_minute_bar(replayed) == (False, False)
    revised = replace(first, high=first.high + 3, close=first.close + 3,
                      received_at_utc=first.received_at_utc + timedelta(seconds=5))
    assert store.insert_minute_bar(revised) == (False, True)
    with store.connect() as db:
        row = db.execute("SELECT close FROM minute_bars WHERE event_time_utc=?", (start.isoformat(timespec="microseconds").replace("+00:00", "Z"),)).fetchone()
        assert row[0] == first.close
        assert db.execute("SELECT COUNT(*) FROM minute_bar_conflicts").fetchone()[0] == 1


def test_out_of_order_minutes_aggregate_correctly(tmp_path):
    _, agg, start, _ = setup(tmp_path)
    result = send_bucket(agg, start, order=(4, 2, 0, 3, 1), now=start + timedelta(minutes=7))
    assert result.status == "finalized"
    assert result.candle.close == 105


@pytest.mark.parametrize("overrides", [
    {"high": 98.0}, {"low": 103.0}, {"close": float("nan")}, {"open": -1.0},
])
def test_invalid_ohlc_rejected_before_persistence(tmp_path, overrides):
    store, _, start, _ = setup(tmp_path)
    with pytest.raises(ValueError):
        store.insert_minute_bar(make_event(start, 0, **overrides))
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM minute_bars").fetchone()[0] == 0


def test_utc_to_new_york_conversion_and_dst_session_opens():
    before = session_for_date(date(2026, 3, 6))[0]
    after = session_for_date(date(2026, 3, 9))[0]
    assert before.astimezone(UTC).hour == 14
    assert after.astimezone(UTC).hour == 13
    assert before.astimezone(NY).hour == after.astimezone(NY).hour == 9
    assert before.astimezone(NY).minute == after.astimezone(NY).minute == 30


def test_session_open_close_holiday_and_early_close_alignment(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    # First bucket begins exactly at the official 09:30 NYSE open.
    result = send_bucket(agg, start, now=start + timedelta(minutes=7))
    assert result.status == "finalized"
    regular_close = session_for_date(REGULAR_DAY)[1].astimezone(UTC)
    assert agg._bucket_start_for_event(regular_close - timedelta(minutes=1)) == regular_close - timedelta(minutes=5)
    assert agg._bucket_start_for_event(regular_close) is None
    assert agg._bucket_start_for_event(datetime(2026, 11, 26, 14, 30, tzinfo=UTC)) is None

    early_day = date(2026, 11, 27)
    early_session = session_for_date(early_day)
    early_close = early_session[1].astimezone(UTC)
    early_store = AlpacaBarStore(tmp_path / "early.sqlite3")
    early_agg = AlpacaFiveMinuteAggregator(early_store, finalization_buffer_seconds=0)
    assert early_agg._bucket_start_for_event(early_close - timedelta(minutes=1)) == early_close - timedelta(minutes=5)
    assert early_agg._bucket_start_for_event(early_close) is None


def test_finalization_buffer_prevents_partial_or_early_bucket(tmp_path):
    store, agg, start, _ = setup(tmp_path, buffer=120)
    send_bucket(agg, start, now=start + timedelta(minutes=6, seconds=59))
    assert store.five_minute_count() == 0
    result = agg.finalize_bucket(start, now=start + timedelta(minutes=7))
    assert result.status == "finalized"
    assert store.five_minute_count() == 1


def test_periodic_due_bucket_recovery_finalizes_without_another_event_in_bucket(tmp_path):
    store, agg, start, _ = setup(tmp_path, buffer=120)
    # All five real-shaped source minutes arrive, but the last arrives before
    # the two-minute buffer is satisfied. No new event is sent afterward.
    result = send_bucket(agg, start, now=start + timedelta(minutes=6))
    assert result.status == "waiting_for_finalization_buffer"
    assert store.five_minute_count() == 0

    swept = agg.recover_due_buckets(now=start + timedelta(minutes=7))
    matching = [item for item in swept if item.candle is not None and item.candle.candle_time_utc == start]
    assert len(matching) == 1
    assert matching[0].status == "finalized"
    assert store.five_minute_count() == 1


def test_periodic_sweep_finalizes_sparse_bucket_with_missing_metadata(tmp_path):
    store, agg, start, _ = setup(tmp_path, buffer=120)
    send_bucket(agg, start, missing=(2,), now=start + timedelta(minutes=6))
    swept = agg.recover_due_buckets(now=start + timedelta(minutes=10))
    matching = [item for item in swept if item.candle and item.candle.candle_time_utc == start]
    assert matching and matching[0].status == "finalized"
    assert store.five_minute_count() == 1


def test_pending_bucket_survives_reconnect_and_finalizes_idempotently(tmp_path):
    store, agg, start, _ = setup(tmp_path, buffer=120)
    for i in (0, 1):
        agg.ingest(make_event(start + timedelta(minutes=i), i), now=start + timedelta(minutes=3))
    # Simulate a disconnect: the next connection delivers the rest of the
    # persisted bucket. It still arrives before the finalization buffer.
    for i in (2, 3, 4):
        result = agg.ingest(make_event(start + timedelta(minutes=i), i), now=start + timedelta(minutes=6))
    assert result.status == "waiting_for_finalization_buffer"
    assert store.five_minute_count() == 0
    recovered = agg.recover_due_buckets(now=start + timedelta(minutes=7))
    assert any(x.status == "finalized" for x in recovered)
    again = agg.recover_due_buckets(now=start + timedelta(minutes=8))
    assert any(x.status == "already_finalized" for x in again)
    assert store.five_minute_count() == 1


def test_restart_recovery_uses_persisted_minutes_and_is_idempotent(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    for i in (0, 1, 2):
        event = make_event(start + i * timedelta(minutes=1), i)
        agg.ingest(event, now=event.received_at_utc)
    restarted = AlpacaFiveMinuteAggregator(store, finalization_buffer_seconds=120,
                                             clock=lambda: start + timedelta(minutes=7))
    for i in (3, 4):
        store.insert_minute_bar(make_event(start + i * timedelta(minutes=1), i))
    recovered = restarted.recover_due_buckets(now=start + timedelta(minutes=7))
    assert any(item.status == "finalized" for item in recovered)
    assert store.five_minute_count() == 1
    again = restarted.recover_due_buckets(now=start + timedelta(minutes=7))
    assert any(item.status == "already_finalized" for item in again)
    assert store.five_minute_count() == 1


def test_late_missing_minute_allows_safe_retry(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    for i in (0, 1, 3, 4):
        event = make_event(start + i * timedelta(minutes=1), i)
        agg.ingest(event, now=event.received_at_utc)
    assert store.five_minute_count() == 0
    event = make_event(start + 2 * timedelta(minutes=1), 2)
    agg.ingest(event, now=event.received_at_utc)
    result = agg.finalize_bucket(start, now=start + timedelta(minutes=7))
    assert result.status == "finalized"
    assert store.five_minute_count() == 1


def test_duplicate_finalized_bucket_does_not_create_duplicate_candle(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    send_bucket(agg, start, now=start + timedelta(minutes=7))
    result = agg.finalize_bucket(start, now=start + timedelta(minutes=8))
    assert result.status == "already_finalized"
    assert store.five_minute_count() == 1


@pytest.mark.parametrize("observed_count", [1, 2, 3, 4, 5])
def test_sparse_policy_aggregates_each_supported_observed_count(tmp_path, observed_count):
    store, agg, start, _ = setup(tmp_path)
    for i in range(observed_count):
        event = make_event(start + i * timedelta(minutes=1), i)
        agg.ingest(event, now=event.received_at_utc)
    result = agg.finalize_bucket(start, now=start + timedelta(minutes=7))
    assert result.status == "finalized"
    candle = result.candle
    assert candle.open == 100.0
    assert candle.close == 100.0 + observed_count
    assert candle.volume == sum(10.0 + i for i in range(observed_count))
    assert candle.trade_count == sum(2 + i for i in range(observed_count))
    assert candle.aggregation_metadata["observed_minute_count"] == observed_count
    assert len(candle.aggregation_metadata["missing_minute_timestamps_utc"]) == 5 - observed_count


def test_empty_bucket_never_creates_a_candle(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    result = agg.finalize_bucket(start, now=start + timedelta(minutes=7))
    assert result.status == "empty_bucket"
    assert store.five_minute_count() == 0


def test_updated_bar_replaces_minute_before_finalization_without_double_count(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    send_bucket(agg, start)
    revised = make_event(start + timedelta(minutes=4), 4,
                         received=start + timedelta(minutes=6), close=110.0,
                         high=111.0, volume=50.0, trade_count=9, is_updated=True)
    agg.ingest(revised, now=revised.received_at_utc)
    agg.ingest(revised, now=revised.received_at_utc)
    result = agg.finalize_bucket(start, now=start + timedelta(minutes=7))
    assert result.status == "finalized"
    assert result.candle.close == 110.0
    assert result.candle.high == 111.0
    assert result.candle.volume == sum(10.0 + i for i in range(4)) + 50.0
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM minute_bar_updates").fetchone()[0] == 1


def test_updated_bar_after_finalization_is_audited_and_cannot_rewrite_candle(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    send_bucket(agg, start, now=start + timedelta(minutes=7))
    before = store.five_minute_candle_at(start)
    revised = make_event(start + timedelta(minutes=4), 4,
                         received=start + timedelta(minutes=8), close=110.0,
                         high=111.0, volume=50.0, trade_count=9, is_updated=True)
    result = agg.ingest(revised, now=revised.received_at_utc)
    after = store.five_minute_candle_at(start)
    assert result.status == "late_updated_bar_ignored"
    assert before == after
    assert store.late_update_count() == 1


def test_known_disconnect_overlapping_bucket_rejects_even_sparse_data(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    for i in (0, 2):
        event = make_event(start + timedelta(minutes=i), i)
        agg.ingest(event, now=event.received_at_utc)
    store.begin_stream_interruption(start + timedelta(minutes=2), "test_disconnect")
    store.end_stream_interruption(start + timedelta(minutes=3))
    result = agg.finalize_bucket(start, now=start + timedelta(minutes=7))
    assert result.status == "known_stream_interruption"
    assert store.five_minute_count() == 0


def test_processing_failure_invalidates_its_bucket(tmp_path):
    store, agg, start, _ = setup(tmp_path)
    event = make_event(start, 0)
    agg.ingest(event, now=event.received_at_utc)
    agg.mark_processing_failure(start + timedelta(minutes=1), detected_at_utc=start + timedelta(minutes=2))
    result = agg.finalize_bucket(start, now=start + timedelta(minutes=7))
    assert result.status == "minute_processing_failure"
    assert store.five_minute_count() == 0


def test_aggregator_writes_only_to_configured_alpaca_database(tmp_path):
    store_path = tmp_path / "separate_alpaca.sqlite3"
    store = AlpacaBarStore(store_path)
    agg = AlpacaFiveMinuteAggregator(store, finalization_buffer_seconds=0)
    start = utc_open()
    send_bucket(agg, start, now=start + timedelta(minutes=5))
    assert store.five_minute_count() == 1
    assert store_path.is_file()

