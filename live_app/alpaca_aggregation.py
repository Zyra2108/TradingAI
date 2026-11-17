"""Strict session-anchored aggregation of Alpaca IEX one-minute bars."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Any

from live_app.alpaca_storage import AlpacaBarStore
from live_app.calendar import NY, session_for_date

UTC = timezone.utc
MINUTE = timedelta(minutes=1)
FIVE_MINUTES = timedelta(minutes=5)


def _aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class AggregatedFiveMinuteCandle:
    candle_time_utc: datetime
    finalized_at_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    trade_count: int | None
    source: str
    feed: str
    aggregation_method: str
    aggregation_metadata: dict[str, Any]


@dataclass(frozen=True)
class AggregationResult:
    status: str
    candle: AggregatedFiveMinuteCandle | None = None
    missing_timestamps_utc: tuple[datetime, ...] = ()
    inserted: bool = False
    conflict: bool = False


class AlpacaFiveMinuteAggregator:
    """Persists input events then finalizes only complete, due five-minute buckets.

    The injected `clock` and `calendar` make the implementation deterministic in
    tests. The default calendar is the project's NYSE calendar helper.
    """

    def __init__(self, store: AlpacaBarStore, *, finalization_buffer_seconds: float = 120,
                 clock: Callable[[], datetime] | None = None,
                 calendar: Callable[[date], tuple[datetime, datetime] | None] = session_for_date):
        if finalization_buffer_seconds < 0:
            raise ValueError("finalization_buffer_seconds must be nonnegative")
        self.store = store
        self.finalization_buffer = timedelta(seconds=finalization_buffer_seconds)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.calendar = calendar
        self._sessions: dict[date, tuple[datetime, datetime] | None] = {}

    def _session(self, day: date) -> tuple[datetime, datetime] | None:
        if day not in self._sessions:
            self._sessions[day] = self.calendar(day)
        return self._sessions[day]

    def _bucket_start_for_event(self, event_time_utc: datetime) -> datetime | None:
        event_time = _aware_utc(event_time_utc, "event_time_utc")
        if event_time.second or event_time.microsecond:
            return None
        session = self._session(event_time.astimezone(NY).date())
        if not session:
            return None
        opened, closed = (_aware_utc(session[0], "session_open"), _aware_utc(session[1], "session_close"))
        if event_time < opened or event_time + MINUTE > closed:
            return None
        offset_seconds = (event_time - opened).total_seconds()
        if offset_seconds % 60:
            return None
        bucket_index = int(offset_seconds // FIVE_MINUTES.total_seconds())
        bucket_start = opened + bucket_index * FIVE_MINUTES
        if bucket_start + FIVE_MINUTES > closed:
            return None
        return bucket_start

    def ingest(self, event: Any, *, now: datetime | None = None) -> AggregationResult:
        """Persist first, then attempt the event's session-anchored bucket."""
        bucket_start = self._bucket_start_for_event(event.event_time_utc)
        if bucket_start is None:
            return AggregationResult("outside_session_or_unaligned")
        if getattr(event, "is_updated", False):
            bucket_end = bucket_start + FIVE_MINUTES
            cutoff = bucket_end + self.finalization_buffer
            inserted, late = self.store.insert_updated_minute_bar(
                event, bucket_start_utc=bucket_start, bucket_end_utc=bucket_end,
                update_cutoff_utc=cutoff,
            )
            if late:
                return AggregationResult("late_updated_bar_ignored", inserted=inserted, conflict=True)
            conflict = False
        else:
            inserted, conflict = self.store.insert_minute_bar(event)
        result = self.finalize_bucket(bucket_start, now=now)
        return AggregationResult(result.status, result.candle, result.missing_timestamps_utc,
                                 inserted, conflict)

    def mark_processing_failure(self, event_time_utc: datetime, *, detected_at_utc: datetime | None = None,
                                reason: str = "minute_processing_failure") -> None:
        bucket = self._bucket_start_for_event(event_time_utc)
        if bucket is not None:
            self.store.record_aggregation_invalidation(bucket, detected_at_utc or self.clock(), reason)

    def finalize_bucket(self, bucket_start_utc: datetime, *, now: datetime | None = None) -> AggregationResult:
        bucket_start = _aware_utc(bucket_start_utc, "bucket_start_utc")
        if bucket_start.second or bucket_start.microsecond:
            return AggregationResult("invalid_bucket_alignment")
        session = self._session(bucket_start.astimezone(NY).date())
        if not session:
            return AggregationResult("outside_session_or_unaligned")
        opened, closed = (_aware_utc(session[0], "session_open"), _aware_utc(session[1], "session_close"))
        offset = (bucket_start - opened).total_seconds()
        if bucket_start < opened or offset % FIVE_MINUTES.total_seconds() or bucket_start + FIVE_MINUTES > closed:
            return AggregationResult("outside_session_or_unaligned")

        clock_now = _aware_utc(now if now is not None else self.clock(), "now")
        bucket_end = bucket_start + FIVE_MINUTES
        if clock_now < bucket_end + self.finalization_buffer:
            return AggregationResult("waiting_for_finalization_buffer")

        rejection = self.store.bucket_rejection_reason(bucket_start, bucket_end, clock_now)
        if rejection:
            self.store.record_gap(bucket_start, clock_now, [], reason=rejection)
            return AggregationResult(rejection)

        expected = [bucket_start + i * MINUTE for i in range(5)]
        update_cutoff = bucket_end + self.finalization_buffer
        rows = self.store.get_minute_bars(bucket_start, bucket_end, update_cutoff_utc=update_cutoff)
        row_by_time = {datetime.fromisoformat(row["event_time_utc"].replace("Z", "+00:00")).astimezone(UTC): row for row in rows}
        missing = [stamp for stamp in expected if stamp not in row_by_time]
        if not row_by_time:
            self.store.record_gap(bucket_start, clock_now, missing, reason="empty_bucket_no_observed_minutes")
            return AggregationResult("empty_bucket", missing_timestamps_utc=tuple(missing))
        if missing:
            self.store.record_gap(bucket_start, clock_now, missing, reason="sparse_observed_minutes")
        ordered = [row_by_time[stamp] for stamp in expected if stamp in row_by_time]
        sources = {row["source"] for row in ordered}
        feeds = {row["feed"] for row in ordered}
        if len(sources) != 1 or len(feeds) != 1:
            return AggregationResult("mixed_source_or_feed")
        volume_values = [row["volume"] for row in ordered if row["volume"] is not None]
        count_values = [row["trade_count"] for row in ordered if row["trade_count"] is not None]
        candle = AggregatedFiveMinuteCandle(
            candle_time_utc=bucket_start,
            finalized_at_utc=clock_now,
            open=float(ordered[0]["open"]),
            high=max(float(row["high"]) for row in ordered),
            low=min(float(row["low"]) for row in ordered),
            close=float(ordered[-1]["close"]),
            volume=sum(volume_values) if volume_values else None,
            trade_count=sum(count_values) if count_values else None,
            source=next(iter(sources)), feed=next(iter(feeds)),
            aggregation_method="observed_minute_ohlc_v2",
            aggregation_metadata={
                "observed_minute_timestamps_utc": [
                    stamp.isoformat(timespec="seconds").replace("+00:00", "Z") for stamp in expected if stamp in row_by_time
                ],
                "observed_minute_count": len(ordered),
                "missing_minute_timestamps_utc": [
                    stamp.isoformat(timespec="seconds").replace("+00:00", "Z") for stamp in missing
                ],
                "session_open_utc": opened.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "session_close_utc": closed.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "volume_and_trade_count_are_audit_only": True,
            },
        )
        inserted, conflict = self.store.insert_five_minute_candle(candle)
        if conflict:
            return AggregationResult("finalized_candle_conflict", candle, tuple(missing), inserted=False, conflict=True)
        return AggregationResult("finalized" if inserted else "already_finalized", candle,
                                 tuple(missing), inserted=inserted)

    def recover_due_buckets(self, *, now: datetime | None = None) -> list[AggregationResult]:
        """Revisit buckets represented in persisted minute data after restart."""
        bucket_starts = {
            bucket for stamp in self.store.minute_bar_timestamps()
            if (bucket := self._bucket_start_for_event(stamp)) is not None
        }
        return [self.finalize_bucket(bucket, now=now) for bucket in sorted(bucket_starts)]
