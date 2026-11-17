from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def session_for_date(day: date) -> tuple[datetime, datetime] | None:
    schedule = mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day)
    if schedule.empty:
        return None
    opened = schedule.iloc[0].market_open.tz_convert("America/New_York").to_pydatetime()
    closed = schedule.iloc[0].market_close.tz_convert("America/New_York").to_pydatetime()
    return opened, closed


def is_market_open(now: datetime) -> bool:
    local = now.astimezone(NY)
    bounds = session_for_date(local.date())
    return bool(bounds and bounds[0] <= local < bounds[1])


def is_poll_window(now: datetime, post_close_minutes: int = 5) -> bool:
    """Poll through a short close grace period to retrieve the final OHLC bar."""
    local = now.astimezone(NY)
    bounds = session_for_date(local.date())
    return bool(bounds and bounds[0] <= local < bounds[1] + timedelta(minutes=post_close_minutes))


def next_session_open(now: datetime) -> datetime:
    local = now.astimezone(NY)
    for offset in range(0, 15):
        day = local.date() + timedelta(days=offset)
        bounds = session_for_date(day)
        if bounds and (bounds[0] > local):
            return bounds[0]
    raise RuntimeError("Could not find the next NYSE session within 14 days")


def candle_is_complete(candle_open: datetime, now: datetime, delay_seconds: int) -> bool:
    """Require candle close plus provider-finalization buffer to have elapsed."""
    local_open = candle_open.replace(tzinfo=NY) if candle_open.tzinfo is None else candle_open.astimezone(NY)
    end = local_open + timedelta(minutes=5)
    session = session_for_date(local_open.date())
    if not session or local_open < session[0] or end > session[1]:
        return False
    return now.astimezone(NY) >= end + timedelta(seconds=delay_seconds)
