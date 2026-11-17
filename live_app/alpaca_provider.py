"""Standalone authenticated Alpaca IEX minute-bar WebSocket adapter.

This module only delivers validated one-minute bar events. It does not persist,
aggregate, generate predictions, or connect to the application engine.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from config import ALPACA_SETTINGS, AlpacaSettings

ALPACA_IEX_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
EXPECTED_SYMBOL = "ECL"
_SILENT_WEBSOCKET_LOGGER = logging.Logger("live_app.alpaca_websocket", level=logging.CRITICAL + 1)
_SILENT_WEBSOCKET_LOGGER.disabled = True
_SILENT_WEBSOCKET_LOGGER.propagate = False
_LOGGER = logging.getLogger(__name__)


def _safe_status_code(value: Any) -> int | None:
    """Return only a bounded numeric protocol status; never expose message text."""
    return value if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599 else None


def _http_status_from_exception(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return _safe_status_code(getattr(response, "status_code", None) or getattr(exc, "status_code", None))


class AlpacaStreamError(RuntimeError):
    """Base error whose message is always a safe, non-provider string."""

    def __init__(self, safe_code: str):
        self.safe_code = safe_code
        super().__init__(safe_code)


class AuthenticationRejected(AlpacaStreamError):
    def __init__(self, safe_code: str = "rejected", provider_status_code: int | None = None):
        self.provider_status_code = _safe_status_code(provider_status_code)
        super().__init__(safe_code)


class ConnectionLimitExceeded(AlpacaStreamError):
    """Alpaca protocol error 406: another stream already uses the account slot."""

    provider_status_code = 406

    def __init__(self):
        super().__init__("connection_limit_exceeded")


class SubscriptionRejected(AlpacaStreamError):
    def __init__(self, safe_code: str = "rejected", provider_status_code: int | None = None):
        self.provider_status_code = _safe_status_code(provider_status_code)
        super().__init__(safe_code)


class InvalidMessage(AlpacaStreamError):
    pass


class ServerReportedError(AlpacaStreamError):
    def __init__(self, code: Any = None):
        safe_code = str(code) if isinstance(code, int) and 0 <= code <= 999 else "unknown"
        self.provider_status_code = _safe_status_code(code)
        super().__init__(f"server_error_{safe_code}")


@dataclass(frozen=True)
class MinuteBarEvent:
    symbol: str
    original_event_timestamp: str
    event_time_utc: datetime
    received_at_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    trade_count: int | None
    is_updated: bool = False


@dataclass
class AlpacaConnectionStatus:
    connected: bool = False
    authenticated: bool = False
    subscribed: bool = False
    run_active: bool = False
    authentication_success_observed: bool = False
    subscription_success_observed: bool = False
    authentication_success_count: int = 0
    subscription_success_count: int = 0
    last_event_time: str | None = None
    last_receive_time: str | None = None
    receive_delay_seconds: float | None = None
    reconnect_count: int = 0
    last_reconnect_delay_seconds: float | None = None
    last_safe_error: str | None = None
    last_error_category: str | None = None
    last_error_transport: str | None = None
    last_error_status_code: int | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidMessage(f"{field_name}_must_be_timezone_aware")
    return value.astimezone(timezone.utc)


def _decode_payload(payload: str | bytes | dict | list) -> list[dict]:
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            payload = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise InvalidMessage("malformed_json") from None
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise InvalidMessage("invalid_message_shape")
    return payload


def _finite_number(item: dict, key: str, *, optional: bool = False) -> float | None:
    if key not in item and optional:
        return None
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InvalidMessage(f"invalid_{key}")
    return float(value)


def parse_minute_bar(payload: str | bytes | dict, received_at: datetime | None = None) -> MinuteBarEvent:
    """Parse one Alpaca stock minute-bar event and validate all supplied fields."""
    messages = _decode_payload(payload)
    if len(messages) != 1:
        raise InvalidMessage("expected_single_bar_event")
    item = messages[0]
    event_type = item.get("T")
    if event_type not in {"b", "u"}:
        if item.get("T") == "error":
            raise ServerReportedError(item.get("code"))
        raise InvalidMessage("unexpected_event_type")
    if item.get("S") != EXPECTED_SYMBOL:
        raise InvalidMessage("unexpected_symbol")
    raw_timestamp = item.get("t")
    if not isinstance(raw_timestamp, str) or not raw_timestamp:
        raise InvalidMessage("missing_event_timestamp")
    try:
        normalized = raw_timestamp[:-1] + "+00:00" if raw_timestamp.endswith("Z") else raw_timestamp
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise InvalidMessage("invalid_event_timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise InvalidMessage("event_timestamp_must_be_utc")
    event_time = _aware_utc(parsed, "event_timestamp")
    receive_time = _aware_utc(received_at or _utc_now(), "receive_timestamp")

    open_value = _finite_number(item, "o")
    high_value = _finite_number(item, "h")
    low_value = _finite_number(item, "l")
    close_value = _finite_number(item, "c")
    volume = _finite_number(item, "v", optional=True)
    if any(value <= 0 for value in (open_value, high_value, low_value, close_value)):
        raise InvalidMessage("nonpositive_ohlc")
    if high_value < max(open_value, low_value, close_value) or low_value > min(open_value, high_value, close_value):
        raise InvalidMessage("invalid_ohlc_range")
    if volume is not None and volume < 0:
        raise InvalidMessage("invalid_volume")

    count = item.get("n") if "n" in item else None
    if "n" in item and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise InvalidMessage("invalid_trade_count")

    return MinuteBarEvent(
        symbol=EXPECTED_SYMBOL,
        original_event_timestamp=raw_timestamp,
        event_time_utc=event_time,
        received_at_utc=receive_time,
        open=open_value,
        high=high_value,
        low=low_value,
        close=close_value,
        volume=volume,
        trade_count=count,
        is_updated=event_type == "u",
    )


class AlpacaIEXWebSocketClient:
    """Reconnectable ECL IEX bar subscriber, deliberately independent of engine/storage."""

    def __init__(
        self,
        settings: AlpacaSettings = ALPACA_SETTINGS,
        *,
        connector: Callable[..., Any] | None = None,
        on_bar: Callable[[MinuteBarEvent], Any] | None = None,
        on_connection_state: Callable[[bool, datetime, str], Any] | None = None,
        on_processing_failure: Callable[[datetime | None, datetime, str], Any] | None = None,
        receive_clock: Callable[[], datetime] = _utc_now,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        ack_timeout_seconds: float = 10.0,
        ping_interval_seconds: float = 20.0,
        ping_timeout_seconds: float = 20.0,
    ):
        if initial_backoff_seconds <= 0 or max_backoff_seconds < initial_backoff_seconds:
            raise ValueError("Invalid reconnect backoff bounds")
        self._settings = settings
        self._connector = connector or self._default_connector
        self._on_bar = on_bar
        self._on_connection_state = on_connection_state
        self._on_processing_failure = on_processing_failure
        self._announced_stream = False
        self._receive_clock = receive_clock
        self._initial_backoff = initial_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._ack_timeout = ack_timeout_seconds
        self._ping_interval = ping_interval_seconds
        self._ping_timeout = ping_timeout_seconds
        self._status = AlpacaConnectionStatus()
        self._shutdown_event: asyncio.Event | None = None
        self._run_active = False
        self._run_started = False

    def __repr__(self) -> str:
        return "AlpacaIEXWebSocketClient(feed='iex', symbol='ECL')"

    @staticmethod
    def _default_connector(uri: str, **kwargs):
        # Lazy import permits mocked adapter tests without installing or using a socket.
        from websockets.asyncio.client import connect

        return connect(uri, **kwargs)

    @property
    def status(self) -> dict:
        return asdict(self._status)

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """Own one stream supervisor: one initial attempt and at most one retry."""
        # Set synchronously before the first await so concurrent callers on this
        # client instance cannot create overlapping WebSocket sessions. The
        # instance is single-use; a fresh application lifecycle creates a fresh
        # provider with a fresh, bounded attempt budget.
        if self._run_active or self._run_started:
            return
        self._run_active = True
        self._run_started = True
        self._status.run_active = True
        try:
            await self._run_with_bounded_reconnect(stop_event)
        finally:
            self._status.connected = False
            self._status.authenticated = False
            self._status.subscribed = False
            self._status.run_active = False
            self._run_active = False

    async def _run_with_bounded_reconnect(self, stop_event: asyncio.Event) -> None:
        self._shutdown_event = stop_event
        if not self._settings.display_authorized:
            self._status.last_safe_error = "display_authorization_required"
            return
        if not self._settings.credentials_configured:
            self._status.last_safe_error = "credentials_missing"
            return

        await self._notify_connection_state(False, "stream_unavailable")
        attempts = 0
        while not stop_event.is_set():
            attempts += 1
            retryable = False
            try:
                await self._run_session(stop_event)
                if stop_event.is_set():
                    break
                self._status.last_safe_error = "connection_closed"
                self._record_failure("connection_closed", "websocket", None)
                retryable = True
            except asyncio.CancelledError:
                raise
            except AuthenticationRejected as exc:
                self._status.last_safe_error = "authentication_rejected"
                self._record_failure("authentication_rejected", "websocket", exc.provider_status_code)
                break
            except ConnectionLimitExceeded as exc:
                self._status.last_safe_error = exc.safe_code
                self._record_failure(exc.safe_code, "websocket", exc.provider_status_code)
                # A second connection attempt cannot free the account's single
                # stream slot. Stop and require the other client to be closed.
                break
            except SubscriptionRejected as exc:
                self._status.last_safe_error = "subscription_rejected"
                self._record_failure("subscription_rejected", "websocket", exc.provider_status_code)
                break
            except ServerReportedError as exc:
                self._status.last_safe_error = exc.safe_code
                status_code = _safe_status_code(getattr(exc, "provider_status_code", None))
                self._record_failure("server_reported_error", "websocket", status_code)
                retryable = status_code is not None and 500 <= status_code <= 599
            except Exception as exc:
                # Never stringify provider/library exceptions: they may contain
                # handshake details, headers, or other sensitive material.
                status_code = _http_status_from_exception(exc)
                if status_code == 406:
                    self._status.last_safe_error = "connection_limit_exceeded"
                    self._record_failure("connection_limit_exceeded", "http_handshake", 406)
                    break
                if status_code in {401, 402, 403}:
                    self._status.last_safe_error = "authentication_rejected"
                    self._record_failure("authentication_rejected", "http_handshake", status_code)
                    break
                category = "connection_failed"
                self._status.last_safe_error = category
                self._record_failure(category, "http_handshake" if status_code is not None else "websocket", status_code)
                retryable = (status_code is None or status_code in {408, 425, 429}
                             or (status_code is not None and 500 <= status_code <= 599))

            self._status.connected = False
            self._status.authenticated = False
            self._status.subscribed = False
            if stop_event.is_set() or not retryable or attempts >= 2:
                break
            # _run_session has exited its async-with block before control gets
            # here, so the old socket's close handshake has completed before
            # another connector call can occur.
            delay = min(self._initial_backoff, self._max_backoff)
            self._status.last_reconnect_delay_seconds = delay
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                if not stop_event.is_set():
                    self._status.reconnect_count += 1
                    continue
            break

    async def _run_session(self, stop_event: asyncio.Event) -> None:
        self._status.connected = False
        self._status.authenticated = False
        self._status.subscribed = False
        connection = self._connector(
            ALPACA_IEX_WS_URL,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            close_timeout=5,
            logger=_SILENT_WEBSOCKET_LOGGER,
        )
        async with connection as websocket:
            self._status.connected = True
            try:
                if not await self._expect_control(websocket, "connected", AuthenticationRejected):
                    return
                await websocket.send(json.dumps({
                    "action": "auth",
                    "key": self._settings.api_key_id,
                    "secret": self._settings.api_secret_key,
                }, separators=(",", ":")))
                if not await self._expect_control(websocket, "authenticated", AuthenticationRejected):
                    return
                self._status.authenticated = True
                self._status.authentication_success_observed = True
                self._status.authentication_success_count += 1
                await websocket.send(json.dumps({"action": "subscribe", "bars": [EXPECTED_SYMBOL],
                                                 "updatedBars": [EXPECTED_SYMBOL]}, separators=(",", ":")))
                if not await self._expect_subscription(websocket):
                    return
                self._status.subscribed = True
                self._status.subscription_success_observed = True
                self._status.subscription_success_count += 1
                await self._notify_connection_state(True, "subscribed")
                await self._consume(websocket, stop_event)
            finally:
                if self._announced_stream:
                    await self._notify_connection_state(False, "disconnected")
                self._status.connected = False
                self._status.authenticated = False
                self._status.subscribed = False

    async def _notify_connection_state(self, connected: bool, reason: str) -> None:
        if self._on_connection_state is None:
            return
        self._announced_stream = connected
        result = self._on_connection_state(connected, _utc_now(), reason)
        if inspect.isawaitable(result):
            await result

    async def _receive_control_frame(self, websocket) -> list[dict] | None:
        receive_task = asyncio.create_task(websocket.recv())
        stop_task = asyncio.create_task(self._shutdown_event.wait()) if self._shutdown_event else None
        tasks = {receive_task}
        if stop_task:
            tasks.add(stop_task)
        done, pending = await asyncio.wait(tasks, timeout=self._ack_timeout, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if stop_task and stop_task in done and stop_task.result():
            return None
        if receive_task not in done:
            raise InvalidMessage("acknowledgement_timeout") from None
        raw = receive_task.result()
        return _decode_payload(raw)

    async def _expect_control(self, websocket, expected: str, failure_type) -> bool:
        messages = await self._receive_control_frame(websocket)
        if messages is None:
            return False
        for item in messages:
            if item.get("T") == "error":
                if _safe_status_code(item.get("code")) == 406:
                    raise ConnectionLimitExceeded()
                if failure_type is AuthenticationRejected:
                    if expected == "authenticated":
                        raise AuthenticationRejected("rejected", _safe_status_code(item.get("code")))
                raise ServerReportedError(item.get("code"))
        if not any(item.get("T") == "success" and item.get("msg") == expected for item in messages):
            if expected == "authenticated":
                raise AuthenticationRejected("unexpected_acknowledgement")
            raise ServerReportedError()
        return True

    async def _expect_subscription(self, websocket) -> bool:
        messages = await self._receive_control_frame(websocket)
        if messages is None:
            return False
        for item in messages:
            if item.get("T") == "error":
                if _safe_status_code(item.get("code")) == 406:
                    raise ConnectionLimitExceeded()
                raise SubscriptionRejected("rejected", _safe_status_code(item.get("code")))
        ack = next((item for item in messages if item.get("T") == "subscription"), None)
        if ack is None or ack.get("bars") != [EXPECTED_SYMBOL] or ack.get("updatedBars") != [EXPECTED_SYMBOL]:
            raise SubscriptionRejected("unexpected_subscription_acknowledgement")
        if any(key not in {"bars", "updatedBars", "T"} and value for key, value in ack.items()):
            raise SubscriptionRejected("unexpected_channel_subscription")
        return True

    def _record_failure(self, category: str, transport: str, status_code: int | None) -> None:
        """Expose and log only safe failure metadata, never provider message text."""
        safe_status = _safe_status_code(status_code)
        self._status.last_error_category = category
        self._status.last_error_transport = transport
        self._status.last_error_status_code = safe_status
        _LOGGER.warning(
            "Alpaca IEX stream failure category=%s transport=%s status_code=%s",
            category, transport, safe_status if safe_status is not None else "unavailable",
        )

    async def _consume(self, websocket, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            receive_task = asyncio.create_task(websocket.recv())
            stop_task = asyncio.create_task(stop_event.wait())
            done, pending = await asyncio.wait(
                {receive_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if stop_task in done and stop_task.result():
                if receive_task in done:
                    receive_task.cancel()
                return
            stop_task.cancel()
            raw = receive_task.result()
            failed_event_time: datetime | None = None
            failed_receive_time = self._receive_clock()
            try:
                messages = _decode_payload(raw)
                for item in messages:
                    failed_event_time = None
                    if item.get("T") == "error":
                        if _safe_status_code(item.get("code")) == 406:
                            raise ConnectionLimitExceeded()
                        raise ServerReportedError(item.get("code"))
                    if item.get("T") == "subscription":
                        continue
                    if item.get("T") not in {"b", "u"}:
                        raise InvalidMessage("unexpected_event_type")
                    if item.get("S") == EXPECTED_SYMBOL:
                        failed_event_time = _safe_raw_event_time(item)
                    failed_receive_time = self._receive_clock()
                    event = parse_minute_bar(item, failed_receive_time)
                    self._status.last_event_time = event.event_time_utc.isoformat()
                    self._status.last_receive_time = event.received_at_utc.isoformat()
                    self._status.receive_delay_seconds = (event.received_at_utc - event.event_time_utc).total_seconds()
                    if self._on_bar is not None:
                        result = self._on_bar(event)
                        if inspect.isawaitable(result):
                            await result
            except InvalidMessage as exc:
                self._status.last_safe_error = exc.safe_code
                if self._on_processing_failure is not None:
                    result = self._on_processing_failure(failed_event_time, failed_receive_time, exc.safe_code)
                    if inspect.isawaitable(result):
                        try:
                            await result
                        except Exception:
                            self._status.last_safe_error = "processing_failure_record_failed"
            except ServerReportedError:
                raise
            except Exception:
                # Callback failures are also redacted; they never include details.
                self._status.last_safe_error = "bar_handler_failed"


def _safe_raw_event_time(item: dict[str, Any]) -> datetime | None:
    """Extract only a valid UTC event timestamp for safe bucket invalidation."""
    raw = item.get("t")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        return None
    return parsed.astimezone(timezone.utc)

