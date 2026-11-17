from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from config import get_alpaca_settings
from live_app.alpaca_provider import (
    ALPACA_IEX_WS_URL,
    AlpacaIEXWebSocketClient,
    AuthenticationRejected,
    InvalidMessage,
    ServerReportedError,
    SubscriptionRejected,
    parse_minute_bar,
)

SECRET = "synthetic-secret-for-redaction-tests"
KEY_ID = "synthetic-key-id"
RECEIVED = datetime(2026, 11, 12, 15, 1, tzinfo=timezone.utc)


def settings(*, authorized=True, credentials=True):
    values = {
        "ALPACA_IEX_DISPLAY_AUTHORIZED": "true" if authorized else "false",
    }
    if credentials:
        values.update({"APCA_API_KEY_ID": KEY_ID, "APCA_API_SECRET_KEY": SECRET})
    return get_alpaca_settings(values)


def frame(message):
    return json.dumps(message)


def handshake_frames(*, auth_ok=True, subscription_ok=True):
    result = [frame({"T": "success", "msg": "connected"})]
    result.append(frame({"T": "success", "msg": "authenticated"}) if auth_ok else
                  frame({"T": "error", "code": 402, "msg": f"bad credential {SECRET}"}))
    if subscription_ok:
        result.append(frame({"T": "subscription", "bars": ["ECL"], "updatedBars": ["ECL"]}))
    else:
        result.append(frame({"T": "error", "code": 410, "msg": f"subscription rejected {SECRET}"}))
    return result


def valid_bar(**overrides):
    value = {"T": "b", "S": "ECL", "t": "2026-11-12T15:00:00Z", "o": 100.0,
             "h": 101.0, "l": 99.0, "c": 100.5, "v": 125.0, "n": 14}
    value.update(overrides)
    return value


class FakeSocket:
    def __init__(self, frames, *, stop_on_subscription_recv=None, close_when_empty=True):
        self.frames = deque(frames)
        self.sent = []
        self.stop_on_subscription_recv = stop_on_subscription_recv
        self.close_when_empty = close_when_empty
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True
        return False

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        if self.frames:
            item = self.frames.popleft()
            if self.stop_on_subscription_recv and '"subscription"' in item:
                self.stop_on_subscription_recv.set()
            return item
        if self.close_when_empty:
            raise ConnectionError("synthetic disconnect")
        await asyncio.Future()


def test_authentication_subscription_and_ecl_bar_are_validated():
    async def case():
        stop = asyncio.Event()
        received = []
        socket = FakeSocket([*handshake_frames(), frame(valid_bar())])
        calls = []
        status_during_callback = []
        client = None

        def connector(uri, **kwargs):
            calls.append((uri, kwargs))
            return socket

        async def on_bar(event):
            received.append(event)
            status_during_callback.append(client.status)
            stop.set()

        client = AlpacaIEXWebSocketClient(settings(), connector=connector, on_bar=on_bar,
                                          receive_clock=lambda: RECEIVED)
        await client.run_forever(stop)

        assert len(calls) == 1
        assert calls[0][0] == ALPACA_IEX_WS_URL
        assert calls[0][1]["ping_interval"] > 0 and calls[0][1]["ping_timeout"] > 0
        assert [item["action"] for item in socket.sent] == ["auth", "subscribe"]
        assert socket.sent[0] == {"action": "auth", "key": KEY_ID, "secret": SECRET}
        assert socket.sent[1] == {"action": "subscribe", "bars": ["ECL"], "updatedBars": ["ECL"]}
        assert client.status["connected"] is False  # clean shutdown closes the socket
        assert client.status["authenticated"] is False
        assert client.status["subscribed"] is False
        assert client.status["authentication_success_observed"] is True
        assert client.status["subscription_success_observed"] is True
        assert client.status["authentication_success_count"] == 1
        assert client.status["subscription_success_count"] == 1
        assert socket.closed is True
        assert len(received) == 1
        assert received[0].original_event_timestamp == "2026-11-12T15:00:00Z"
        assert received[0].event_time_utc == datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)
        assert received[0].received_at_utc == RECEIVED
        assert status_during_callback[0]["connected"] is True
        assert status_during_callback[0]["authenticated"] is True
        assert status_during_callback[0]["subscribed"] is True
        assert client.status["last_event_time"] == "2026-11-12T15:00:00+00:00"
        assert client.status["receive_delay_seconds"] == 60

    asyncio.run(case())


def test_authentication_failure_logs_only_safe_category_and_websocket_status(caplog):
    async def case():
        stop = asyncio.Event()
        sockets = [FakeSocket(handshake_frames(auth_ok=False))]
        calls = []
        client = AlpacaIEXWebSocketClient(settings(), connector=lambda *a, **k: calls.append(1) or sockets.pop(0))
        await client.run_forever(stop)
        assert len(calls) == 1
        assert client.status["last_safe_error"] == "authentication_rejected"
        assert client.status["last_error_category"] == "authentication_rejected"
        assert client.status["last_error_transport"] == "websocket"
        assert client.status["last_error_status_code"] == 402
        assert SECRET not in repr(client)
        assert SECRET not in repr(client.status)

    with caplog.at_level(logging.WARNING, logger="live_app.alpaca_provider"):
        asyncio.run(case())
    assert "category=authentication_rejected" in caplog.text
    assert "transport=websocket" in caplog.text
    assert "status_code=402" in caplog.text
    assert SECRET not in caplog.text


def test_connection_limit_406_before_connected_ack_is_not_mislabelled_or_retried(caplog):
    async def case():
        stop = asyncio.Event()
        first = FakeSocket([frame({"T":"error", "code":406, "msg":f"connection limit {SECRET}"})])
        calls = []
        client = AlpacaIEXWebSocketClient(settings(), connector=lambda *_a, **_k: calls.append(1) or first,
            initial_backoff_seconds=.001, max_backoff_seconds=.002)
        await client.run_forever(stop)
        assert calls == [1]
        assert client.status["last_safe_error"] == "connection_limit_exceeded"
        assert client.status["last_error_category"] == "connection_limit_exceeded"
        assert client.status["last_error_transport"] == "websocket"
        assert client.status["last_error_status_code"] == 406

    with caplog.at_level(logging.WARNING, logger="live_app.alpaca_provider"):
        asyncio.run(case())
    assert "category=connection_limit_exceeded" in caplog.text
    assert "status_code=406" in caplog.text
    assert SECRET not in caplog.text


def test_connection_limit_406_during_auth_is_not_reported_as_bad_credentials(caplog):
    async def case():
        socket = FakeSocket([
            frame({"T": "success", "msg": "connected"}),
            frame({"T": "error", "code": 406, "msg": f"connection limit {SECRET}"}),
        ])
        client = AlpacaIEXWebSocketClient(settings(), connector=lambda *_a, **_k: socket)
        await client.run_forever(asyncio.Event())
        assert [item["action"] for item in socket.sent] == ["auth"]
        assert client.status["last_safe_error"] == "connection_limit_exceeded"
        assert client.status["last_error_category"] == "connection_limit_exceeded"
        assert client.status["last_error_status_code"] == 406
        assert SECRET not in repr(client.status)

    with caplog.at_level(logging.WARNING, logger="live_app.alpaca_provider"):
        asyncio.run(case())
    assert "connection_limit_exceeded" in caplog.text
    assert SECRET not in caplog.text


def test_http_handshake_status_is_logged_without_exception_text_or_credentials(caplog):
    class SyntheticHTTPStatusError(Exception):
        def __init__(self):
            super().__init__(f"sensitive headers {SECRET}")
            self.response = SimpleNamespace(status_code=401)

    async def case():
        stop = asyncio.Event()

        def connector(*_args, **_kwargs):
            stop.set()
            raise SyntheticHTTPStatusError()

        client = AlpacaIEXWebSocketClient(settings(), connector=connector,
                                          initial_backoff_seconds=.01, max_backoff_seconds=.01)
        await client.run_forever(stop)
        assert client.status["last_safe_error"] == "authentication_rejected"
        assert client.status["last_error_category"] == "authentication_rejected"
        assert client.status["last_error_transport"] == "http_handshake"
        assert client.status["last_error_status_code"] == 401

    with caplog.at_level(logging.WARNING, logger="live_app.alpaca_provider"):
        asyncio.run(case())
    assert "transport=http_handshake" in caplog.text
    assert "status_code=401" in caplog.text
    assert SECRET not in caplog.text


def test_subscription_rejection_is_reported_without_provider_text():
    async def case():
        socket = FakeSocket(handshake_frames(subscription_ok=False))
        client = AlpacaIEXWebSocketClient(settings(), connector=lambda *_a, **_k: socket)
        await client.run_forever(asyncio.Event())
        assert client.status["last_safe_error"] == "subscription_rejected"
        assert SECRET not in repr(client.status)

    asyncio.run(case())


def test_display_gate_off_and_missing_credentials_prevent_connection():
    async def case():
        called = []
        unauthorized = AlpacaIEXWebSocketClient(settings(authorized=False), connector=lambda *_a, **_k: called.append(1))
        await unauthorized.run_forever(asyncio.Event())
        missing = AlpacaIEXWebSocketClient(settings(credentials=False), connector=lambda *_a, **_k: called.append(1))
        await missing.run_forever(asyncio.Event())
        assert called == []
        assert unauthorized.status["last_safe_error"] == "display_authorization_required"
        assert missing.status["last_safe_error"] == "credentials_missing"

    asyncio.run(case())


def test_valid_minute_bar_preserves_raw_and_normalizes_utc_times():
    event = parse_minute_bar(valid_bar(), RECEIVED)
    assert event.symbol == "ECL"
    assert event.original_event_timestamp == "2026-11-12T15:00:00Z"
    assert event.event_time_utc.tzinfo is timezone.utc
    assert event.received_at_utc == RECEIVED
    assert (event.open, event.high, event.low, event.close) == (100.0, 101.0, 99.0, 100.5)
    assert event.volume == 125.0 and event.trade_count == 14


def test_updated_bar_message_is_parsed_as_full_replacement():
    event = parse_minute_bar(valid_bar(T="u", c=100.75, v=130, n=15), RECEIVED)
    assert event.is_updated is True
    assert event.close == 100.75 and event.volume == 130 and event.trade_count == 15


def test_updated_bar_channel_is_subscribed_and_event_is_delivered():
    async def case():
        stop = asyncio.Event()
        delivered = []
        states = []
        revised = valid_bar(T="u", h=102.0, c=101.25, v=150, n=18)
        socket = FakeSocket([*handshake_frames(), frame(revised)])

        async def on_bar(event):
            delivered.append(event)
            stop.set()

        client = AlpacaIEXWebSocketClient(
            settings(), connector=lambda *_a, **_k: socket, on_bar=on_bar,
            on_connection_state=lambda *args: states.append(args), receive_clock=lambda: RECEIVED,
        )
        await client.run_forever(stop)
        assert socket.sent[1] == {"action": "subscribe", "bars": ["ECL"], "updatedBars": ["ECL"]}
        assert len(delivered) == 1 and delivered[0].is_updated
        assert delivered[0].close == 101.25
        assert [state[0] for state in states] == [False, True, False]

    asyncio.run(case())


def test_invalid_ecl_bar_reports_only_timestamp_and_safe_failure_category():
    async def case():
        stop = asyncio.Event()
        failures = []
        socket = FakeSocket([*handshake_frames(), frame(valid_bar(h=98.0)), frame(valid_bar())])

        async def on_bar(_event):
            stop.set()

        client = AlpacaIEXWebSocketClient(
            settings(), connector=lambda *_a, **_k: socket, on_bar=on_bar,
            on_processing_failure=lambda *args: failures.append(args), receive_clock=lambda: RECEIVED,
        )
        await asyncio.wait_for(client.run_forever(stop), timeout=2)
        assert failures == [(datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc), RECEIVED, "invalid_ohlc_range")]

    asyncio.run(case())


@pytest.mark.parametrize("payload", [
    valid_bar(S="AAPL"),
    valid_bar(T="t"),
    valid_bar(h=99.5),
    valid_bar(l=100.1),
    valid_bar(o=float("nan")),
    valid_bar(v=-1),
    valid_bar(v=float("inf")),
    valid_bar(n=-1),
    valid_bar(n=1.5),
    valid_bar(t="2026-11-12T15:00:00"),
    valid_bar(t="2026-11-12T17:00:00+02:00"),
    valid_bar(t="not-a-timestamp"),
])
def test_invalid_minute_bar_fields_are_rejected(payload):
    with pytest.raises(InvalidMessage):
        parse_minute_bar(payload, RECEIVED)


def test_volume_and_trade_count_may_be_absent():
    bar = valid_bar()
    del bar["v"]
    del bar["n"]
    event = parse_minute_bar(bar, RECEIVED)
    assert event.volume is None
    assert event.trade_count is None


def test_null_volume_or_trade_count_are_invalid_when_fields_are_present():
    with pytest.raises(InvalidMessage, match="invalid_v"):
        parse_minute_bar(valid_bar(v=None), RECEIVED)
    with pytest.raises(InvalidMessage, match="invalid_trade_count"):
        parse_minute_bar(valid_bar(n=None), RECEIVED)


def test_malformed_json_and_unexpected_event_type_are_safe():
    with pytest.raises(InvalidMessage, match="malformed_json"):
        parse_minute_bar("{bad json", RECEIVED)
    with pytest.raises(InvalidMessage, match="unexpected_event_type"):
        parse_minute_bar({"T": "q", "S": "ECL"}, RECEIVED)


def test_malformed_stream_frame_is_skipped_and_next_bar_is_delivered():
    async def case():
        stop = asyncio.Event()
        delivered = []
        socket = FakeSocket([*handshake_frames(), "{malformed", frame(valid_bar())])

        async def on_bar(event):
            delivered.append(event)
            stop.set()

        client = AlpacaIEXWebSocketClient(settings(), connector=lambda *_a, **_k: socket,
                                          on_bar=on_bar, receive_clock=lambda: RECEIVED)
        await client.run_forever(stop)
        assert len(delivered) == 1
        assert client.status["last_safe_error"] == "malformed_json"

    asyncio.run(case())


def test_server_errors_are_sanitized():
    with pytest.raises(ServerReportedError) as error:
        parse_minute_bar({"T": "error", "code": 500, "msg": f"server echoed {SECRET}"}, RECEIVED)
    assert error.value.safe_code == "server_error_500"
    assert SECRET not in str(error.value)


def test_disconnect_reconnects_and_resubscribes_with_capped_backoff():
    async def case():
        stop = asyncio.Event()
        sockets = [
            FakeSocket(handshake_frames(), close_when_empty=True),
            FakeSocket(handshake_frames(), stop_on_subscription_recv=stop),
        ]
        first_socket = sockets[0]
        calls = []

        def connector(uri, **kwargs):
            if calls:
                assert first_socket.closed is True
            calls.append(uri)
            return sockets.pop(0)

        client = AlpacaIEXWebSocketClient(settings(), connector=connector,
            initial_backoff_seconds=0.001, max_backoff_seconds=0.002)
        await client.run_forever(stop)
        assert len(calls) == 2
        assert client.status["reconnect_count"] == 1
        assert client.status["last_reconnect_delay_seconds"] == 0.001
        assert all(call == ALPACA_IEX_WS_URL for call in calls)

    asyncio.run(case())


def test_second_transient_failure_stops_after_one_reconnect():
    async def case():
        sockets = [FakeSocket(handshake_frames()), FakeSocket(handshake_frames())]
        opened_sockets = []
        calls = []

        def connector(*_args, **_kwargs):
            calls.append(1)
            socket = sockets.pop(0)
            opened_sockets.append(socket)
            return socket

        client = AlpacaIEXWebSocketClient(settings(), connector=connector,
            initial_backoff_seconds=.001, max_backoff_seconds=.01)
        await asyncio.wait_for(client.run_forever(asyncio.Event()), timeout=2)
        assert len(calls) == 2
        assert client.status["reconnect_count"] == 1
        assert client.status["last_safe_error"] == "connection_failed"
        assert all(socket.closed for socket in opened_sockets)

    asyncio.run(case())


def test_concurrent_run_forever_calls_do_not_open_overlapping_sessions():
    async def case():
        stop = asyncio.Event()
        calls = []
        socket = FakeSocket(handshake_frames(), close_when_empty=False)

        def connector(*_args, **_kwargs):
            calls.append(1)
            return socket

        client = AlpacaIEXWebSocketClient(settings(), connector=connector)
        first = asyncio.create_task(client.run_forever(stop))
        for _ in range(100):
            if client.status["subscribed"]:
                break
            await asyncio.sleep(.001)
        assert client.status["connected"] is True
        assert client.status["authenticated"] is True
        assert client.status["subscribed"] is True
        await client.run_forever(asyncio.Event())
        assert calls == [1]
        assert client.status["run_active"] is True
        stop.set()
        await asyncio.wait_for(first, timeout=2)
        assert socket.closed is True
        assert client.status["run_active"] is False

    asyncio.run(case())


def test_http_406_connection_limit_stops_without_reconnect():
    class SyntheticHTTPStatusError(Exception):
        def __init__(self):
            super().__init__(f"sensitive response {SECRET}")
            self.response = SimpleNamespace(status_code=406)

    async def case():
        calls = []

        def connector(*_args, **_kwargs):
            calls.append(1)
            raise SyntheticHTTPStatusError()

        client = AlpacaIEXWebSocketClient(settings(), connector=connector,
            initial_backoff_seconds=.001, max_backoff_seconds=.01)
        await client.run_forever(asyncio.Event())
        assert calls == [1]
        assert client.status["reconnect_count"] == 0
        assert client.status["last_safe_error"] == "connection_limit_exceeded"
        assert client.status["last_error_category"] == "connection_limit_exceeded"
        assert client.status["last_error_transport"] == "http_handshake"
        assert client.status["last_error_status_code"] == 406
        assert SECRET not in repr(client.status)

    asyncio.run(case())


def test_server_error_and_connection_exception_never_leak_secrets():
    async def case():
        stop = asyncio.Event()
        first = FakeSocket([*handshake_frames(), "{" + "\"T\":\"error\",\"code\":500,\"msg\":\"" + SECRET + "\"}"])
        second = FakeSocket(handshake_frames(), stop_on_subscription_recv=stop)
        sockets = deque([first, second])
        client = AlpacaIEXWebSocketClient(settings(), connector=lambda *_a, **_k: sockets.popleft(),
            initial_backoff_seconds=0.001, max_backoff_seconds=0.002)
        await client.run_forever(stop)
        assert client.status["last_safe_error"] == "server_error_500"
        assert SECRET not in repr(client.status)
        assert SECRET not in repr(client)

    asyncio.run(case())


def test_connector_exception_is_redacted_from_safe_status():
    async def case():
        stop = asyncio.Event()

        def connector(*_args, **_kwargs):
            stop.set()
            raise RuntimeError(f"synthetic handshake header contained {SECRET}")

        client = AlpacaIEXWebSocketClient(settings(), connector=connector)
        await client.run_forever(stop)
        assert client.status["last_safe_error"] == "connection_failed"
        assert SECRET not in repr(client.status)
        assert SECRET not in repr(client)

    asyncio.run(case())


