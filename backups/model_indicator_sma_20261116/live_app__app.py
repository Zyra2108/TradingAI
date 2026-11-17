from __future__ import annotations

import asyncio
import csv
import io
import ipaddress
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from config import ALPACA_SETTINGS, TWELVE_DATA_API_KEY
from live_app.calendar import NY, UTC, is_poll_window, session_for_date
from live_app.engine import IndicatorEngine, POLL_SECONDS

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
load_dotenv(ROOT / ".env")
DISPLAY_AUTHORIZED = os.getenv("TWELVE_DATA_DISPLAY_AUTHORIZED", "false").strip().lower() == "true"
ALPACA_MAINTENANCE_INTERVAL_SECONDS = 30
_LOGGER = logging.getLogger(__name__)


def _epoch_utc(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).timestamp())


def _et_display(value: str | None) -> str | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(NY).isoformat(timespec="seconds")


def _market_state(now: datetime) -> str:
    local = now.astimezone(NY)
    session = session_for_date(local.date())
    if session is None:
        return "Market closed (NYSE holiday or weekend)"
    if local < session[0]:
        return "Market closed (pre-open)"
    if local >= session[1]:
        return "Market closed (after close)"
    return "Market open"


def create_app(engine: IndicatorEngine | None = None, run_poller: bool = True, *,
               alpaca_engine=None, alpaca_settings=ALPACA_SETTINGS,
               alpaca_provider_factory=None, historical_replay=None,
               unified_historical_market=None) -> FastAPI:
    """Build legacy Twelve Data mode or the separately gated Alpaca IEX mode."""
    alpaca_mode = engine is None and (alpaca_engine is not None or alpaca_settings.display_authorized)
    if alpaca_mode:
        if alpaca_engine is None:
            from live_app.alpaca_engine import AlpacaPredictionEngine
            alpaca_engine = AlpacaPredictionEngine()
        instance = alpaca_engine
        provider = None

        async def on_alpaca_bar(event):
            try:
                await asyncio.to_thread(instance.ingest_minute_event, event, now=event.received_at_utc)
            except Exception:
                # Mark only the affected session-aligned five-minute bucket;
                # the safe provider callback then records no sensitive detail.
                await asyncio.to_thread(instance.mark_minute_processing_failure,
                                        event.event_time_utc, event.received_at_utc)
                raise
            await asyncio.to_thread(instance.settle_due_predictions, now=event.received_at_utc)

        async def on_alpaca_connection_state(connected, at_utc, reason):
            await asyncio.to_thread(instance.set_stream_connection, connected, at_utc, reason)

        async def on_alpaca_processing_failure(event_time_utc, received_at_utc, safe_reason):
            # If the payload timestamp itself is unusable, conservatively mark
            # the receive-time bucket; no provider text is retained.
            affected_time = event_time_utc or received_at_utc
            await asyncio.to_thread(instance.mark_minute_processing_failure,
                                     affected_time, received_at_utc, f"invalid_stream_event_{safe_reason}")

        if alpaca_settings.display_authorized:
            if alpaca_provider_factory is None:
                from live_app.alpaca_provider import AlpacaIEXWebSocketClient
                provider = AlpacaIEXWebSocketClient(alpaca_settings, on_bar=on_alpaca_bar,
                                                    on_connection_state=on_alpaca_connection_state,
                                                    on_processing_failure=on_alpaca_processing_failure)
            else:
                provider = alpaca_provider_factory(alpaca_settings, on_alpaca_bar,
                                                   on_alpaca_connection_state,
                                                   on_alpaca_processing_failure)
    else:
        instance = engine or IndicatorEngine(TWELVE_DATA_API_KEY or "", DISPLAY_AUTHORIZED)
        provider = None

    stream_task = None
    maintenance_task = None
    stop_event: asyncio.Event | None = None

    async def legacy_poll_loop():
        while True:
            if not instance.display_authorized or not instance.api_key:
                await asyncio.sleep(30)
                continue
            if not is_poll_window(instance.now()):
                await asyncio.sleep(60)
                continue
            await asyncio.to_thread(instance.poll_once)
            await asyncio.sleep(POLL_SECONDS)

    async def alpaca_maintenance_loop(stop: asyncio.Event):
        while not stop.is_set():
            # Revisit persisted minute-bar buckets as well as settlements. A
            # completed bucket's 120s finalization buffer can expire after its
            # last source event, so event-driven finalization alone is not
            # sufficient. Engine recovery is idempotent and skips retroactive
            # predictions after restart.
            try:
                await asyncio.to_thread(instance.recover_aggregations)
            except Exception as exc:
                _LOGGER.error("Alpaca periodic aggregation recovery failed error_type=%s",
                              type(exc).__name__)
            try:
                await asyncio.to_thread(instance.settle_due_predictions)
            except Exception as exc:
                _LOGGER.error("Alpaca periodic settlement check failed error_type=%s",
                              type(exc).__name__)
            try:
                await asyncio.wait_for(stop.wait(), timeout=ALPACA_MAINTENANCE_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal stream_task, maintenance_task, stop_event
        if alpaca_mode:
            await asyncio.to_thread(instance.recover_aggregations)
            await asyncio.to_thread(instance.settle_due_predictions)
            if run_poller and alpaca_settings.display_authorized and provider is not None:
                stop_event = asyncio.Event()
                if stream_task is None or stream_task.done():
                    stream_task = asyncio.create_task(provider.run_forever(stop_event))
                if maintenance_task is None or maintenance_task.done():
                    maintenance_task = asyncio.create_task(alpaca_maintenance_loop(stop_event))
                app.state.stream_task = stream_task
                app.state.maintenance_task = maintenance_task
        elif run_poller:
            stream_task = asyncio.create_task(legacy_poll_loop())
        try:
            yield
        finally:
            if stop_event is not None:
                stop_event.set()
            tasks = [task for task in (stream_task, maintenance_task) if task is not None]
            for task in tasks:
                if task is not None and not task.done():
                    try:
                        await asyncio.wait_for(task, timeout=15)
                    except asyncio.TimeoutError:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
            if not alpaca_mode and stream_task is not None:
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title="ECL Paper Trading Indicator", version="2.0.0", lifespan=lifespan)
    app.state.engine = instance
    app.state.provider = provider
    app.state.mode = "alpaca_iex" if alpaca_mode else "twelve_data"
    app.state.historical_replay = historical_replay
    app.state.unified_historical_market = unified_historical_market
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def loopback_only(request: Request, call_next):
        # Protect both the Alpaca app and the legacy dashboard/report routes.
        # Binding Uvicorn to 127.0.0.1 remains the primary boundary; this also
        # rejects accidental non-loopback binds instead of exposing SQLite data.
        host = request.client.host if request.client else ""
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            return JSONResponse(status_code=403, content={"detail": "Dashboard is local-only"})
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = "alpaca.html" if alpaca_mode else "index.html"
        return (STATIC / page).read_text(encoding="utf-8")

    @app.get("/api/health")
    def health():
        if alpaca_mode:
            return {"ok": True, "mode": "alpaca_iex", "model": "frozen Alpaca candidate",
                    "sha256": instance.model_sha256}
        return {"ok": True, "model": "frozen Balanced LightGBM B", "sha256": instance.model_sha256}

    @app.get("/api/status")
    def status():
        if not alpaca_mode:
            payload = instance.status_payload()
            payload["database"] = str(instance.store.path.relative_to(ROOT)) if instance.store.path.is_relative_to(ROOT) else "configured external path"
            payload["model_sha256"] = instance.model_sha256
            return payload

        now = datetime.now(UTC)
        if not alpaca_settings.display_authorized:
            return {
                "mode": "alpaca_iex", "provider": "Alpaca", "feed": "IEX",
                "display_authorized": False, "state": "authorization_required",
                "message": "Private display is disabled until ALPACA_IEX_DISPLAY_AUTHORIZED=true.",
                "model_ready": True, "model_sha256": instance.model_sha256,
                "market_state": _market_state(now), "missing_data_warnings": [],
            }
        provider_status = getattr(provider, "status", {}) if provider is not None else {}
        if callable(provider_status):
            provider_status = provider_status()
        provider_status = provider_status or {}
        stream_error = provider_status.get("last_safe_error")
        if not alpaca_settings.credentials_configured:
            state, message = "credentials_missing", "Alpaca credentials are not configured."
        elif (provider_status.get("connected") and provider_status.get("authenticated")
              and provider_status.get("subscribed")):
            if stream_error:
                state = "connected_with_warning"
                message = f"Alpaca IEX subscription is active; last safe event status: {stream_error}."
            else:
                state, message = "connected", "Receiving ECL minute bars from Alpaca IEX."
        elif stream_error:
            state = str(stream_error)
            category = provider_status.get("last_error_category")
            transport = provider_status.get("last_error_transport")
            status_code = provider_status.get("last_error_status_code")
            if category and transport:
                status_text = f"status {status_code}" if isinstance(status_code, int) else "status unavailable"
                message = f"Alpaca IEX failure category: {category}; transport: {transport}; {status_text}."
            else:
                message = "Alpaca IEX stream is not currently delivering bars. Safe diagnostics are unavailable."
        elif provider_status.get("connected") and provider_status.get("authenticated"):
            state = "authenticated"
            message = "WebSocket transport is connected and authenticated; waiting for the ECL IEX subscription acknowledgement."
        elif provider_status.get("connected"):
            state = "transport_connected"
            message = "WebSocket transport is connected; awaiting Alpaca authentication acknowledgement."
        else:
            state, message = "connecting", "Waiting for Alpaca IEX stream subscription."
        engine_status = instance.status()
        candles = instance.store.dashboard_candles(1)
        latest_et = _et_display(candles[-1]["candle_time_utc"]) if candles else None
        warnings = instance.store.unresolved_gaps(10)
        return {
            "mode": "alpaca_iex", "provider": "Alpaca", "feed": "IEX",
            "display_authorized": True, "state": state, "message": message,
            "credentials_configured": alpaca_settings.credentials_configured,
            "market_state": _market_state(now), "model_ready": engine_status["model_ready"],
            "feature_ready": engine_status["feature_ready"],
            "last_aggregation_status": engine_status.get("last_aggregation_status"),
            "latest_prediction_attempt": engine_status.get("latest_prediction_attempt"),
            "provider_error": stream_error,
            "engine_error": engine_status.get("last_safe_error"),
            "model_sha256": instance.model_sha256,
            "database": "data/live/ecl_alpaca_iex.sqlite3",
            "connected": bool(provider_status.get("connected")),
            "authenticated": bool(provider_status.get("authenticated")),
            "subscribed": bool(provider_status.get("subscribed")),
            "run_active": bool(provider_status.get("run_active")),
            "authentication_success_observed": bool(provider_status.get("authentication_success_observed")),
            "subscription_success_observed": bool(provider_status.get("subscription_success_observed")),
            "authentication_success_count": int(provider_status.get("authentication_success_count", 0)),
            "subscription_success_count": int(provider_status.get("subscription_success_count", 0)),
            "last_event_time_utc": provider_status.get("last_event_time"),
            "last_event_time_et": _et_display(provider_status.get("last_event_time")),
            "last_receive_time_utc": provider_status.get("last_receive_time"),
            "receive_delay_seconds": provider_status.get("receive_delay_seconds"),
            "reconnect_count": int(provider_status.get("reconnect_count", 0)),
            "last_safe_error": stream_error or engine_status.get("last_safe_error"),
            "last_error_category": provider_status.get("last_error_category"),
            "last_error_transport": provider_status.get("last_error_transport"),
            "last_error_status_code": provider_status.get("last_error_status_code"),
            "last_completed_candle_time_et": latest_et,
            "pending_settlements": engine_status["pending_settlements"],
            "missing_data_warnings": warnings,
            "late_updated_bar_count": engine_status["late_updated_bar_count"],
            "latest_late_updated_bar": engine_status["latest_late_updated_bar"],
            "open_stream_interruption_count": engine_status["open_stream_interruption_count"],
        }

    def require_alpaca_display():
        if alpaca_mode and not alpaca_settings.display_authorized:
            raise HTTPException(status_code=403, detail="Alpaca display authorization is disabled")

    def require_alpaca_mode():
        if not alpaca_mode:
            raise HTTPException(status_code=404, detail="Historical Alpaca replay is unavailable in this mode")
        require_alpaca_display()

    def replay_service():
        if app.state.historical_replay is None:
            from live_app.historical_replay import HistoricalReplay
            app.state.historical_replay = HistoricalReplay(candidate=instance.candidate)
        return app.state.historical_replay

    @app.get("/replay", response_class=HTMLResponse)
    def replay_page():
        require_alpaca_mode()
        return RedirectResponse(url="/", status_code=307)

    def historical_market():
        if app.state.unified_historical_market is None:
            from live_app.unified_market import UnifiedHistoricalMarket
            app.state.unified_historical_market = UnifiedHistoricalMarket()
        return app.state.unified_historical_market

    @app.get("/api/market-timeline")
    def market_timeline():
        require_alpaca_mode()
        try:
            market = historical_market()
            history = market.candles()
            coverage = dict(market.coverage)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail="Permitted historical market data could not be loaded") from exc
        live_rows = instance.store.dashboard_candles(100000)
        merged = {int(row["time"]): row for row in history}
        live_count = 0
        for row in live_rows:
            stamp = _epoch_utc(row["candle_time_utc"])
            merged[stamp] = {
                "time": stamp, "candle_time_utc": row["candle_time_utc"],
                "open": row["open"], "high": row["high"], "low": row["low"],
                "close": row["close"], "volume": row.get("volume"),
                "source": "live", "feed": row.get("feed", "IEX"), "live": True,
            }
            live_count += 1
        candles = [merged[key] for key in sorted(merged)]
        coverage.update({
            "live_candle_count": live_count,
            "timeline_candle_count": len(candles),
            "historical_candle_count": len(history),
            "historical_backfill_status": coverage.get("backfill_manifest_status", "not_present"),
            "timeline_start_utc": candles[0]["candle_time_utc"] if candles else None,
            "timeline_end_utc": candles[-1]["candle_time_utc"] if candles else None,
        })
        return {"source": "Alpaca IEX", "candles": candles, "coverage": coverage}

    @app.get("/api/market-candle-details")
    def market_candle_details(candle_time_utc: str):
        require_alpaca_mode()
        try:
            stamp = datetime.fromisoformat(candle_time_utc.replace("Z", "+00:00"))
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                raise ValueError("timestamp must include timezone")
            stamp = stamp.astimezone(UTC)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Candle timestamp must be timezone-aware ISO-8601") from exc
        record = instance.store.prediction_detail_at(stamp)
        if record is None:
            return {"status": "no_prediction", "message": "No prediction recorded", "prediction": None}
        record["candle_time_et"] = _et_display(record["candle_time_utc"])
        record["prediction_created_et"] = _et_display(record["prediction_created_utc"])
        record["outcome_candle_time_et"] = _et_display(record["outcome_candle_time_utc"])
        settled = record["actual_target"] is not None
        record["settlement_status"] = "SETTLED" if settled else "PENDING"
        if settled:
            record["observed_et"] = _et_display(record["observed_utc"])
        else:
            # Keep an unverified future outcome out of the response entirely.
            for key in ("observed_utc", "endpoint_candle_time_utc", "actual_target", "actual_return_pct", "correct", "observed_et", "actual_close"):
                record.pop(key, None)
        return {"status": "settled" if settled else "pending", "prediction": record}

    @app.get("/api/replay/dates")
    def replay_dates():
        require_alpaca_mode()
        try:
            service = replay_service()
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail="Permitted historical replay data could not be loaded") from exc
        return {"source": "Alpaca IEX", "mode": "historical_replay", "dates": service.dates()}

    @app.get("/api/replay/candles")
    def replay_candles(day: str):
        require_alpaca_mode()
        try:
            return replay_service().candles(day)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/replay/predict")
    def replay_predict(day: str, candle_time: str):
        require_alpaca_mode()
        raise HTTPException(status_code=410, detail="Retrospective inference is disabled; only immutable recorded predictions are shown.")

    @app.get("/api/candles")
    def candles(limit: int = 500):
        if alpaca_mode:
            require_alpaca_display()
            return [{"time": _epoch_utc(row["candle_time_utc"]), "open": row["open"],
                     "high": row["high"], "low": row["low"], "close": row["close"],
                     "volume": row.get("volume"), "source": "live", "feed": row.get("feed", "IEX"),
                     "live": True}
                    for row in instance.store.dashboard_candles(limit)]
        result = []
        for row in instance.store.candles(max(1, min(limit, 2000))):
            dt = datetime.fromisoformat(row["candle_time"])
            dt = dt.replace(tzinfo=NY) if dt.tzinfo is None else dt
            result.append({"time": int(dt.timestamp()), "open": row["open"], "high": row["high"],
                           "low": row["low"], "close": row["close"]})
        return result

    @app.get("/api/predictions")
    def predictions(limit: int = 200):
        if alpaca_mode:
            require_alpaca_display()
            result = []
            for row in instance.store.prediction_history(limit):
                row["candle_time_epoch"] = _epoch_utc(row["candle_time_utc"])
                row["candle_time_et"] = _et_display(row["candle_time_utc"])
                row["prediction_created_et"] = _et_display(row["prediction_created_utc"])
                row["outcome_candle_time_et"] = _et_display(row["outcome_candle_time_utc"])
                row["settlement_status"] = "SETTLED" if row["actual_target"] is not None else "PENDING"
                result.append(row)
            return result
        result = []
        for row in instance.store.predictions(max(1, min(limit, 1000))):
            stamp = datetime.fromisoformat(row["candle_time"])
            stamp = stamp.replace(tzinfo=NY) if stamp.tzinfo is None else stamp
            row["candle_time_epoch"] = int(stamp.timestamp())
            row["input_candle"] = json.loads(row.pop("input_candle_json"))
            row["input_features"] = json.loads(row.pop("input_features_json"))
            result.append(row)
        return result

    @app.get("/api/report")
    def report():
        if alpaca_mode:
            require_alpaca_display()
            return instance.store.performance_report()
        return instance.store.report()

    @app.get("/api/report.csv")
    def report_csv():
        output = io.StringIO()
        writer = csv.writer(output)
        if alpaca_mode:
            require_alpaca_display()
            columns = ["candle_time_utc", "prediction_created_utc", "prediction", "probability_down",
                       "probability_neutral", "probability_up", "confidence", "reference_close",
                       "horizon_minutes", "outcome_candle_time_utc", "actual_target", "actual_return_pct",
                       "correct", "model_sha256"]
            writer.writerow(columns)
            for row in instance.store.prediction_history(100000):
                writer.writerow([row.get(column) for column in columns])
            filename = "ecl_alpaca_iex_prediction_report.csv"
        else:
            writer.writerow(["candle_time", "prediction_created_utc", "prediction", "probability_down",
                             "probability_neutral", "probability_up", "confidence", "horizon_minutes",
                             "outcome_candle_time", "actual_target", "actual_return_pct", "correct",
                             "model_sha256", "input_candle_json", "input_features_json"])
            for row in instance.store.predictions(100000):
                writer.writerow([row.get(k) for k in ["candle_time", "prediction_created_utc", "prediction",
                    "probability_down", "probability_neutral", "probability_up", "confidence", "horizon_minutes",
                    "outcome_candle_time", "actual_target", "actual_return_pct", "correct", "model_sha256",
                    "input_candle_json", "input_features_json"]])
            filename = "ecl_prediction_report.csv"
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={filename}"})

    return app


app = create_app()
