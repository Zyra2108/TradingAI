from __future__ import annotations

import asyncio
import csv
import io
import ipaddress
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from config import ALPACA_SETTINGS
from live_app.calendar import NY, UTC, session_for_date

STATIC = Path(__file__).resolve().parent / "static"
ALPACA_MAINTENANCE_INTERVAL_SECONDS = 30
ALPACA_REST_CATCHUP_INTERVAL_SECONDS = 300
RESEARCH_PREVIEW_INTERVAL_SECONDS = 30
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


def _eligible_research_preview_candles(candles, *, now: datetime, recorder):
    """Restrict research inference to recent, finalized current-session IEX bars."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Preview eligibility clock must be timezone-aware")
    local_day = now.astimezone(NY).date()
    session = session_for_date(local_day)
    if session is None:
        return []
    opened, closed = (bound.astimezone(UTC) for bound in session)
    eligible = []
    for candle in candles:
        stamp = datetime.fromtimestamp(int(candle["time"]), UTC)
        if str(candle.get("feed", "")).upper() != "IEX":
            continue
        stamp_text = candle.get("candle_time_utc")
        if (stamp.astimezone(NY).date() != local_day or stamp < opened
                or stamp + timedelta(minutes=20) > closed
                or now.astimezone(UTC) - stamp > timedelta(minutes=15)
                or now.astimezone(UTC) < stamp + timedelta(minutes=7)
                or not stamp_text or recorder.is_completed(stamp_text)):
            continue
        eligible.append((stamp, candle))
    return sorted(eligible, key=lambda item: item[0])


def create_app(run_poller: bool = True, *,
               alpaca_engine=None, alpaca_settings=ALPACA_SETTINGS,
               alpaca_provider_factory=None, historical_replay=None,
               unified_historical_market=None, historical_signal_store=None,
               research_model_service=None) -> FastAPI:
    """Build the ECL dashboard with Alpaca IEX as its only market-data provider."""
    if alpaca_engine is None:
        from live_app.alpaca_engine import AlpacaPredictionEngine
        alpaca_engine = AlpacaPredictionEngine()
    instance = alpaca_engine
    provider = None

    async def on_alpaca_bar(event):
        try:
            await asyncio.to_thread(instance.ingest_minute_event, event, now=event.received_at_utc)
        except Exception:
            # Mark only the affected five-minute bucket without persisting sensitive provider details.
            await asyncio.to_thread(instance.mark_minute_processing_failure,
                                    event.event_time_utc, event.received_at_utc)
            raise
        await asyncio.to_thread(instance.settle_due_predictions, now=event.received_at_utc)

    async def on_alpaca_connection_state(connected, at_utc, reason):
        await asyncio.to_thread(instance.set_stream_connection, connected, at_utc, reason)

    async def on_alpaca_processing_failure(event_time_utc, received_at_utc, safe_reason):
        affected_time = event_time_utc or received_at_utc
        await asyncio.to_thread(instance.mark_minute_processing_failure,
                                 affected_time, received_at_utc, f"invalid_stream_event_{safe_reason}")

    if alpaca_settings.display_authorized and alpaca_settings.credentials_configured:
        if alpaca_provider_factory is None:
            from live_app.alpaca_provider import AlpacaIEXWebSocketClient
            provider = AlpacaIEXWebSocketClient(alpaca_settings, on_bar=on_alpaca_bar,
                                                on_connection_state=on_alpaca_connection_state,
                                                on_processing_failure=on_alpaca_processing_failure)
        else:
            provider = alpaca_provider_factory(alpaca_settings, on_alpaca_bar,
                                               on_alpaca_connection_state,
                                               on_alpaca_processing_failure)
    current_refresher = None
    if alpaca_settings.display_authorized and alpaca_settings.credentials_configured:
        from live_app.alpaca_rest_history import AlpacaCurrentBarsRefresher
        current_refresher = AlpacaCurrentBarsRefresher(alpaca_settings)
    stream_task = None
    maintenance_task = None
    current_refresh_task = None
    research_preview_task = None
    stop_event: asyncio.Event | None = None
    research_preview_recorder = None

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

    async def current_market_refresh_loop(stop: asyncio.Event, app_ref):
        """Reconcile recent completed REST bars separately from live predictions."""
        while not stop.is_set():
            if current_refresher is not None:
                try:
                    known_candles, _, _ = combined_market_candles()
                    outcome = await asyncio.to_thread(current_refresher.refresh, known_candles,
                                                      now=datetime.now(UTC))
                    historical_market().refresh_current_cache()
                    app_ref.state.current_market_refresh = outcome
                except Exception as exc:
                    safe_status = current_refresher.status
                    safe_status["last_safe_error"] = safe_status.get("last_safe_error") or "refresh_failed"
                    safe_status["error_type"] = type(exc).__name__
                    app_ref.state.current_market_refresh = safe_status
                    _LOGGER.warning("Alpaca IEX REST catch-up failed error_type=%s safe_category=%s",
                                    type(exc).__name__, safe_status.get("last_safe_error"))
            try:
                await asyncio.wait_for(stop.wait(), timeout=ALPACA_REST_CATCHUP_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass

    async def research_preview_loop(stop: asyncio.Event, app_ref):
        """Record learned model outputs for recent finalized IEX candles separately.

        This loop reads only the app's verified historical/current candle
        timeline. It never writes the immutable production prediction ledger.
        """
        from live_app.research_preview_history import MODEL_IDS, preview_history

        recorder = app_ref.state.research_preview_recorder
        retry_after: dict[str, datetime] = {}
        last_outcome_sync: datetime | None = None
        while not stop.is_set():
            now = datetime.now(UTC)
            try:
                await asyncio.to_thread(historical_market().refresh_current_cache)
                candles, _, _ = await asyncio.to_thread(combined_market_candles)
                candidates = [item for item in _eligible_research_preview_candles(
                    candles, now=now, recorder=recorder)
                    if now >= retry_after.get(item[1].get("candle_time_utc"), datetime.min.replace(tzinfo=UTC))]
                for stamp, candle in candidates:
                    prefix = [row for row in candles if int(row["time"]) <= int(candle["time"])]
                    results = []
                    for model_id in MODEL_IDS:
                        try:
                            result = await asyncio.to_thread(research_service().predict_latest, prefix, model_id)
                            results.append(result)
                        except Exception as exc:
                            _LOGGER.error("Research preview inference failed model_id=%s error_type=%s",
                                          model_id, type(exc).__name__)
                            results.append({"model_id": model_id, "status": "model_error",
                                            "error_type": type(exc).__name__})
                    if any(result.get("status") == "model_error" for result in results):
                        retry_after[candle["candle_time_utc"]] = now + timedelta(minutes=1)
                        continue
                    retry_after.pop(candle["candle_time_utc"], None)
                    provider_status = getattr(provider, "status", {}) if provider is not None else {}
                    if callable(provider_status):
                        provider_status = provider_status()
                    record = {
                        "checked_at_utc": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
                        "candle_time_utc": candle.get("candle_time_utc"),
                        "reference_price": candle.get("close"),
                        "provider_state": provider_status.get("state") if isinstance(provider_status, dict) else None,
                        "authenticated": provider_status.get("authenticated") if isinstance(provider_status, dict) else None,
                        "subscribed": provider_status.get("subscribed") if isinstance(provider_status, dict) else None,
                        "source": "verified Alpaca IEX finalized 5-minute candle",
                        "models": results,
                    }
                    recorder.append(record)
                if (recorder.has_completed_records
                        and (candidates or last_outcome_sync is None
                             or now - last_outcome_sync >= timedelta(minutes=1))):
                    # Sync the separate preview ledger periodically so pending
                    # outcomes settle even when no browser is open.
                    await asyncio.to_thread(preview_history, 1, timeline=candles, now=now)
                    last_outcome_sync = now
            except Exception as exc:
                _LOGGER.warning("Research preview recording loop failed error_type=%s", type(exc).__name__)
            try:
                await asyncio.wait_for(stop.wait(), timeout=RESEARCH_PREVIEW_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal stream_task, maintenance_task, current_refresh_task, research_preview_task
        nonlocal research_preview_recorder, stop_event
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
        if run_poller and alpaca_settings.display_authorized and current_refresher is not None:
            if stop_event is None:
                stop_event = asyncio.Event()
            from live_app.research_preview_history import ResearchPreviewRecorder
            if research_preview_recorder is None:
                research_preview_recorder = ResearchPreviewRecorder()
            app.state.research_preview_recorder = research_preview_recorder
            if current_refresh_task is None or current_refresh_task.done():
                current_refresh_task = asyncio.create_task(current_market_refresh_loop(stop_event, app))
            app.state.current_refresh_task = current_refresh_task
            if research_preview_task is None or research_preview_task.done():
                research_preview_task = asyncio.create_task(research_preview_loop(stop_event, app))
            app.state.research_preview_task = research_preview_task
        try:
            yield
        finally:
            if stop_event is not None:
                stop_event.set()
            tasks = [task for task in (stream_task, maintenance_task, current_refresh_task,
                                       research_preview_task) if task is not None]
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

    app = FastAPI(title="ECL Paper Trading Indicator", version="2.0.0", lifespan=lifespan)
    app.state.engine = instance
    app.state.provider = provider
    app.state.mode = "alpaca_iex"
    app.state.historical_replay = historical_replay
    app.state.unified_historical_market = unified_historical_market
    app.state.historical_signal_store = historical_signal_store
    app.state.research_model_service = research_model_service
    app.state.current_market_refresh = current_refresher.status if current_refresher is not None else {
        "state": "credentials_missing", "last_safe_error": "credentials_missing"}
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def loopback_only(request: Request, call_next):
        # Reject accidental non-loopback binds instead of exposing local data.
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
        return (STATIC / "alpaca.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": "alpaca_iex", "provider": "Alpaca IEX",
                "model": "frozen Alpaca candidate", "sha256": instance.model_sha256}

    @app.get("/api/status")
    def status():
        now = datetime.now(UTC)
        if not alpaca_settings.display_authorized:
            return {
                "mode": "alpaca_iex", "provider": "Alpaca", "feed": "IEX",
                "display_authorized": False, "state": "authorization_required",
                "message": "Private display is disabled until ALPACA_IEX_DISPLAY_AUTHORIZED=true.",
                "model_ready": True, "model_sha256": instance.model_sha256,
                "missing_data_warning_count": 0,
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
        # The visible timeline may be fresher than the production prediction
        # ledger while a stream is warming up. Report the latest genuine
        # completed market candle without implying that it has a prediction.
        try:
            market = historical_market()
            refresh_current = getattr(market, "refresh_current_cache", None)
            if callable(refresh_current):
                refresh_current()
            timeline = market.candles()
            if timeline:
                historical_latest = timeline[-1]["candle_time_utc"]
                if latest_et is None or _epoch_utc(historical_latest) > _epoch_utc(
                        candles[-1]["candle_time_utc"] if candles else "1970-01-01T00:00:00Z"):
                    latest_et = _et_display(historical_latest)
        except (FileNotFoundError, ValueError, OSError):
            pass
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
            "current_market_refresh": app.state.current_market_refresh,
            "research_preview_recording_enabled": bool(research_preview_task is not None and not research_preview_task.done()),
            "research_preview_log_file": (research_preview_recorder.path.name
                                           if research_preview_recorder is not None else None),
            "pending_settlements": engine_status["pending_settlements"],
            "missing_data_warnings": warnings,
            "missing_data_warning_count": instance.store.unresolved_gap_count(),
            "late_updated_bar_count": engine_status["late_updated_bar_count"],
            "latest_late_updated_bar": engine_status["latest_late_updated_bar"],
            "open_stream_interruption_count": engine_status["open_stream_interruption_count"],
        }

    def require_alpaca_display():
        if not alpaca_settings.display_authorized:
            raise HTTPException(status_code=403, detail="Alpaca display authorization is disabled")

    def require_alpaca_mode():
        require_alpaca_display()

    @app.get("/api/data-quality/warnings")
    def data_quality_warnings():
        require_alpaca_mode()
        warnings = instance.store.unresolved_gaps_all()
        return {"count": instance.store.unresolved_gap_count(), "warnings": warnings}

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

    def combined_market_candles():
        """Return de-duplicated real historical and persisted live candles."""
        market = historical_market()
        refresh_current = getattr(market, "refresh_current_cache", None)
        if callable(refresh_current):
            refresh_current()
        history = market.candles()
        merged = {int(row["time"]): row for row in history}
        live_rows = instance.store.dashboard_candles(100000)
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
        return [merged[key] for key in sorted(merged)], history, live_count

    def research_service():
        if app.state.research_model_service is None:
            from live_app.research_models import ResearchModelService
            app.state.research_model_service = ResearchModelService(
                feature_order=list(instance.feature_order)
            )
        return app.state.research_model_service

    def saved_historical_signal_store():
        if app.state.historical_signal_store is not None:
            return app.state.historical_signal_store
        from live_app.historical_model_signals import DEFAULT_HISTORICAL_SIGNAL_DB, HistoricalSignalStore
        # A dashboard read must not create or generate a historical prediction ledger.
        if not DEFAULT_HISTORICAL_SIGNAL_DB.is_file():
            return None
        app.state.historical_signal_store = HistoricalSignalStore(DEFAULT_HISTORICAL_SIGNAL_DB)
        return app.state.historical_signal_store

    @app.get("/api/market-timeline")
    def market_timeline(tail: int | None = None):
        require_alpaca_mode()
        if tail is not None and not 1 <= tail <= 2000:
            raise HTTPException(status_code=400, detail="Timeline tail must be between 1 and 2000 candles")
        try:
            market = historical_market()
            refresh_current = getattr(market, "refresh_current_cache", None)
            if callable(refresh_current):
                refresh_current()
            coverage = dict(market.coverage)
            candles, history, live_count = combined_market_candles()
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail="Permitted historical market data could not be loaded") from exc
        coverage.update({
            "live_candle_count": live_count,
            "timeline_candle_count": len(candles),
            "historical_candle_count": len(history),
            "historical_backfill_status": coverage.get("backfill_manifest_status", "not_present"),
            "timeline_start_utc": candles[0]["candle_time_utc"] if candles else None,
            "timeline_end_utc": candles[-1]["candle_time_utc"] if candles else None,
        })
        # The browser loads the full permitted history once, then polls a small
        # tail so newly refreshed REST-cache bars reach the chart without
        # repeatedly transferring the entire historical dataset.
        visible_candles = candles[-tail:] if tail is not None else candles
        return {"source": "Alpaca IEX", "candles": visible_candles, "coverage": coverage}

    @app.get("/api/research-models")
    def research_models():
        require_alpaca_display()
        try:
            return research_service().catalog()
        except (OSError, KeyError, ValueError) as exc:
            _LOGGER.error("Research model evaluation summaries unavailable error_type=%s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Saved research model summaries are unavailable") from exc

    @app.get("/api/research-model-prediction")
    def research_model_prediction(model_id: str):
        require_alpaca_display()
        try:
            candles, _, _ = combined_market_candles()
            return research_service().predict_latest(candles, model_id)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail="Unknown research model selection") from exc
        except (ValueError, RuntimeError, OSError) as exc:
            _LOGGER.error("Research model preview failed model_id=%s error_type=%s", model_id, type(exc).__name__)
            raise HTTPException(status_code=503, detail="Research model preview is unavailable; production inference is unaffected") from exc

    @app.get("/api/research-model-preview-history")
    def research_model_preview_history(limit: int = 300):
        require_alpaca_mode()
        try:
            from live_app.research_preview_history import preview_history
            candles, _, _ = combined_market_candles()
            return preview_history(limit, timeline=candles, now=datetime.now(UTC))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research preview log limit is invalid") from exc

    @app.get("/api/research-model-preview-report.csv")
    def research_model_preview_report_csv():
        """Export only separately stored experimental previews and their verified outcomes."""
        require_alpaca_mode()
        try:
            from live_app.research_preview_history import all_preview_records
            candles, _, _ = combined_market_candles()
            records = all_preview_records(timeline=candles, now=datetime.now(UTC))
        except (OSError, ValueError) as exc:
            _LOGGER.error("Research preview CSV export failed error_type=%s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Separate research preview log is unavailable") from exc
        output = io.StringIO()
        writer = csv.writer(output)
        columns = [
            "candle_time_utc", "generated_at_utc", "model_id", "model_name", "prediction", "signal",
            "probability_down", "probability_neutral", "probability_up",
            "predicted_return_percentage_points", "reference_price", "horizon_minutes",
            "outcome_status", "endpoint_candle_time_utc", "actual_target", "actual_return_pct",
            "actual_close", "correct", "model_sha256", "source", "persisted_to_production_ledger",
        ]
        writer.writerow(columns)
        for row in records:
            probabilities = row.get("probabilities") or {}
            writer.writerow([
                row.get("candle_time_utc"), row.get("generated_at_utc"), row.get("model_id"),
                row.get("model_name"), row.get("prediction"), row.get("signal"),
                probabilities.get("DOWN"), probabilities.get("NEUTRAL"), probabilities.get("UP"),
                row.get("predicted_return_percentage_points"), row.get("reference_price"),
                row.get("horizon_minutes"), row.get("outcome_status"),
                row.get("endpoint_candle_time_utc"), row.get("actual_target"),
                row.get("actual_return_pct"), row.get("actual_close"), row.get("correct"),
                row.get("model_sha256"), row.get("source"), row.get("persisted_to_production_ledger", False),
            ])
        filename = "ecl_alpaca_iex_research_preview_report.csv"
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={filename}"})

    @app.get("/api/research-model-previews-at-candle")
    def research_model_previews_at_candle(candle_time_utc: str):
        require_alpaca_mode()
        try:
            stamp = datetime.fromisoformat(candle_time_utc.replace("Z", "+00:00"))
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                raise ValueError("timestamp must include timezone")
            stamp = stamp.astimezone(UTC)
            from live_app.research_preview_history import previews_for_candle
            candles, _, _ = combined_market_candles()
            return previews_for_candle(stamp, timeline=candles, now=datetime.now(UTC))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Candle timestamp must be timezone-aware ISO-8601") from exc

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
            historical_store = saved_historical_signal_store()
            historical = historical_store.at(stamp) if historical_store is not None else None
            if historical is not None:
                historical["source_type"] = "historical_backtest"
                historical["candle_time_et"] = _et_display(historical["candle_time_utc"])
                historical["prediction_created_et"] = _et_display(historical["prediction_created_utc"])
                historical["outcome_candle_time_et"] = _et_display(historical["outcome_candle_time_utc"])
                historical["settlement_status"] = historical.pop("outcome_status")
                historical.pop("input_features_json", None)
                if historical["settlement_status"] != "SETTLED":
                    for key in ("actual_target", "actual_return_pct", "actual_close", "correct"):
                        historical.pop(key, None)
                return {"status": "settled" if historical["settlement_status"] == "SETTLED" else "outcome_unavailable",
                        "prediction": historical}
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

    @app.get("/api/historical-model-signals")
    def historical_model_signals(limit: int = 100000):
        require_alpaca_mode()
        store = saved_historical_signal_store()
        if store is None:
            return {"source": "historical_backtest", "signals": []}
        rows = store.rows(limit)
        for row in rows:
            row["candle_time_epoch"] = _epoch_utc(row["candle_time_utc"])
            row["candle_time_et"] = _et_display(row["candle_time_utc"])
            row["prediction_created_et"] = _et_display(row["prediction_created_utc"])
            row["outcome_candle_time_et"] = _et_display(row["outcome_candle_time_utc"])
            row["source_type"] = row["source"]
            row["settlement_status"] = row["outcome_status"]
            row.pop("input_features_json", None)
        return {"source": "historical_backtest", "signals": rows}

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
        require_alpaca_display()
        return [{"time": _epoch_utc(row["candle_time_utc"]), "open": row["open"],
                 "high": row["high"], "low": row["low"], "close": row["close"],
                 "volume": row.get("volume"), "source": "live", "feed": row.get("feed", "IEX"),
                 "live": True}
                for row in instance.store.dashboard_candles(limit)]

    @app.get("/api/predictions")
    def predictions(limit: int = 200):
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

    @app.get("/api/report")
    def report():
        require_alpaca_display()
        return instance.store.performance_report()

    @app.get("/api/report.csv")
    def report_csv():
        require_alpaca_display()
        output = io.StringIO()
        writer = csv.writer(output)
        columns = ["candle_time_utc", "prediction_created_utc", "prediction", "probability_down",
                   "probability_neutral", "probability_up", "confidence", "reference_close",
                   "horizon_minutes", "outcome_candle_time_utc", "actual_target", "actual_return_pct",
                   "correct", "model_sha256"]
        writer.writerow(columns)
        for row in instance.store.prediction_history(100000):
            writer.writerow([row.get(column) for column in columns])
        filename = "ecl_alpaca_iex_prediction_report.csv"
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={filename}"})

    return app


app = create_app()
