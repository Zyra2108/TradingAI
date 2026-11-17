"""Alpaca-only prediction and outcome engine for the frozen ECL candidate.

The engine does not create provider connections or start a server.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable

import numpy as np
import pandas as pd

from config import ALPACA_DB_PATH
from live_app.alpaca_aggregation import AggregatedFiveMinuteCandle, AlpacaFiveMinuteAggregator
from live_app.alpaca_storage import AlpacaBarStore
from live_app.calendar import NY, session_for_date
from live_app.features import calculate_b_features, ready_feature_row
from live_app.model_loader import EXPECTED_CLASSES, EXPECTED_HORIZON_MINUTES, EXPECTED_POLICY, LoadedCandidate, load_alpaca_candidate

UTC = timezone.utc
HORIZON_MINUTES = EXPECTED_HORIZON_MINUTES
FINALIZATION_DELAY_SECONDS = 120


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _from_utc_text(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


class AlpacaPredictionEngine:
    """Consumes finalized candles, predicts once, then settles complete paths."""

    def __init__(self, *, model_loader: Callable[[], LoadedCandidate] | None = None,
                 candidate: LoadedCandidate | None = None, store: AlpacaBarStore | None = None,
                 aggregator: AlpacaFiveMinuteAggregator | None = None,
                 db_path: Path | str | None = None,
                 now_fn: Callable[[], datetime] | None = None,
                 finalization_delay_seconds: int = FINALIZATION_DELAY_SECONDS):
        if candidate is None:
            candidate = (model_loader or load_alpaca_candidate)()
        self.candidate = candidate
        self.model = candidate.model
        self.specification = candidate.specification
        self.model_sha256 = candidate.sha256
        self.feature_order = list(self.specification["features_exact_order"])
        self.classes = list(self.specification["class_order"])
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self.started_at_utc = _utc(self.now_fn(), "startup time")
        self.finalization_delay = timedelta(seconds=finalization_delay_seconds)
        self.store = store or AlpacaBarStore(db_path if db_path is not None else ALPACA_DB_PATH)
        # Close an interruption left open by a prior process at this startup
        # boundary. The recorded interval remains available to reject any
        # bucket that overlapped the outage.
        self.store.close_open_stream_interruptions(self.started_at_utc)
        self.aggregator = aggregator or AlpacaFiveMinuteAggregator(
            self.store, finalization_buffer_seconds=finalization_delay_seconds, clock=self.now_fn
        )
        if Path(self.aggregator.store.path).resolve() != Path(self.store.path).resolve():
            raise ValueError("Alpaca aggregator and prediction engine must share the same database")
        self._validate_frozen_contract()
        # The provider callback and the periodic recovery sweep both run in
        # worker threads. Serialize their aggregate->predict sequence so a
        # due bucket can only be finalized/considered for prediction once at
        # a time within this process. SQLite primary keys remain the durable
        # idempotency guard across restarts and competing processes.
        self._aggregation_lock = RLock()
        self._feature_ready = False
        self._last_safe_error: str | None = None
        self._last_aggregation_status: str | None = None
        self._latest_prediction_attempt: dict[str, Any] | None = None
        self._latest_prediction_summary: dict[str, Any] | None = self._summarize(self.store.latest_prediction())

    def _validate_frozen_contract(self) -> None:
        if len(self.feature_order) != 24 or len(set(self.feature_order)) != 24:
            raise ValueError("Frozen model must specify exactly 24 unique features")
        if self.classes != EXPECTED_CLASSES:
            raise ValueError("Frozen model class order is invalid")
        if self.specification.get("prediction_policy") != EXPECTED_POLICY:
            raise ValueError("Frozen model prediction policy is invalid")
        target = self.specification.get("target", {})
        if target.get("horizon_minutes") != HORIZON_MINUTES:
            raise ValueError("Frozen model horizon is invalid")
        if (target.get("up") != "future return > +0.10%"
                or target.get("down") != "future return < -0.10%"
                or target.get("neutral") != "otherwise"):
            raise ValueError("Frozen model target thresholds are invalid")
        if list(getattr(self.model, "feature_names_in_", [])) != self.feature_order:
            raise ValueError("Model feature order does not match frozen specification")
        if [str(x) for x in getattr(self.model, "classes_", [])] != EXPECTED_CLASSES:
            raise ValueError("Model class order does not match frozen specification")

    def _assert_inference_contract(self, frame: pd.DataFrame) -> None:
        expected_columns = ["datetime", *self.feature_order]
        if list(frame.columns) != expected_columns:
            raise ValueError("Feature builder output names/order do not match frozen contract")
        if list(getattr(self.model, "feature_names_in_", [])) != self.feature_order:
            raise ValueError("Model feature order changed before inference")
        if [str(x) for x in getattr(self.model, "classes_", [])] != EXPECTED_CLASSES:
            raise ValueError("Model class order changed before inference")
        if "volume" in self.feature_order or "trade_count" in self.feature_order:
            raise ValueError("Volume and trade count are not model features")

    @staticmethod
    def _summarize(prediction: dict[str, Any] | None) -> dict[str, Any] | None:
        if prediction is None:
            return None
        return {
            "candle_time_utc": prediction["candle_time_utc"],
            "prediction_created_utc": prediction["prediction_created_utc"],
            "prediction": prediction["prediction"],
            "confidence": float(prediction["confidence"]),
        }

    def status(self) -> dict[str, Any]:
        return {
            "provider": "Alpaca",
            "feed": "IEX",
            "model_ready": True,
            "model_sha256": self.model_sha256,
            "feature_ready": self._feature_ready,
            "last_aggregation_status": self._last_aggregation_status,
            "latest_prediction_attempt": (dict(self._latest_prediction_attempt)
                                           if self._latest_prediction_attempt is not None else None),
            "latest_prediction": self._latest_prediction_summary,
            "pending_settlements": len(self.store.pending_predictions()),
            "last_safe_error": self._last_safe_error,
            "late_updated_bar_count": self.store.late_update_count(),
            "latest_late_updated_bar": self.store.latest_late_update(),
            "open_stream_interruption_count": self.store.open_stream_interruption_count(),
        }

    def set_stream_connection(self, connected: bool, at_utc: datetime, reason: str) -> None:
        """Persist known stream interruption boundaries without touching candles."""
        at = _utc(at_utc, "stream status timestamp")
        if connected:
            self.store.end_stream_interruption(at)
        else:
            self.store.begin_stream_interruption(at, reason or "websocket_disconnected")

    def mark_minute_processing_failure(self, event_time_utc: datetime,
                                       detected_at_utc: datetime | None = None,
                                       reason: str = "minute_processing_failure") -> None:
        self.aggregator.mark_processing_failure(
            event_time_utc, detected_at_utc=detected_at_utc or self.now_fn(), reason=reason
        )

    def ingest_minute_event(self, event: Any, *, now: datetime | None = None) -> dict[str, Any]:
        with self._aggregation_lock:
            result = self.aggregator.ingest(event, now=now)
            self._last_aggregation_status = result.status
            prediction_result = None
            if result.status == "finalized" and result.inserted and result.candle is not None:
                prediction_result = self.predict_finalized_candle(result.candle, now=now)
            return {
                "aggregation_status": result.status,
                "minute_inserted": result.inserted,
                "minute_conflict": result.conflict,
                "prediction": prediction_result,
            }

    def recover_aggregations(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        """Recover completed buckets, while the startup guard blocks old predictions."""
        output = []
        with self._aggregation_lock:
            for result in self.aggregator.recover_due_buckets(now=now):
                self._last_aggregation_status = result.status
                prediction_result = None
                if result.status == "finalized" and result.inserted and result.candle is not None:
                    prediction_result = self.predict_finalized_candle(result.candle, now=now)
                output.append({"aggregation_status": result.status, "prediction": prediction_result})
        return output

    def _record_prediction_attempt(self, candle_time: datetime, status: str, *,
                                   reason: str | None = None,
                                   missing_features: list[str] | None = None,
                                   error_type: str | None = None) -> None:
        self._latest_prediction_attempt = {
            "status": status,
            "candle_time_utc": _utc(candle_time, "candle_time_utc").isoformat(timespec="seconds").replace("+00:00", "Z"),
            "reason": reason,
            "missing_features": list(missing_features or []),
            "error_type": error_type,
        }

    def predict_finalized_candle(self, candle: AggregatedFiveMinuteCandle,
                                 *, now: datetime | None = None) -> dict[str, Any]:
        """Run one prediction attempt and keep its diagnostic separate from aggregation status."""
        candle_time = _utc(candle.candle_time_utc, "candle_time_utc")
        try:
            result = self._predict_finalized_candle(candle, now=now)
        except Exception as exc:
            self._record_prediction_attempt(
                candle_time, "prediction_error",
                reason="prediction_generation_or_persistence_error",
                error_type=type(exc).__name__,
            )
            raise
        status = result.get("status", "prediction_not_generated")
        if status == "features_not_ready":
            self._record_prediction_attempt(
                candle_time, "feature_warmup_incomplete",
                reason="consecutive_candle_warmup_insufficient",
                missing_features=result.get("missing", []),
            )
        elif status == "predicted":
            self._record_prediction_attempt(candle_time, "prediction_saved")
        elif status == "already_predicted":
            self._record_prediction_attempt(candle_time, "prediction_already_saved")
        elif status == "candle_finalization_buffer_not_met":
            self._record_prediction_attempt(candle_time, "candle_awaiting_finalization",
                                            reason="finalization_buffer_not_met")
        else:
            self._record_prediction_attempt(candle_time, status, reason=status)
        return result

    def _predict_finalized_candle(self, candle: AggregatedFiveMinuteCandle,
                                  *, now: datetime | None = None) -> dict[str, Any]:
        candle_time = _utc(candle.candle_time_utc, "candle_time_utc")
        if self.store.prediction_at(candle_time) is not None:
            return {"status": "already_predicted", "candle_time_utc": candle_time.isoformat()}
        stored_candle = self.store.five_minute_candle_at(candle_time)
        if stored_candle is None:
            return {"status": "finalized_candle_not_persisted", "candle_time_utc": candle_time.isoformat()}
        finalized_at = _from_utc_text(stored_candle["finalized_at_utc"])
        required_ready_at = candle_time + timedelta(minutes=5) + self.finalization_delay
        if finalized_at < required_ready_at:
            return {"status": "candle_finalization_buffer_not_met", "candle_time_utc": candle_time.isoformat()}
        # A candle already closed when this process started is historical recovery.
        if candle_time + timedelta(minutes=5) <= self.started_at_utc:
            return {"status": "retroactive_prediction_skipped", "candle_time_utc": candle_time.isoformat()}

        local_time = candle_time.astimezone(NY)
        session = session_for_date(local_time.date())
        if not session:
            return {"status": "outside_session", "candle_time_utc": candle_time.isoformat()}
        endpoint_time = candle_time + timedelta(minutes=HORIZON_MINUTES)
        endpoint_local = endpoint_time.astimezone(NY)
        if (local_time.date() != endpoint_local.date()
                or local_time < session[0]
                or local_time + timedelta(minutes=5) > session[1]
                or endpoint_local + timedelta(minutes=5) > session[1]
                or (local_time - session[0]).total_seconds() % 300 != 0):
            return {"status": "horizon_crosses_session", "candle_time_utc": candle_time.isoformat()}

        session_open_utc = session[0].astimezone(UTC)
        session_close_utc = session[1].astimezone(UTC)
        candles = self.store.five_minute_candles_for_session(session_open_utc, session_close_utc)
        if not candles:
            self._feature_ready = False
            return {"status": "features_not_ready", "missing": ["session candles unavailable"]}
        utc_times = pd.to_datetime([row["candle_time_utc"] for row in candles], utc=True)
        local_naive = utc_times.tz_convert(NY).tz_localize(None)
        ohlc = pd.DataFrame({
            "datetime": local_naive,
            "open": [row["open"] for row in candles],
            "high": [row["high"] for row in candles],
            "low": [row["low"] for row in candles],
            "close": [row["close"] for row in candles],
        })
        feature_frame = calculate_b_features(ohlc, self.feature_order)
        candle_local_naive = pd.Timestamp(local_time.replace(tzinfo=None))
        feature_row, missing = ready_feature_row(feature_frame, candle_local_naive)
        self._feature_ready = not missing
        if missing:
            self._last_safe_error = "feature_row_unavailable"
            return {"status": "features_not_ready", "missing": missing}

        self._assert_inference_contract(feature_frame)
        x = feature_row[self.feature_order].to_frame().T
        if list(x.columns) != self.feature_order:
            raise ValueError("Inference feature order does not match frozen specification")
        if any(not np.isfinite(float(value)) for value in x.iloc[0].tolist()):
            self._last_safe_error = "non_finite_features"
            raise ValueError("Feature row contains non-finite values")
        raw_probabilities = np.asarray(self.model.predict_proba(x), dtype=float)
        if raw_probabilities.shape != (1, 3):
            self._last_safe_error = "invalid_probability_shape"
            raise ValueError("Frozen model returned an invalid probability shape")
        probabilities = raw_probabilities[0]
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-8)):
            self._last_safe_error = "invalid_probabilities"
            raise ValueError("Frozen model probabilities must be finite, nonnegative, and sum to one")
        prediction = EXPECTED_CLASSES[int(np.argmax(probabilities))]
        current = _utc(now if now is not None else self.now_fn(), "prediction time")
        delay = max(0.0, (finalized_at - (candle_time + timedelta(minutes=5))).total_seconds())
        inserted = self.store.insert_prediction({
            "candle_time_utc": candle_time,
            "prediction_created_utc": current,
            "prediction": prediction,
            "probability_down": float(probabilities[0]),
            "probability_neutral": float(probabilities[1]),
            "probability_up": float(probabilities[2]),
            "confidence": float(probabilities.max()),
            "reference_close": float(stored_candle["close"]),
            "horizon_minutes": HORIZON_MINUTES,
            "outcome_candle_time_utc": endpoint_time,
            "model_sha256": self.model_sha256,
            "input_features": {name: float(feature_row[name]) for name in self.feature_order},
            "provider_delay_seconds": delay,
        })
        self._last_safe_error = None
        if not inserted:
            return {"status": "already_predicted", "candle_time_utc": candle_time.isoformat()}
        self._latest_prediction_summary = self._summarize(self.store.prediction_at(candle_time))
        return {
            "status": "predicted", "candle_time_utc": candle_time.isoformat(),
            "prediction": prediction,
            "probabilities": {name: float(value) for name, value in zip(EXPECTED_CLASSES, probabilities)},
            "confidence": float(probabilities.max()),
        }

    @staticmethod
    def _valid_settlement_path(decision_time: datetime, endpoint_time: datetime,
                               session: tuple[datetime, datetime]) -> bool:
        decision = _utc(decision_time, "decision time")
        endpoint = _utc(endpoint_time, "endpoint time")
        if endpoint != decision + timedelta(minutes=HORIZON_MINUTES):
            return False
        decision_local, endpoint_local = decision.astimezone(NY), endpoint.astimezone(NY)
        if decision_local.date() != endpoint_local.date():
            return False
        opened, closed = session
        if (decision_local < opened or decision_local + timedelta(minutes=5) > closed
                or endpoint_local + timedelta(minutes=5) > closed
                or (decision_local - opened).total_seconds() % 300 != 0):
            return False
        return True

    def settle_due_predictions(self, *, now: datetime | None = None) -> int:
        current = _utc(now if now is not None else self.now_fn(), "settlement time")
        settled = 0
        for prediction in self.store.pending_predictions():
            decision = _from_utc_text(prediction["candle_time_utc"])
            endpoint = _from_utc_text(prediction["outcome_candle_time_utc"])
            session = session_for_date(decision.astimezone(NY).date())
            if not session or not self._valid_settlement_path(decision, endpoint, session):
                continue
            endpoint_close_at = endpoint + timedelta(minutes=5) + self.finalization_delay
            if current < endpoint_close_at:
                continue
            expected = [decision + timedelta(minutes=step) for step in (0, 5, 10, 15)]
            path = [self.store.five_minute_candle_at(stamp) for stamp in expected]
            if any(candle is None for candle in path):
                continue
            endpoint_finalized_at = _from_utc_text(path[-1]["finalized_at_utc"])
            if endpoint_finalized_at < endpoint_close_at or endpoint_finalized_at > current:
                continue
            # Primary keys make duplicate timestamps impossible; requiring exact
            # lookups at all four expected timestamps prevents gaps being hidden.
            target_close = float(path[-1]["close"])
            reference_close = float(prediction["reference_close"])
            if not np.isfinite([target_close, reference_close]).all() or reference_close <= 0:
                continue
            return_pct = (target_close / reference_close - 1.0) * 100.0
            actual = "UP" if return_pct > 0.10 else "DOWN" if return_pct < -0.10 else "NEUTRAL"
            settled += int(self.store.insert_outcome({
                "prediction_candle_time_utc": decision,
                "observed_utc": current,
                "endpoint_candle_time_utc": endpoint,
                "actual_target": actual,
                "actual_return_pct": return_pct,
                "correct": prediction["prediction"] == actual,
            }))
        return settled
