"""Read-only, pre-test Alpaca IEX historical replay using the frozen candidate."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from live_app.calendar import NY, session_for_date
from live_app.features import calculate_b_features, ready_feature_row
from live_app.model_loader import EXPECTED_CLASSES, EXPECTED_HORIZON_MINUTES, load_alpaca_candidate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDLE_PATH = ROOT / "data" / "alpaca_replay_v1" / "ecl_iex_5min_validation_20260806_20260924.csv"
FIRST_ALLOWED_DATE = date(2026, 8, 6)
LAST_ALLOWED_DATE = date(2026, 9, 24)
HORIZON = EXPECTED_HORIZON_MINUTES


def _is_regular_session_candle(timestamp: pd.Timestamp, bounds: tuple[datetime, datetime] | None) -> bool:
    """Validate a five-minute candle by its open and exclusive close boundary.

    NYSE sessions use [open, close) for candle-open timestamps. A candle is
    valid when it opens at/after session open and its five-minute interval ends
    at or before the session close. This admits 15:55 ET on a normal day and
    12:55 ET on a 13:00 early close, while rejecting extended-hours bars.
    """
    if bounds is None:
        return False
    local = timestamp.to_pydatetime()
    if local.tzinfo is None:
        local = local.replace(tzinfo=NY)
    else:
        local = local.astimezone(NY)
    session_open, session_close = bounds
    candle_close = local + timedelta(minutes=5)
    return session_open <= local and candle_close <= session_close


def _parse_utc(value: str | datetime) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Candle timestamp must be timezone-aware UTC")
    return stamp.astimezone(timezone.utc)


class HistoricalReplay:
    """Replay historical candles without writing to live stores or databases."""

    def __init__(self, *, candidate: Any | None = None, candle_path: Path | str = DEFAULT_CANDLE_PATH):
        self.candidate = candidate or load_alpaca_candidate()
        self.feature_order = list(self.candidate.specification["features_exact_order"])
        self.classes = list(self.candidate.specification["class_order"])
        if len(self.feature_order) != 24 or self.classes != EXPECTED_CLASSES:
            raise ValueError("Frozen replay model contract is invalid")
        if list(getattr(self.candidate.model, "feature_names_in_", [])) != self.feature_order:
            raise ValueError("Frozen replay model feature order mismatch")
        if [str(x) for x in getattr(self.candidate.model, "classes_", [])] != EXPECTED_CLASSES:
            raise ValueError("Frozen replay model class order mismatch")
        self.candle_path = Path(candle_path)
        self._bars = self._load_allowed_bars(self.candle_path)

    @staticmethod
    def _load_allowed_bars(path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError("Permitted historical replay candle cache is unavailable")
        data = pd.read_csv(path)
        required = {"timestamp_utc", "datetime", "open", "high", "low", "close"}
        if not required.issubset(data.columns):
            raise ValueError("Historical replay candle cache has an invalid schema")
        utc = pd.to_datetime(data["timestamp_utc"], utc=True, errors="raise")
        et_dates = utc.dt.tz_convert("America/New_York").dt.date
        if ((et_dates < FIRST_ALLOWED_DATE) | (et_dates > LAST_ALLOWED_DATE)).any():
            raise ValueError("Historical replay cache includes observations outside the permitted pre-test period")
        if utc.duplicated().any() or not utc.is_monotonic_increasing:
            raise ValueError("Historical replay timestamps must be unique and chronological")
        expected_local = utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
        local = pd.to_datetime(data["datetime"], errors="raise")
        if not local.equals(expected_local):
            raise ValueError("Historical replay UTC and New York timestamps disagree")
        values = data[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="raise")
        arr = values.to_numpy(dtype=float)
        if not np.isfinite(arr).all() or (arr <= 0).any():
            raise ValueError("Historical replay OHLC contains invalid prices")
        if (values.high < values[["open", "close", "low"]].max(axis=1)).any() or (values.low > values[["open", "close", "high"]].min(axis=1)).any():
            raise ValueError("Historical replay contains invalid OHLC candles")
        if not local.dt.minute.mod(5).eq(0).all() or not local.dt.second.eq(0).all():
            raise ValueError("Historical replay bars are not aligned to five-minute boundaries")
        sessions = {day: session_for_date(day) for day in set(local.dt.date)}
        in_session = [
            _is_regular_session_candle(timestamp, sessions[timestamp.date()])
            for timestamp in local
        ]
        frame = pd.DataFrame({"timestamp_utc": utc, "datetime": local, **values.to_dict(orient="list")})
        # Alpaca's historical 5Min response includes one 09:25 ET extended-
        # hours candle. Keep the regular-session rule strict and exclude bars
        # whose five-minute interval is not wholly inside the official session.
        excluded = int((~pd.Series(in_session, dtype=bool)).sum())
        frame = frame.loc[in_session].reset_index(drop=True)
        frame.attrs["excluded_out_of_session_rows"] = excluded
        if frame.empty:
            raise ValueError("Historical replay cache has no valid regular-session candles")
        return frame

    def dates(self) -> list[str]:
        return sorted({stamp.date().isoformat() for stamp in self._bars["datetime"]})

    def candles(self, day: str) -> list[dict[str, Any]]:
        parsed = date.fromisoformat(day)
        if parsed < FIRST_ALLOWED_DATE or parsed > LAST_ALLOWED_DATE:
            raise ValueError("Date is outside the permitted pre-test replay period")
        subset = self._bars.loc[self._bars.datetime.dt.date.eq(parsed)]
        return [{"time": int(row.timestamp_utc.timestamp()), "candle_time_utc": row.timestamp_utc.isoformat().replace("+00:00", "Z"),
                 "candle_time_et": row.datetime.isoformat(sep=" "), "open": float(row.open), "high": float(row.high),
                 "low": float(row.low), "close": float(row.close)} for row in subset.itertuples(index=False)]

    def predict(self, day: str, candle_time: str) -> dict[str, Any]:
        parsed_day = date.fromisoformat(day)
        if parsed_day < FIRST_ALLOWED_DATE or parsed_day > LAST_ALLOWED_DATE:
            raise ValueError("Date is outside the permitted pre-test replay period")
        selected_utc = _parse_utc(candle_time)
        selected_et = selected_utc.astimezone(NY)
        if selected_et.date() != parsed_day:
            raise ValueError("Selected candle does not belong to the requested New York date")
        rows = self._bars.loc[self._bars.timestamp_utc.le(pd.Timestamp(selected_utc))]
        rows = rows.loc[rows.datetime.dt.date.eq(parsed_day)]
        match = rows.loc[rows.timestamp_utc.eq(pd.Timestamp(selected_utc))]
        if len(match) != 1:
            raise ValueError("Selected completed candle is unavailable")
        session = session_for_date(parsed_day)
        if not session:
            raise ValueError("Selected date is not an NYSE session")

        # Crucially, only the prefix through the selected completed candle enters feature generation.
        ohlc = rows[["datetime", "open", "high", "low", "close"]].copy()
        feature_frame = calculate_b_features(ohlc, self.feature_order)
        if list(feature_frame.columns) != ["datetime", *self.feature_order]:
            raise ValueError("Historical feature names/order differ from the frozen contract")
        candle_local = pd.Timestamp(selected_et.replace(tzinfo=None))
        feature_row, missing = ready_feature_row(feature_frame, candle_local)
        if feature_row is None:
            continuous_candles = self._continuous_prefix_count(rows, candle_local)
            segment_start = selected_et.replace(tzinfo=None) - timedelta(minutes=5 * (continuous_candles - 1))
            reset_reason = "missing_5_minute_candle" if segment_start > session[0].replace(tzinfo=None) else "trading_session_boundary"
            return {"status": "unavailable", "reason": "insufficient_warmup", "missing_features": missing,
                    "feature_context": {
                        "continuous_candles": continuous_candles,
                        "segment_start_et": segment_start.replace(tzinfo=NY).isoformat(timespec="seconds"),
                        "reset_reason": reset_reason,
                        "warmup_rule": "The original training pipeline resets rolling features at each trading session and every missing five-minute candle.",
                    },
                    "candle_time_utc": selected_utc.isoformat().replace("+00:00", "Z"), "source": "Alpaca IEX", "historical": True}
        x = feature_row[self.feature_order].to_frame().T
        if list(x.columns) != self.feature_order or not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError("Historical replay feature row is invalid")
        if "volume" in self.feature_order or "trade_count" in self.feature_order:
            raise ValueError("Volume fields must not be model inputs")
        probabilities = np.asarray(self.candidate.model.predict_proba(x), dtype=float)
        if probabilities.shape != (1, 3) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
            raise ValueError("Frozen model returned invalid class probabilities")
        p = probabilities[0]
        prediction = EXPECTED_CLASSES[int(np.argmax(p))]
        chosen = match.iloc[0]
        outcome = self._outcome(selected_utc, float(chosen.close), prediction)
        return {"status": "ok", "historical": True, "retrospective": True, "source": "Alpaca IEX",
                "candle_time_utc": selected_utc.isoformat().replace("+00:00", "Z"),
                "candle_time_et": selected_et.isoformat(timespec="seconds"),
                "prediction": prediction, "probabilities": dict(zip(EXPECTED_CLASSES, map(float, p))),
                "confidence": float(p.max()), "reference_close": float(chosen.close),
                "horizon_minutes": HORIZON, "model_sha256": self.candidate.sha256,
                "outcome": outcome}

    @staticmethod
    def _continuous_prefix_count(rows: pd.DataFrame, selected_local: pd.Timestamp) -> int:
        """Count selected candle and immediately preceding continuous bars only."""
        times = rows.loc[rows.datetime.le(selected_local), "datetime"].tolist()
        if not times:
            return 0
        count = 1
        reverse_times = list(reversed(times))
        for later, earlier in zip(reverse_times, reverse_times[1:]):
            if later - earlier != pd.Timedelta(minutes=5):
                break
            count += 1
        return count

    def _outcome(self, decision: datetime, reference_close: float, prediction: str) -> dict[str, Any]:
        endpoint = decision + timedelta(minutes=HORIZON)
        decision_et, endpoint_et = decision.astimezone(NY), endpoint.astimezone(NY)
        target_date = endpoint_et.date()
        session = session_for_date(decision_et.date())
        if (not session or target_date != decision_et.date() or decision_et < session[0]
                or decision_et + timedelta(minutes=5) > session[1]
                or endpoint_et + timedelta(minutes=5) > session[1]):
            return {"status": "unavailable", "reason": "session_boundary", "target_time_utc": endpoint.isoformat().replace("+00:00", "Z")}
        expected = [decision + timedelta(minutes=offset) for offset in (0, 5, 10, 15)]
        indexed = self._bars.set_index("timestamp_utc")
        if any(pd.Timestamp(stamp) not in indexed.index for stamp in expected):
            return {"status": "unavailable", "reason": "missing_or_discontinuous_candles", "target_time_utc": endpoint.isoformat().replace("+00:00", "Z")}
        target_close = float(indexed.loc[pd.Timestamp(endpoint), "close"])
        ret = target_close / reference_close - 1.0
        actual = "UP" if ret > 0.001 else "DOWN" if ret < -0.001 else "NEUTRAL"
        return {"status": "available", "target_time_utc": endpoint.isoformat().replace("+00:00", "Z"),
                "target_time_et": endpoint_et.isoformat(timespec="seconds"), "target_close": target_close,
                "return": ret, "actual": actual, "correct": prediction == actual}
