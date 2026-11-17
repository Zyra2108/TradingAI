"""Causal feature/target construction from the isolated fetched IEX minute file."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW_PATH = OUT / "raw_ecl_iex_1min_train_validation.csv"
RAW_5M_PATH = OUT / "raw_ecl_iex_5min_train_validation.csv"
SPLIT_DATA = ROOT / "data" / "alpaca_rebuild_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NY = "America/New_York"
TARGET_HORIZON_MINUTES = 10
TARGET_THRESHOLD_PCT = 0.10

ONE_MINUTE_FEATURES = [
    "close", "return_1_pct", "return_3_pct", "return_5_pct", "return_10_pct",
    "realized_vol_3", "realized_vol_5", "realized_vol_10",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "volume_log1p", "volume_ratio_10", "volume_zscore_10",
    "sma_ratio_3", "sma_ratio_5", "sma_ratio_10",
    "ema_ratio_3", "ema_ratio_5", "ema_ratio_10", "rsi_7",
    "trend_agreement_3_10", "momentum_acceleration_3_10",
    "minutes_since_open", "minutes_until_close", "session_progress", "day_of_week", "time_sin", "time_cos",
]
FEATURE_METADATA = {"datetime", "timestamp_utc", "label", "target_end_utc", "future_return_pct",
                    "reference_close", "volume", "trade_count", "vwap", "split", "session_date",
                    "regime", "regime_volatility"}


def split_date_ranges() -> dict[str, tuple[date, date]]:
    """Derive permitted date bounds by opening timestamps from train/validation only."""
    result = {}
    for name in ("train", "validation"):
        stamps = pd.to_datetime(pd.read_csv(SPLIT_DATA / f"{name}.csv", usecols=["datetime"])["datetime"], errors="raise")
        result[name] = (stamps.min().date(), stamps.max().date())
    return result


def split_date_sets() -> dict[str, set[date]]:
    """Exact date membership from the permitted train and validation files only."""
    result = {}
    for name in ("train", "validation"):
        stamps = pd.to_datetime(pd.read_csv(SPLIT_DATA / f"{name}.csv", usecols=["datetime"])["datetime"], errors="raise")
        result[name] = set(stamps.dt.date)
    if result["train"] & result["validation"]:
        raise ValueError("Existing train and validation date sets overlap")
    return result


def session_map(start: date, end: date) -> dict[date, tuple[datetime, datetime]]:
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    mapping = {}
    for idx, row in schedule.iterrows():
        day = pd.Timestamp(idx).date()
        mapping[day] = (row.market_open.tz_convert(NY).to_pydatetime(),
                        row.market_close.tz_convert(NY).to_pydatetime())
    return mapping


def clean_and_audit(raw_path: Path = RAW_PATH) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return regular-session valid 1m bars, the minute gap ledger, and QA summary."""
    raw = pd.read_csv(raw_path)
    required = {"timestamp_utc", "open", "high", "low", "close", "volume", "trade_count"}
    if not required.issubset(raw.columns):
        raise ValueError("Fetched Alpaca CSV is missing expected columns")
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True, errors="raise")
    raw["datetime"] = raw.timestamp_utc.dt.tz_convert(NY).dt.tz_localize(None)
    for name in ("open", "high", "low", "close", "volume", "trade_count"):
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    raw = raw.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    duplicate_mask = raw.timestamp_utc.duplicated(keep="first")
    duplicates = int(duplicate_mask.sum())
    dedup = raw.loc[~duplicate_mask].copy()
    ohlc = dedup[["open", "high", "low", "close"]].to_numpy(float)
    finite = np.isfinite(ohlc).all(axis=1)
    positive = (ohlc > 0).all(axis=1)
    valid_range = ((dedup.high >= dedup[["open", "low", "close"]].max(axis=1)) &
                   (dedup.low <= dedup[["open", "high", "close"]].min(axis=1))).to_numpy()
    valid_ohlc = finite & positive & valid_range
    aligned = dedup.timestamp_utc.dt.second.eq(0).to_numpy() & dedup.timestamp_utc.dt.microsecond.eq(0).to_numpy()

    first_day = dedup.datetime.dt.date.min()
    last_day = dedup.datetime.dt.date.max()
    sessions = session_map(first_day, last_day)
    opens = dedup.datetime.map(lambda d: sessions.get(d.date(), (None, None))[0])
    closes = dedup.datetime.map(lambda d: sessions.get(d.date(), (None, None))[1])
    minute_end = dedup.datetime + pd.Timedelta(minutes=1)
    regular = np.asarray([(op is not None and op.replace(tzinfo=None) <= stamp.to_pydatetime() and
                           stamp.to_pydatetime() < close.replace(tzinfo=None) and
                           end.to_pydatetime() <= close.replace(tzinfo=None))
                          for stamp, end, op, close in zip(dedup.datetime, minute_end, opens, closes)], dtype=bool)
    outside_session = int((~regular).sum())
    invalid_rows = int((~valid_ohlc).sum())
    eligible = dedup.loc[valid_ohlc & aligned & regular].copy().reset_index(drop=True)
    valid_times = set(eligible.datetime)
    gaps = []
    expected_total = 0
    for day, (opened, closed) in sessions.items():
        local_open, local_close = opened.replace(tzinfo=None), closed.replace(tzinfo=None)
        expected = pd.date_range(local_open, local_close - timedelta(minutes=1), freq="1min")
        expected_total += len(expected)
        missing = [stamp for stamp in expected if stamp.to_pydatetime() not in valid_times]
        gaps.extend({"session_date": day.isoformat(), "missing_timestamp_et": stamp.isoformat(),
                     "session_open_et": local_open.isoformat(), "session_close_et": local_close.isoformat()}
                    for stamp in missing)
    gap_frame = pd.DataFrame(gaps, columns=["session_date", "missing_timestamp_et", "session_open_et", "session_close_et"])
    volume = dedup.loc[valid_ohlc & regular, "volume"]
    missing_volume = int(volume.isna().sum())
    zero_volume = int(volume.eq(0).sum())
    volume_values = volume.dropna()
    early_closes = [{"date": day.isoformat(), "close_et": bounds[1].isoformat()}
                    for day, bounds in sessions.items() if bounds[1].time() < datetime.strptime("16:00", "%H:%M").time()]
    summary = {
        "raw_rows": int(len(raw)), "raw_unique_timestamps": int(len(dedup)),
        "duplicate_timestamps": duplicates, "invalid_ohlc_rows": invalid_rows,
        "unaligned_minute_timestamps": int((~aligned).sum()),
        "out_of_session_rows_including_extended_hours": outside_session,
        "valid_regular_session_rows": int(len(eligible)),
        "first_raw_timestamp_new_york": raw.datetime.min().isoformat(sep=" "),
        "last_raw_timestamp_new_york": raw.datetime.max().isoformat(sep=" "),
        "first_regular_timestamp_new_york": eligible.datetime.min().isoformat(sep=" ") if len(eligible) else None,
        "last_regular_timestamp_new_york": eligible.datetime.max().isoformat(sep=" ") if len(eligible) else None,
        "trading_dates_with_regular_bars": int(eligible.datetime.dt.date.nunique()),
        "expected_regular_minute_intervals": int(expected_total),
        "missing_expected_regular_minute_intervals": int(len(gap_frame)),
        "regular_interval_coverage": float(len(eligible) / expected_total) if expected_total else None,
        "missing_volume_rows": missing_volume, "zero_volume_rows": zero_volume,
        "volume_min": float(volume_values.min()) if len(volume_values) else None,
        "volume_median": float(volume_values.median()) if len(volume_values) else None,
        "volume_p90": float(volume_values.quantile(.90)) if len(volume_values) else None,
        "volume_max": float(volume_values.max()) if len(volume_values) else None,
        "early_close_sessions": early_closes,
        "missing_bars_are_not_filled": True,
        "duplicate_handling_for_model_data": "Keep first source observation; record all duplicates in quality summary; never average or overwrite.",
        "invalid_ohlc_handling": "Exclude from model/label data; raw CSV remains unchanged.",
    }
    gap_frame.to_csv(OUT / "missing_expected_minutes.csv", index=False)
    (OUT / "data_quality.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    eligible["volume"] = pd.to_numeric(eligible.volume, errors="coerce")
    return eligible, gap_frame, summary


def clean_five_minute_reference(raw_path: Path = RAW_5M_PATH) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate direct IEX 5-minute bars for the identical permitted window."""
    raw = pd.read_csv(raw_path)
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True, errors="raise")
    raw["datetime"] = raw.timestamp_utc.dt.tz_convert(NY).dt.tz_localize(None)
    for name in ("open", "high", "low", "close", "volume", "trade_count"):
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    raw = raw.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    dup = raw.timestamp_utc.duplicated(keep="first")
    dedup = raw.loc[~dup].copy()
    vals = dedup[["open", "high", "low", "close"]].to_numpy(float)
    valid = np.isfinite(vals).all(axis=1) & (vals > 0).all(axis=1)
    valid &= ((dedup.high >= dedup[["open", "low", "close"]].max(axis=1)) &
              (dedup.low <= dedup[["open", "high", "close"]].min(axis=1))).to_numpy()
    sessions = session_map(dedup.datetime.dt.date.min(), dedup.datetime.dt.date.max())
    regular = []
    for i, row in enumerate(dedup.itertuples(index=False)):
        bounds = sessions.get(row.datetime.date())
        if not bounds or not valid[i]:
            continue
        op, close = (value.replace(tzinfo=None) for value in bounds)
        stamp = row.datetime.to_pydatetime()
        if op <= stamp and stamp + timedelta(minutes=5) <= close and (stamp - op).total_seconds() % 300 == 0:
            regular.append(i)
    clean = dedup.iloc[regular].copy().reset_index(drop=True)
    clean["session_date"] = clean.datetime.dt.date.astype(str)
    observed = set(clean.datetime)
    gaps = []
    expected_count = 0
    for day, (op_aware, close_aware) in sessions.items():
        op, close = op_aware.replace(tzinfo=None), close_aware.replace(tzinfo=None)
        expected = pd.date_range(op, close - timedelta(minutes=5), freq="5min")
        expected_count += len(expected)
        gaps.extend({"session_date": day.isoformat(), "missing_timestamp_et": stamp.isoformat()}
                    for stamp in expected if stamp not in observed)
    gap_frame = pd.DataFrame(gaps, columns=["session_date", "missing_timestamp_et"])
    quality = {"raw_rows": int(len(raw)), "duplicate_timestamps": int(dup.sum()),
               "invalid_ohlc_rows": int((~valid).sum()), "valid_regular_session_rows": int(len(clean)),
               "out_of_session_or_unaligned_rows": int(int(valid.sum()) - len(clean)),
               "expected_regular_5min_intervals": int(expected_count),
               "missing_expected_regular_5min_intervals": int(len(gap_frame)),
               "regular_interval_coverage": float(len(clean) / expected_count) if expected_count else None,
               "trading_dates_with_bars": int(clean.session_date.nunique()),
               "zero_volume_rows": int(clean.volume.eq(0).sum()),
               "missing_volume_rows": int(clean.volume.isna().sum()),
               "bar_source": "Alpaca IEX historical 5Min endpoint",
               "missing_bars_filled": False}
    gap_frame.to_csv(OUT / "missing_expected_5min_reference.csv", index=False)
    (OUT / "five_minute_reference_quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    return clean, gap_frame, quality


def _segments(frame: pd.DataFrame, interval: int):
    """Yield continuous same-session groups; gaps reset all rolling state."""
    for _, day in frame.groupby(frame.datetime.dt.date, sort=True):
        day = day.sort_values("datetime", kind="stable").reset_index(drop=True)
        split = day.datetime.diff().dt.total_seconds().div(interval * 60).ne(1)
        split.iloc[0] = True
        ids = split.cumsum()
        for _, segment in day.groupby(ids, sort=False):
            yield segment.reset_index(drop=True)


def one_minute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Causal trailing OHLCV features for each continuous session segment."""
    output = []
    if bars.empty:
        return pd.DataFrame(columns=["datetime", "session_date", *ONE_MINUTE_FEATURES])
    all_sessions = session_map(bars.datetime.min().date(), bars.datetime.max().date())
    for segment in _segments(bars, 1):
        # Short segments cannot warm even the 10-minute features, so skipping
        # their feature arithmetic changes no usable rows and saves work.
        if len(segment) < 11:
            continue
        s = segment.copy()
        close, open_, high, low = (s[name].astype(float) for name in ("close", "open", "high", "low"))
        f = pd.DataFrame(index=s.index)
        f["close"] = close
        for lag in (1, 3, 5, 10):
            f[f"return_{lag}_pct"] = close.pct_change(lag) * 100.0
        ret = close.pct_change() * 100.0
        for window in (3, 5, 10):
            f[f"realized_vol_{window}"] = ret.rolling(window, min_periods=window).std(ddof=1)
        f["range_pct"] = (high - low) / close * 100.0
        f["body_pct"] = (close - open_) / open_ * 100.0
        f["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close * 100.0
        f["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close * 100.0
        vol = pd.to_numeric(s.volume, errors="coerce")
        f["volume_log1p"] = np.log1p(vol.where(vol >= 0))
        prior = vol.shift(1)
        prior_mean = prior.rolling(10, min_periods=10).mean()
        prior_std = prior.rolling(10, min_periods=10).std(ddof=1)
        f["volume_ratio_10"] = vol / prior_mean.where(prior_mean > 0)
        f["volume_zscore_10"] = (vol - prior_mean) / prior_std.where(prior_std > 0)
        for window in (3, 5, 10):
            f[f"sma_ratio_{window}"] = close / close.rolling(window, min_periods=window).mean() - 1.0
            ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
            f[f"ema_ratio_{window}"] = close / ema - 1.0
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(7, min_periods=7).mean()
        loss = -delta.clip(upper=0).rolling(7, min_periods=7).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        rsi = rsi.mask((gain > 0) & (loss == 0), 100.0).mask((gain == 0) & (loss > 0), 0.0)
        rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)
        f["rsi_7"] = rsi
        r3, r10 = f["return_3_pct"], f["return_10_pct"]
        f["trend_agreement_3_10"] = np.sign(r3) * np.sign(r10)
        f["momentum_acceleration_3_10"] = r3 - r10 * (3.0 / 10.0)
        day = s.datetime.iloc[0].date()
        session_open, session_close = all_sessions[day]
        local_open, local_close = session_open.replace(tzinfo=None), session_close.replace(tzinfo=None)
        duration = max(1.0, (local_close - local_open).total_seconds() / 60.0)
        minute = (s.datetime - local_open).dt.total_seconds() / 60.0
        f["minutes_since_open"] = minute
        f["minutes_until_close"] = duration - minute - 1.0
        progress = (minute / max(duration - 1.0, 1.0)).clip(0, 1)
        f["session_progress"] = progress
        f["day_of_week"] = s.datetime.dt.dayofweek.astype(float)
        f["time_sin"] = np.sin(2 * np.pi * progress)
        f["time_cos"] = np.cos(2 * np.pi * progress)
        f.insert(0, "datetime", s.datetime)
        f["session_date"] = day.isoformat()
        output.append(f)
    if not output:
        return pd.DataFrame(columns=["datetime", "session_date", *ONE_MINUTE_FEATURES])
    result = pd.concat(output, ignore_index=True)
    return result[["datetime", "session_date", *ONE_MINUTE_FEATURES]]


def aggregate_complete_five_minute(bars_1m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build five-minute candles only from five complete expected source minutes."""
    rows, gap_rows = [], []
    available = {row.datetime: row for row in bars_1m.itertuples(index=False)}
    first, last = bars_1m.datetime.min().date(), bars_1m.datetime.max().date()
    for day, (opened, closed) in session_map(first, last).items():
        open_local, close_local = opened.replace(tzinfo=None), closed.replace(tzinfo=None)
        bucket_starts = pd.date_range(open_local, close_local - timedelta(minutes=5), freq="5min")
        for start in bucket_starts:
            expected = [start.to_pydatetime() + timedelta(minutes=i) for i in range(5)]
            present = [available.get(stamp) for stamp in expected]
            if any(item is None for item in present):
                missing = [stamp.isoformat() for stamp, item in zip(expected, present) if item is None]
                gap_rows.append({"session_date": day.isoformat(), "bucket_start_et": start.isoformat(),
                                 "missing_minutes_et": json.dumps(missing)})
                continue
            seq = present
            vols = [float(x.volume) for x in seq if pd.notna(x.volume)]
            rows.append({"datetime": start.to_pydatetime(), "session_date": day.isoformat(),
                         "open": float(seq[0].open), "high": max(float(x.high) for x in seq),
                         "low": min(float(x.low) for x in seq), "close": float(seq[-1].close),
                         "volume": sum(vols) if vols else np.nan,
                         "trade_count": sum(int(x.trade_count) for x in seq if pd.notna(x.trade_count)),
                         "source_minutes": 5})
    return pd.DataFrame(rows), pd.DataFrame(gap_rows, columns=["session_date", "bucket_start_et", "missing_minutes_et"])


def five_minute_b_features(bars_5m: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Use the original historical B calculation, with its frozen feature order."""
    spec = json.loads((ROOT / "models" / "alpaca_final_freeze_v1" / "frozen_candidate_spec.json").read_text(encoding="utf-8"))
    feature_order = list(spec["features_exact_order"])
    from live_app.features import calculate_b_features
    frame = bars_5m[["datetime", "open", "high", "low", "close"]].copy()
    frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
    features = calculate_b_features(frame, feature_order)
    return features, feature_order


def generate_labels(bars: pd.DataFrame, interval_minutes: int, horizon_minutes: int = TARGET_HORIZON_MINUTES) -> pd.DataFrame:
    """Label exact future candle timestamps; reject session/gap crossings."""
    if horizon_minutes % interval_minutes:
        raise ValueError("Horizon must be an integer multiple of the candle interval")
    source = bars.sort_values("datetime", kind="stable").reset_index(drop=True)
    if source.datetime.duplicated().any():
        raise ValueError("Target input timestamps must be unique")
    lookup = {stamp: i for i, stamp in enumerate(source.datetime)}
    first, last = source.datetime.min().date(), source.datetime.max().date()
    sessions = session_map(first, last)
    interval = timedelta(minutes=interval_minutes)
    horizon = timedelta(minutes=horizon_minutes)
    labels = []
    for row in source.itertuples(index=False):
        decision = row.datetime.to_pydatetime() if isinstance(row.datetime, pd.Timestamp) else row.datetime
        endpoint = decision + horizon
        bounds = sessions.get(decision.date())
        end_bounds = sessions.get(endpoint.date())
        if (bounds is None or end_bounds is None or endpoint.date() != decision.date()
                or decision < bounds[0].replace(tzinfo=None)
                or endpoint + interval > bounds[1].replace(tzinfo=None)):
            labels.append({"datetime": decision, "target_end": endpoint, "future_return_pct": np.nan,
                           "label": "UNAVAILABLE", "target_available": False})
            continue
        start_i = lookup.get(decision)
        end_i = lookup.get(endpoint)
        required_steps = horizon_minutes // interval_minutes
        # Check every expected timestamp explicitly. Merely checking the row
        # offset could accept an irregular/off-grid candle inside the path.
        path_is_continuous = (start_i is not None and end_i is not None and
            all(lookup.get(decision + interval * step) == start_i + step
                for step in range(required_steps + 1)))
        if not path_is_continuous:
            labels.append({"datetime": decision, "target_end": endpoint, "future_return_pct": np.nan,
                           "label": "UNAVAILABLE", "target_available": False})
            continue
        reference, future = float(source.iloc[start_i].close), float(source.iloc[end_i].close)
        ret = (future / reference - 1.0) * 100.0
        label = "UP" if ret > TARGET_THRESHOLD_PCT else "DOWN" if ret < -TARGET_THRESHOLD_PCT else "NO_SIGNAL"
        labels.append({"datetime": decision, "target_end": endpoint, "future_return_pct": ret,
                       "label": label, "target_available": True})
    return pd.DataFrame(labels)

