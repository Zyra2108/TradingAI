"""Independent causal ECL IEX minute features and endpoint-only labels.

This module reads only the explicitly approved train/validation cache when run
as a script. Importing it has no filesystem side effects.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
APPROVED_CACHE = ROOT / "models/alpaca_1min_research_v1/raw_ecl_iex_1min_train_validation.csv"
MIN_DATE, MAX_DATE = date(2025, 11, 13), date(2026, 9, 24)
TRAIN_END, VALIDATION_START = date(2026, 8, 5), date(2026, 8, 6)
HORIZONS = (10, 15)
THRESHOLD_PCT = 0.10
FEATURES = [
    "close", "return_1_pct", "return_3_pct", "return_5_pct", "return_10_pct",
    "realized_vol_3", "realized_vol_5", "realized_vol_10", "range_pct", "body_pct",
    "upper_wick_pct", "lower_wick_pct", "volume_log1p", "volume_ratio_10", "volume_zscore_10",
    "sma_ratio_3", "sma_ratio_5", "sma_ratio_10", "ema_ratio_3", "ema_ratio_5", "ema_ratio_10",
    "rsi_7", "trend_agreement_3_10", "momentum_acceleration_3_10", "minutes_since_open",
    "minutes_until_close", "session_progress", "day_of_week", "time_sin", "time_cos",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_provenance(path: Path = APPROVED_CACHE) -> dict:
    resolved = path.resolve()
    if resolved != APPROVED_CACHE.resolve():
        raise ValueError("Only the approved train/validation minute cache may be read")
    meta_path = resolved.with_name("fetch_metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("symbol") != "ECL" or meta.get("feed") != "iex" or meta.get("timeframe") != "1Min"
            or meta.get("permitted_new_york_date_range") != [MIN_DATE.isoformat(), MAX_DATE.isoformat()]
            or not meta.get("request_success")):
        raise ValueError("Approved source provenance check failed")
    if sha256(resolved) != meta.get("raw_csv_sha256"):
        raise ValueError("Approved source cache hash does not match fetch metadata")
    return {"path": str(resolved), "sha256": meta["raw_csv_sha256"], "test_data_accessed": False}


def _sessions(first: date, last: date):
    schedule = mcal.get_calendar("NYSE").schedule(start_date=first, end_date=last)
    return {pd.Timestamp(k).date(): (v.market_open.tz_convert("America/New_York").tz_localize(None),
                                     v.market_close.tz_convert("America/New_York").tz_localize(None))
            for k, v in schedule.iterrows()}


def load_clean_cache(path: Path = APPROVED_CACHE) -> tuple[pd.DataFrame, dict]:
    provenance = verify_provenance(path)
    raw = pd.read_csv(path)  # Exact, allowlisted path above; no globbing or split discovery.
    required = {"timestamp_utc", "open", "high", "low", "close", "volume", "trade_count"}
    if not required.issubset(raw.columns):
        raise ValueError("Approved cache is missing required OHLCV columns")
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True, errors="raise")
    raw["datetime"] = raw.timestamp_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume", "trade_count"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if raw.datetime.dt.date.min() < MIN_DATE or raw.datetime.dt.date.max() > MAX_DATE:
        raise ValueError("Approved cache contains timestamps outside permitted dates")
    raw = raw.sort_values("timestamp_utc", kind="stable").drop_duplicates("timestamp_utc", keep="first")
    ohlc = raw[["open", "high", "low", "close"]]
    valid = (np.isfinite(ohlc).all(axis=1) & (ohlc > 0).all(axis=1)
             & (raw.high >= raw[["open", "low", "close"]].max(axis=1))
             & (raw.low <= raw[["open", "high", "close"]].min(axis=1))
             & raw.timestamp_utc.dt.second.eq(0) & raw.timestamp_utc.dt.microsecond.eq(0))
    clean = raw.loc[valid].copy()
    sessions = _sessions(clean.datetime.dt.date.min(), clean.datetime.dt.date.max())
    regular = []
    for r in clean.itertuples():
        session = sessions.get(r.datetime.date())
        regular.append(bool(session and session[0] <= r.datetime < session[1]
                            and r.datetime + timedelta(minutes=1) <= session[1]))
    clean = clean.loc[regular].sort_values("datetime", kind="stable").reset_index(drop=True)
    clean["session_date"] = clean.datetime.dt.date.astype(str)
    return clean, {**provenance, "raw_rows": len(raw), "valid_regular_rows": len(clean),
                   "trading_dates": int(clean.session_date.nunique()), "date_min": str(clean.datetime.dt.date.min()),
                   "date_max": str(clean.datetime.dt.date.max()), "session_calendar": "NYSE"}


def _continuous_segments(bars: pd.DataFrame):
    for _, day in bars.groupby("session_date", sort=True):
        day = day.sort_values("datetime", kind="stable").reset_index(drop=True)
        breaks = day.datetime.diff().ne(pd.Timedelta(minutes=1))
        breaks.iloc[0] = True
        for _, seg in day.groupby(breaks.cumsum(), sort=False):
            yield seg.reset_index(drop=True)


def causal_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Trailing features reset at session boundaries and every missing minute."""
    sessions = _sessions(bars.datetime.dt.date.min(), bars.datetime.dt.date.max())
    outputs = []
    for s in _continuous_segments(bars):
        c, o, h, l = (s[x].astype(float) for x in ("close", "open", "high", "low"))
        f = pd.DataFrame(index=s.index)
        f["close"] = c
        for n in (1, 3, 5, 10):
            f[f"return_{n}_pct"] = c.pct_change(n) * 100
        ret = c.pct_change() * 100
        for n in (3, 5, 10):
            f[f"realized_vol_{n}"] = ret.rolling(n, min_periods=n).std(ddof=1)
        f["range_pct"] = (h-l)/c*100
        f["body_pct"] = (c-o)/o*100
        f["upper_wick_pct"] = (h-pd.concat([o,c],axis=1).max(axis=1))/c*100
        f["lower_wick_pct"] = (pd.concat([o,c],axis=1).min(axis=1)-l)/c*100
        v = s.volume.where(s.volume >= 0)
        f["volume_log1p"] = np.log1p(v)
        prior = v.shift(1); mean = prior.rolling(10,min_periods=10).mean(); std=prior.rolling(10,min_periods=10).std(ddof=1)
        f["volume_ratio_10"] = v/mean.where(mean>0); f["volume_zscore_10"]=(v-mean)/std.where(std>0)
        for n in (3,5,10):
            f[f"sma_ratio_{n}"]=c/c.rolling(n,min_periods=n).mean()-1
            ema=c.ewm(span=n,adjust=False,min_periods=n).mean(); f[f"ema_ratio_{n}"]=c/ema-1
        d=c.diff(); gain=d.clip(lower=0).rolling(7,min_periods=7).mean(); loss=-d.clip(upper=0).rolling(7,min_periods=7).mean()
        rs=gain/loss.replace(0,np.nan); rsi=100-100/(1+rs)
        rsi=rsi.mask((gain>0)&(loss==0),100).mask((gain==0)&(loss>0),0).mask((gain==0)&(loss==0),50)
        f["rsi_7"]=rsi; f["trend_agreement_3_10"]=np.sign(f.return_3_pct)*np.sign(f.return_10_pct)
        f["momentum_acceleration_3_10"]=f.return_3_pct-f.return_10_pct*0.3
        day=s.datetime.iloc[0].date(); op,cl=sessions[day]; duration=(cl-op).total_seconds()/60
        minute=(s.datetime-op).dt.total_seconds()/60
        f["minutes_since_open"]=minute; f["minutes_until_close"]=duration-minute-1
        progress=(minute/max(duration-1,1)).clip(0,1); f["session_progress"]=progress
        f["day_of_week"]=s.datetime.dt.dayofweek.astype(float)
        f["time_sin"]=np.sin(2*np.pi*progress); f["time_cos"]=np.cos(2*np.pi*progress)
        f.insert(0,"datetime",s.datetime); f.insert(1,"session_date",s.session_date)
        outputs.append(f)
    if not outputs: return pd.DataFrame(columns=["datetime","session_date",*FEATURES])
    return pd.concat(outputs,ignore_index=True)[["datetime","session_date",*FEATURES]]


def endpoint_targets(bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Definition A using sorted timestamp arrays and exact endpoint lookup.

    Runtime is O(n log n) for endpoint lookup; no per-reference DataFrame row
    indexing is performed. Missing intermediate bars are deliberately ignored.
    """
    if horizon not in HORIZONS:
        raise ValueError("horizon must be 10 or 15 minutes")
    ordered = bars.sort_values("datetime", kind="stable").reset_index(drop=True)
    if ordered.datetime.duplicated().any():
        raise ValueError("duplicate reference timestamps")
    if ordered.empty:
        return pd.DataFrame(columns=["datetime", "target_end", "future_return_pct", "label", "exclusion_reason"])

    # Explicitly normalize to nanoseconds. DatetimeIndex.asi8 is expressed in
    # the index's native unit in current pandas versions (often microseconds),
    # which must not be compared with Timedelta.value (nanoseconds).
    times = pd.DatetimeIndex(ordered.datetime)
    timestamp_ns = times.to_numpy(dtype="datetime64[ns]").view(np.int64)
    horizon_ns = pd.Timedelta(minutes=horizon).value
    endpoint_ns = timestamp_ns + horizon_ns
    positions = np.searchsorted(timestamp_ns, endpoint_ns, side="left")
    in_bounds = positions < len(timestamp_ns)
    endpoint_exists = np.zeros(len(timestamp_ns), dtype=bool)
    endpoint_exists[in_bounds] = timestamp_ns[positions[in_bounds]] == endpoint_ns[in_bounds]

    close = ordered.close.to_numpy(dtype=np.float64, copy=False)
    future_close = np.full(len(ordered), np.nan, dtype=np.float64)
    future_close[endpoint_exists] = close[positions[endpoint_exists]]
    returns = (future_close / close - 1.0) * 100.0

    dates = ordered.datetime.dt.date
    sessions = _sessions(dates.min(), dates.max())
    session_ok = np.zeros(len(ordered), dtype=bool)
    for day, row_indices in dates.groupby(dates, sort=False).groups.items():
        bounds = sessions.get(day)
        if bounds is None:
            continue
        ix = np.asarray(row_indices, dtype=np.intp)
        open_ns = np.datetime64(bounds[0], "ns").astype(np.int64)
        close_ns = np.datetime64(bounds[1], "ns").astype(np.int64)
        # The target is the close of the endpoint candle, so its full minute
        # must lie within the same official session (including early closes).
        session_ok[ix] = ((timestamp_ns[ix] >= open_ns)
                          & (endpoint_ns[ix] + pd.Timedelta(minutes=1).value <= close_ns))

    valid = session_ok & endpoint_exists
    labels = np.full(len(ordered), "UNAVAILABLE", dtype=object)
    labels[valid & (returns > THRESHOLD_PCT)] = "UP"
    labels[valid & (returns < -THRESHOLD_PCT)] = "DOWN"
    neutral = valid & (returns >= -THRESHOLD_PCT) & (returns <= THRESHOLD_PCT)
    labels[neutral] = "NO_SIGNAL"
    reasons = np.full(len(ordered), None, dtype=object)
    reasons[~session_ok] = "session_or_endpoint_boundary"
    reasons[session_ok & ~endpoint_exists] = "missing_exact_endpoint"

    return pd.DataFrame({
        "datetime": ordered.datetime.to_numpy(),
        "target_end": (timestamp_ns + horizon_ns).astype("datetime64[ns]"),
        "future_return_pct": returns,
        "label": labels,
        "exclusion_reason": reasons,
    })


def build_datasets() -> dict:
    bars,audit=load_clean_cache(); feats=causal_features(bars)
    base=bars[["datetime","session_date"]].merge(feats,on=["datetime","session_date"],validate="one_to_one")
    results={"source_audit":audit,"horizons":{}}
    dates=sorted(bars.datetime.dt.date.unique())
    train_dates={d for d in dates if d <= TRAIN_END}
    val_dates={d for d in dates if d >= VALIDATION_START}
    if not train_dates or not val_dates or max(train_dates) >= min(val_dates):
        raise ValueError("Approved chronological train/validation date boundary is not present")
    OUT.mkdir(parents=True,exist_ok=True)
    for h in HORIZONS:
        target=endpoint_targets(bars,h); ds=base.merge(target,on="datetime",validate="one_to_one")
        ready=ds[FEATURES].notna().all(axis=1)
        eligible=ds.label.isin(["UP","DOWN","NO_SIGNAL"]) & ready
        ds["split"]="excluded"; ds.loc[ds.datetime.dt.date.isin(train_dates),"split"]="train"
        ds.loc[ds.datetime.dt.date.isin(val_dates),"split"]="validation"
        boundary=min(val_dates); purge=(ds.split=="train") & (ds.target_end>=pd.Timestamp(boundary))
        ds.loc[purge,"split"]="purged"
        ds["feature_ready"]=ready
        path=OUT/f"dataset_{h}min.csv"; ds.to_csv(path,index=False)
        results["horizons"][str(h)]={"rows":len(ds),"feature_ready":int(ready.sum()),"eligible_feature_ready":int((eligible & ~purge).sum()),
            "purged_train_rows":int(purge.sum()),"train_dates":[str(min(train_dates)),str(max(train_dates))],
            "validation_dates":[str(min(val_dates)),str(max(val_dates))],"output":str(path)}
    (OUT/"pipeline_manifest.json").write_text(json.dumps({**results,"source_path":str(APPROVED_CACHE.resolve()),
        "source_sha256":audit["sha256"],"test_period_accessed":False,"models_trained":False,
        "feature_order":FEATURES,"definition":"A endpoint-only exact t+H; same NYSE session; ±0.10%"},indent=2)+"\n",encoding="utf-8")
    return results
