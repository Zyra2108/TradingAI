"""Compare sparse Alpaca IEX minute-bar aggregation with native 5Min bars.

Inputs are the pre-existing, explicitly train/validation-only cache created by
the Alpaca 1-minute research. This script never reads a test split or makes a
network request. Run from the project root or from any working directory.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_DIR = ROOT / "models" / "alpaca_1min_research_v1"
MINUTE_PATH = SOURCE_DIR / "raw_ecl_iex_1min_train_validation.csv"
NATIVE_PATH = SOURCE_DIR / "raw_ecl_iex_5min_train_validation.csv"
MINUTE_META_PATH = SOURCE_DIR / "fetch_metadata.json"
NATIVE_META_PATH = SOURCE_DIR / "fetch_5min_metadata.json"
SPEC_PATH = ROOT / "models" / "alpaca_final_freeze_v1" / "frozen_candidate_spec.json"
START_DATE = date(2025, 11, 13)
END_DATE = date(2026, 9, 24)
UTC = timezone.utc
OHLC = ["open", "high", "low", "close"]
FIELDS = [*OHLC, "volume", "trade_count"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_metadata() -> dict:
    minute_meta = json.loads(MINUTE_META_PATH.read_text(encoding="utf-8"))
    native_meta = json.loads(NATIVE_META_PATH.read_text(encoding="utf-8"))
    for label, meta, timeframe in (("minute", minute_meta, "1Min"), ("native", native_meta, "5Min")):
        if meta.get("symbol") != "ECL" or meta.get("feed") != "iex" or meta.get("timeframe") != timeframe:
            raise ValueError(f"{label} cache provenance does not match ECL IEX {timeframe}")
        if meta.get("permitted_new_york_date_range") != [START_DATE.isoformat(), END_DATE.isoformat()]:
            raise ValueError(f"{label} cache date provenance differs from permitted train/validation range")
        if not meta.get("request_success"):
            raise ValueError(f"{label} cache fetch metadata reports failure")
        if meta.get("credential_values_saved") or meta.get("request_headers_saved"):
            raise ValueError(f"{label} metadata unexpectedly includes secret material")
    return {"minute": minute_meta, "native": native_meta}


def _session_map(days: set[date]):
    from live_app.calendar import NY, session_for_date

    result = {}
    for day in days:
        session = session_for_date(day)
        if session:
            result[day] = (pd.Timestamp(session[0]).tz_convert("UTC"), pd.Timestamp(session[1]).tz_convert("UTC"))
    return NY, result


def _read_source(path: Path, kind: str, allowed_start: date, allowed_end: date, sessions):
    raw = pd.read_csv(path)
    required = {"timestamp_utc", *FIELDS}
    if not required.issubset(raw.columns):
        raise ValueError(f"{path.name} missing columns: {sorted(required-set(raw.columns))}")
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="raise")
    raw["date_ny"] = raw["timestamp_utc"].dt.tz_convert("America/New_York").dt.date
    # Metadata is checked before opening either CSV; assert the file itself also
    # stays within that already-permitted boundary before further processing.
    if raw["date_ny"].min() < allowed_start or raw["date_ny"].max() > allowed_end:
        raise ValueError(f"{path.name} contains rows outside the permitted historical date range")
    for field in FIELDS:
        raw[field] = pd.to_numeric(raw[field], errors="coerce")
    raw["ohlc_valid"] = (
        raw["open"].notna() & raw["high"].notna() & raw["low"].notna() & raw["close"].notna()
        & np.isfinite(raw[OHLC]).all(axis=1)
        & (raw["high"] >= raw[["open", "low", "close"]].max(axis=1))
        & (raw["low"] <= raw[["open", "high", "close"]].min(axis=1))
    )
    raw["session_valid"] = False
    raw["aligned"] = False
    raw["bucket_start_utc"] = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    raw["missing_minutes"] = None
    for day, group in raw.groupby("date_ny", sort=False):
        session = sessions.get(day)
        if session is None:
            continue
        opened, closed = session
        in_session = (group["timestamp_utc"] >= opened) & (group["timestamp_utc"] < closed)
        if kind == "1Min":
            end_ok = group["timestamp_utc"] + pd.Timedelta(minutes=1) <= closed
            offset = (group["timestamp_utc"] - opened).dt.total_seconds()
            aligned = (offset % 60 == 0) & end_ok
            bucket = opened + pd.to_timedelta(np.floor(offset / 300) * 300, unit="s")
        else:
            end_ok = group["timestamp_utc"] + pd.Timedelta(minutes=5) <= closed
            offset = (group["timestamp_utc"] - opened).dt.total_seconds()
            aligned = (offset % 300 == 0) & end_ok
            bucket = group["timestamp_utc"]
        raw.loc[group.index, "session_valid"] = (in_session & end_ok).to_numpy()
        raw.loc[group.index, "aligned"] = (in_session & aligned).to_numpy()
        raw.loc[group.index, "bucket_start_utc"] = bucket.to_numpy()
    duplicate = raw["timestamp_utc"].duplicated(keep=False)
    quality = {
        "source_file": path.name,
        "rows_total": int(len(raw)),
        "date_min_new_york": raw["date_ny"].min().isoformat(),
        "date_max_new_york": raw["date_ny"].max().isoformat(),
        "duplicate_rows": int(duplicate.sum()),
        "invalid_ohlc_rows": int((~raw["ohlc_valid"]).sum()),
        "out_of_session_or_nonexistent_session_rows": int((~raw["session_valid"]).sum()),
        "unaligned_rows": int((raw["session_valid"] & ~raw["aligned"]).sum()),
        "volume_missing_rows": int(raw["volume"].isna().sum()),
        "trade_count_missing_rows": int(raw["trade_count"].isna().sum()),
    }
    reason = pd.Series("eligible", index=raw.index, dtype="object")
    reason.loc[~raw["ohlc_valid"]] = "invalid_ohlc"
    reason.loc[~raw["session_valid"]] = "outside_or_invalid_session"
    reason.loc[raw["session_valid"] & ~raw["aligned"]] = "unaligned_timestamp_or_bucket_crosses_close"
    reason.loc[duplicate] = "duplicate_timestamp"
    excluded = raw.loc[reason.ne("eligible"), ["timestamp_utc", *FIELDS]].copy()
    excluded.insert(0, "source_file", path.name)
    excluded.insert(2, "reason", reason.loc[reason.ne("eligible")])
    excluded["timestamp_new_york"] = excluded["timestamp_utc"].dt.tz_convert("America/New_York").astype(str)
    clean = raw.loc[raw["ohlc_valid"] & raw["aligned"] & ~duplicate].copy()
    clean = clean.sort_values("timestamp_utc").reset_index(drop=True)
    return clean, quality, excluded


def reconstruct(minute: pd.DataFrame, sessions: dict):
    rows = []
    for bucket, group in minute.groupby("bucket_start_utc", sort=True):
        bucket = pd.Timestamp(bucket)
        day = bucket.tz_convert("America/New_York").date()
        opened, closed = sessions[day]
        expected = [bucket + pd.Timedelta(minutes=i) for i in range(5)]
        observed = set(group["timestamp_utc"])
        missing = [t.isoformat().replace("+00:00", "Z") for t in expected if t not in observed]
        group = group.sort_values("timestamp_utc")
        volumes = group["volume"].dropna()
        counts = group["trade_count"].dropna()
        rows.append({
            "timestamp_utc": bucket,
            "date_ny": day,
            "open": float(group.iloc[0]["open"]),
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group.iloc[-1]["close"]),
            "volume": float(volumes.sum()) if len(volumes) else np.nan,
            "trade_count": float(counts.sum()) if len(counts) else np.nan,
            "observed_minute_count": int(group["timestamp_utc"].nunique()),
            "missing_minutes": missing,
            "session_open_utc": opened,
            "session_close_utc": closed,
        })
    return pd.DataFrame(rows)


def prepare_native(native: pd.DataFrame):
    out = native[["timestamp_utc", "date_ny", *FIELDS]].copy()
    out = out.rename(columns={"timestamp_utc": "native_timestamp_utc"})
    return out


def compare_bars(rebuilt: pd.DataFrame, native: pd.DataFrame):
    rebuilt = rebuilt.rename(columns={"timestamp_utc": "bucket_start_utc"})
    native = native.rename(columns={"native_timestamp_utc": "bucket_start_utc"})
    merged = rebuilt.merge(native, on=["bucket_start_utc", "date_ny"], how="outer", suffixes=("_rebuilt", "_native"), indicator=True)
    matched = merged.loc[merged["_merge"] == "both"].copy()
    for field in FIELDS:
        matched[f"{field}_abs_diff"] = (matched[f"{field}_rebuilt"] - matched[f"{field}_native"]).abs()
        matched[f"{field}_exact"] = matched[f"{field}_rebuilt"].eq(matched[f"{field}_native"])
    matched["ohlc_exact"] = matched[[f"{c}_exact" for c in OHLC]].all(axis=1)
    matched["ohlc_tol_1e8"] = np.isclose(matched[[f"{c}_rebuilt" for c in OHLC]], matched[[f"{c}_native" for c in OHLC]], rtol=0, atol=1e-8).all(axis=1)
    matched["volume_comparable"] = matched["volume_rebuilt"].notna() & matched["volume_native"].notna()
    matched["volume_exact"] = matched["volume_comparable"] & matched["volume_exact"]
    matched["trade_count_comparable"] = matched["trade_count_rebuilt"].notna() & matched["trade_count_native"].notna()
    matched["trade_count_exact"] = matched["trade_count_comparable"] & matched["trade_count_exact"]
    mismatch = matched.loc[~matched["ohlc_exact"] | ~matched["volume_exact"] | ~matched["trade_count_exact"]].copy()
    for col in ("bucket_start_utc", "date_ny"):
        mismatch[col] = mismatch[col].astype(str)
    mismatch_rows = []
    for _, row in mismatch.iterrows():
        mismatch_rows.append({
            "bucket_start_utc": row["bucket_start_utc"], "date_ny": row["date_ny"],
            "observed_minute_count": int(row.get("observed_minute_count", 0)),
            "missing_minutes": json.dumps(row.get("missing_minutes", [])),
            **{f"{field}_abs_diff": row[f"{field}_abs_diff"] for field in FIELDS},
            "ohlc_exact": bool(row["ohlc_exact"]), "volume_exact": bool(row["volume_exact"]),
            "trade_count_exact": bool(row["trade_count_exact"]),
        })
    mismatch_columns = ["bucket_start_utc", "date_ny", "observed_minute_count", "missing_minutes",
                        *[f"{field}_abs_diff" for field in FIELDS], "ohlc_exact", "volume_exact", "trade_count_exact"]
    return merged, matched, pd.DataFrame(mismatch_rows, columns=mismatch_columns)


def _feature_audit(rebuilt: pd.DataFrame, native: pd.DataFrame, feature_order: list[str]):
    from live_app.features import calculate_b_features

    def calc(data):
        if "native_timestamp_utc" in data.columns:
            data = data.rename(columns={"native_timestamp_utc": "timestamp_utc"})
        frame = data[["timestamp_utc", *OHLC]].copy()
        local = pd.to_datetime(frame.pop("timestamp_utc"), utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        frame.insert(0, "datetime", local)
        return calculate_b_features(frame, feature_order)

    rf, nf = calc(rebuilt), calc(native)
    rf = rf.set_index("datetime")
    nf = nf.set_index("datetime")
    common = rf.index.intersection(nf.index)
    rows = []
    for feature in feature_order:
        a, b = rf.loc[common, feature], nf.loc[common, feature]
        finite = a.notna() & b.notna() & np.isfinite(a.astype(float)) & np.isfinite(b.astype(float))
        diffs = (a[finite].astype(float) - b[finite].astype(float)).abs()
        rows.append({
            "feature": feature,
            "common_timestamp_rows": len(common),
            "both_finite_rows": int(finite.sum()),
            "exact_equal_rows": int((a[finite].astype(float) == b[finite].astype(float)).sum()),
            "exact_equal_rate": float((a[finite].astype(float) == b[finite].astype(float)).mean()) if finite.any() else None,
            "max_abs_difference": float(diffs.max()) if len(diffs) else None,
            "mean_abs_difference": float(diffs.mean()) if len(diffs) else None,
            "different_nan_status_rows": int((a.isna() != b.isna()).sum()),
        })
    feature_df = pd.DataFrame(rows)
    common_ready = rf.loc[common, feature_order].notna().all(axis=1) & nf.loc[common, feature_order].notna().all(axis=1)
    ready_match = 0
    for idx in common[common_ready]:
        if np.array_equal(rf.loc[idx, feature_order].to_numpy(float), nf.loc[idx, feature_order].to_numpy(float)):
            ready_match += 1
    def segments(data):
        timestamp_column = "native_timestamp_utc" if "native_timestamp_utc" in data.columns else "timestamp_utc"
        ts = pd.to_datetime(data[timestamp_column], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        diff = ts.diff()
        days = ts.dt.date
        starts = days.ne(days.shift()) | diff.ne(pd.Timedelta(minutes=5))
        return int(starts.sum()), int(data.shape[0])
    rseg, _ = segments(rebuilt)
    nseg, _ = segments(native)
    audit = {
        "feature_count": len(feature_order),
        "feature_order_matches_frozen_spec": len(feature_order) == 24 and len(set(feature_order)) == 24,
        "native_rows": len(nf), "reconstructed_rows": len(rf), "common_timestamps": len(common),
        "native_ready_rows": int(nf[feature_order].notna().all(axis=1).sum()),
        "reconstructed_ready_rows": int(rf[feature_order].notna().all(axis=1).sum()),
        "common_rows_ready_in_both": int(common_ready.sum()),
        "common_ready_rows_all_features_exact": ready_match,
        "native_feature_segments": nseg, "reconstructed_feature_segments": rseg,
        "all_feature_nan_status_identical_on_common_timestamps": bool((rf.loc[common, feature_order].isna() == nf.loc[common, feature_order].isna()).all().all()),
    }
    return audit, feature_df


def run() -> dict:
    metadata = _validate_metadata()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    feature_order = list(spec["features_exact_order"])
    if len(feature_order) != 24:
        raise ValueError("Frozen B feature specification must contain exactly 24 names")
    # Read only the specifically named train/validation caches; never look up or
    # open any test split file.
    raw_min = pd.read_csv(MINUTE_PATH, usecols=["timestamp_utc"])
    raw_5m = pd.read_csv(NATIVE_PATH, usecols=["timestamp_utc"])
    for label, frame in (("1Min", raw_min), ("5Min", raw_5m)):
        times = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise").dt.tz_convert("America/New_York").dt.date
        if times.min() < START_DATE or times.max() > END_DATE:
            raise ValueError(f"{label} cached timestamps fall outside verified permitted dates")
    days = set(pd.to_datetime(raw_min["timestamp_utc"], utc=True).dt.tz_convert("America/New_York").dt.date)
    days |= set(pd.to_datetime(raw_5m["timestamp_utc"], utc=True).dt.tz_convert("America/New_York").dt.date)
    _, sessions = _session_map(days)
    minute, minute_quality, minute_excluded = _read_source(MINUTE_PATH, "1Min", START_DATE, END_DATE, sessions)
    native, native_quality, native_excluded = _read_source(NATIVE_PATH, "5Min", START_DATE, END_DATE, sessions)
    reconstructed = reconstruct(minute, sessions)
    native = prepare_native(native)
    merged, matched, mismatches = compare_bars(reconstructed, native)
    matched_count = len(matched)
    exact_ohlc = int(matched["ohlc_exact"].sum()) if matched_count else 0
    volume_comparable = matched["volume_comparable"]
    trade_comparable = matched["trade_count_comparable"]
    bucket_rows = []
    for n in range(1, 6):
        group = matched.loc[matched["observed_minute_count"] == n]
        bucket_rows.append({
            "observed_minute_count": n,
            "matched_buckets": int(len(group)),
            "exact_ohlc_matches": int(group["ohlc_exact"].sum()),
            "ohlc_exact_rate": float(group["ohlc_exact"].mean()) if len(group) else None,
            "exact_volume_matches": int(group.loc[group["volume_comparable"], "volume_exact"].sum()),
            "volume_comparable": int(group["volume_comparable"].sum()),
            "exact_trade_count_matches": int(group.loc[group["trade_count_comparable"], "trade_count_exact"].sum()),
            "trade_count_comparable": int(group["trade_count_comparable"].sum()),
        })
    bucket_df = pd.DataFrame(bucket_rows)
    date_df = matched.groupby("date_ny", as_index=False).agg(
        matched_buckets=("bucket_start_utc", "count"),
        exact_ohlc_matches=("ohlc_exact", "sum"),
        exact_volume_matches=("volume_exact", "sum"),
        exact_trade_count_matches=("trade_count_exact", "sum"),
    )
    date_df["date_ny"] = date_df["date_ny"].astype(str)
    date_df["ohlc_exact_rate"] = date_df["exact_ohlc_matches"] / date_df["matched_buckets"]
    date_df["volume_exact_rate"] = date_df["exact_volume_matches"] / date_df["matched_buckets"]
    date_df["trade_count_exact_rate"] = date_df["exact_trade_count_matches"] / date_df["matched_buckets"]
    feature_audit, feature_by_name = _feature_audit(reconstructed, native, feature_order)
    excluded_rows = pd.concat([minute_excluded, native_excluded], ignore_index=True)
    excluded_rows["timestamp_utc"] = excluded_rows["timestamp_utc"].astype(str)
    merged["bucket_start_utc"] = merged["bucket_start_utc"].astype(str)
    merged["date_ny"] = merged["date_ny"].astype(str)
    count_dist = reconstructed["observed_minute_count"].value_counts().to_dict()
    summary = {
        "provenance": {
            "symbol": "ECL", "feed": "iex", "minute_timeframe": "1Min", "native_timeframe": "5Min",
            "permitted_dates_new_york": [START_DATE.isoformat(), END_DATE.isoformat()],
            "train_dates": ["2025-11-13", "2026-08-05"], "validation_dates": ["2026-08-06", "2026-09-24"],
            "test_data_accessed": False, "network_requests_made": False,
        },
        "source_quality": {"minute": minute_quality, "native_5min": native_quality},
        "counts": {
            "valid_minute_bars": len(minute), "valid_native_5min_bars": len(native),
            "reconstructed_sparse_5min_bars": len(reconstructed),
            "reconstructed_buckets_by_observed_minute_count": {str(k): int(v) for k, v in sorted(count_dist.items())},
            "matched_buckets": matched_count,
            "missing_native_buckets": int((merged["_merge"] == "left_only").sum()),
            "missing_reconstructed_buckets": int((merged["_merge"] == "right_only").sum()),
        },
        "parity": {
            "exact_ohlc_matches": exact_ohlc,
            "exact_ohlc_match_rate": float(matched["ohlc_exact"].mean()) if matched_count else None,
            "ohlc_matches_with_abs_tolerance_1e-8": int(matched["ohlc_tol_1e8"].sum()) if matched_count else 0,
            "exact_volume_matches": int(matched.loc[volume_comparable, "volume_exact"].sum()),
            "volume_comparable_buckets": int(volume_comparable.sum()),
            "exact_volume_match_rate": float(matched.loc[volume_comparable, "volume_exact"].mean()) if volume_comparable.any() else None,
            "exact_trade_count_matches": int(matched.loc[trade_comparable, "trade_count_exact"].sum()),
            "trade_count_comparable_buckets": int(trade_comparable.sum()),
            "exact_trade_count_match_rate": float(matched.loc[trade_comparable, "trade_count_exact"].mean()) if trade_comparable.any() else None,
            "max_abs_difference": {field: float(matched[f"{field}_abs_diff"].max()) if matched_count else None for field in FIELDS},
        },
        "by_observed_minute_count": bucket_rows,
        "feature_compatibility": feature_audit,
        "data_revision_caveat": "The two cached timeframes were fetched separately; late updates or historical revisions may explain isolated differences.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    bucket_df.to_csv(OUT / "parity_by_observed_minute_count.csv", index=False)
    date_df.to_csv(OUT / "parity_by_date.csv", index=False)
    mismatches.to_csv(OUT / "mismatch_details.csv", index=False)
    feature_by_name.to_csv(OUT / "feature_compatibility_by_feature.csv", index=False)
    merged[["bucket_start_utc", "date_ny", "observed_minute_count", "missing_minutes", "_merge"]].to_csv(OUT / "bucket_coverage.csv", index=False)
    excluded_rows.to_csv(OUT / "excluded_source_rows.csv", index=False)
    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_files": {
            str(MINUTE_PATH.relative_to(ROOT)): sha256(MINUTE_PATH),
            str(NATIVE_PATH.relative_to(ROOT)): sha256(NATIVE_PATH),
            str(MINUTE_META_PATH.relative_to(ROOT)): sha256(MINUTE_META_PATH),
            str(NATIVE_META_PATH.relative_to(ROOT)): sha256(NATIVE_META_PATH),
            str(SPEC_PATH.relative_to(ROOT)): sha256(SPEC_PATH),
            str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve()),
            str((OUT / "test_sparse_aggregation.py").resolve().relative_to(ROOT)): sha256(OUT / "test_sparse_aggregation.py"),
        },
        "date_boundary_assertion": "Both train_validation cache files were verified against provenance metadata and all NY dates were <= 2026-09-24 before aggregation.",
        "test_dataset_or_period_read": False,
        "network_requests_made": False,
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "calendar": "live_app.calendar.session_for_date; America/New_York; regular session/early close aware",
        "bucket_definition": "5 minutes from NYSE regular session open; [start, start+5m); no fill; omit zero-observation buckets",
        "aggregation": {"open": "first observed minute open", "high": "max observed minute high", "low": "min observed minute low", "close": "last observed minute close", "volume": "sum available observed minute volumes", "trade_count": "sum available counts"},
    }
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
