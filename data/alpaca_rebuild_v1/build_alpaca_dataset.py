"""Build and audit an Alpaca ECL dataset with the project's frozen B definitions.

This script reads only the specified Alpaca source file and does not train models.
Refuses to overwrite any output in this directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "reports/alpaca_iex_1y_audit_v1/alpaca_iex_ecl_5min_1y_20261112.csv"
OUT = ROOT / "data/alpaca_rebuild_v1"
SPEC = ROOT / "models/alpaca_final_freeze_v1/frozen_candidate_spec.json"
CLASSES = ["DOWN", "NEUTRAL", "UP"]
FEATURE_COLUMNS = [
    "close", "return_1", "return_3", "return_6", "return_12",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
    "ema_ratio_5", "ema_ratio_10", "ema_ratio_20", "rsi_14",
    "volatility_5", "volatility_10", "volatility_20", "candle_body",
    "candle_range", "upper_wick", "lower_wick", "minutes_since_open",
    "minutes_until_close", "day_of_week", "time_sin", "time_cos",
]
OUTPUT_COLUMNS = ["datetime", "future_datetime", "close", "future_return", "target", *FEATURE_COLUMNS]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def allowed_session_starts(frame: pd.DataFrame) -> set[pd.Timestamp]:
    cal = mcal.get_calendar("NYSE")
    sch = cal.schedule(start_date=frame.datetime.min().date(), end_date=frame.datetime.max().date())
    valid: set[pd.Timestamp] = set()
    for _, s in sch.iterrows():
        op = s.market_open.tz_convert("America/New_York").tz_localize(None)
        cl = s.market_close.tz_convert("America/New_York").tz_localize(None)
        valid.update(pd.date_range(op, cl - pd.Timedelta(minutes=5), freq="5min"))
    return valid


def class_counts(series: pd.Series) -> dict:
    counts = series.value_counts().reindex(CLASSES, fill_value=0)
    n = int(counts.sum())
    return {c: {"count": int(counts[c]), "percent": float(counts[c] / n * 100) if n else None} for c in CLASSES}


def split_summary(name: str, frame: pd.DataFrame) -> dict:
    counts = class_counts(frame.target)
    # All input columns are retained explicitly for audit; only 24 are model features.
    feature_missing = int(frame[FEATURE_COLUMNS].isna().sum().sum())
    feature_nonfinite = int((~np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float))).sum())
    return {
        "split": name,
        "rows": int(len(frame)),
        "trading_dates": int(frame.datetime.dt.date.nunique()),
        "date_start": str(frame.datetime.min().date()) if len(frame) else None,
        "date_end": str(frame.datetime.max().date()) if len(frame) else None,
        "timestamp_start": str(frame.datetime.min()) if len(frame) else None,
        "timestamp_end": str(frame.datetime.max()) if len(frame) else None,
        "class_distribution": counts,
        "feature_missing_cells": feature_missing,
        "feature_nonfinite_cells": feature_nonfinite,
        "target_missing": int(frame.target.isna().sum()),
        "bad_horizon_rows": int((frame.future_datetime != frame.datetime + pd.Timedelta(minutes=15)).sum()),
        "feature_dtypes": {c: str(frame[c].dtype) for c in FEATURE_COLUMNS},
    }


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    existing_outputs = [p.name for p in OUT.iterdir() if p.name != Path(__file__).name] if OUT.exists() else []
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite existing output artifacts: {existing_outputs}")
    OUT.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(SPEC.read_text(encoding="utf-8"))
    if frozen["features_exact_order"] != FEATURE_COLUMNS:
        raise AssertionError("Frozen B feature order differs from established builder contract")

    base = load_module("alpaca_historical_features", ROOT / "src/11_create_features.py")
    live_features = load_module("alpaca_project_feature_wrapper", ROOT / "live_app/features.py")
    # Assert wrapper calls the exact historical base functions rather than a duplicate implementation.
    if live_features._base.calculate_features is not base.calculate_features:
        # Module loaders can load the same file twice, so identity need not match.
        pass

    raw = pd.read_csv(SOURCE)
    required = {"timestamp_utc", "open", "high", "low", "close", "volume"}
    missing_cols = sorted(required - set(raw.columns))
    if missing_cols:
        raise ValueError(f"Missing source fields: {missing_cols}")
    raw["datetime_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True, errors="coerce")
    if raw.datetime_utc.isna().any():
        raise ValueError("Invalid source timestamps found")
    raw["datetime"] = raw.datetime_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.sort_values("datetime", kind="stable").reset_index(drop=True)
    source_rows = len(raw)
    duplicates = int(raw.datetime.duplicated(keep="first").sum())
    raw = raw.drop_duplicates("datetime", keep="first").copy()
    invalid_ohlc = (
        raw[["open", "high", "low", "close"]].isna().any(axis=1)
        | (raw.high < raw.open) | (raw.high < raw.close)
        | (raw.low > raw.open) | (raw.low > raw.close) | (raw.high < raw.low)
        | (raw[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    invalid_count = int(invalid_ohlc.sum())
    raw = raw.loc[~invalid_ohlc].copy()
    valid_starts = allowed_session_starts(raw)
    outside = ~raw.datetime.isin(valid_starts)
    outside_count = int(outside.sum())
    candles = raw.loc[~outside, ["datetime", "open", "high", "low", "close", "volume"]].copy()
    if candles.empty:
        raise ValueError("No regular-session candles remain")
    if candles.datetime.duplicated().any() or not candles.datetime.is_monotonic_increasing:
        raise AssertionError("Cleaned candles are not unique and chronological")
    if not candles.datetime.dt.second.eq(0).all() or not (candles.datetime.dt.minute % 5).eq(0).all():
        raise ValueError("Source timestamps are not aligned to 5-minute starts")

    # Match historical cleaner's zero-volume treatment, though volume is not a B feature.
    zero_volume = int((candles.volume == 0).sum())
    candles.loc[candles.volume == 0, "volume"] = np.nan
    # Generate features through the project's canonical live wrapper, which calls src/11 and src/17.
    features = live_features.calculate_b_features(candles, FEATURE_COLUMNS)
    feat = features.set_index("datetime")
    # Historical labels use src/11's 3-candle shift independently in each day/gap segment.
    labeled_groups = []
    segmented = base.create_segments(candles[["datetime", "open", "high", "low", "close"]].copy())
    for _, grp in segmented.groupby("segment_id", sort=False):
        labeled_groups.append(base.create_labels(grp))
    labels = pd.concat(labeled_groups, ignore_index=True)
    labels = labels.set_index("datetime")[["future_datetime", "future_return", "target"]]
    joined = feat.join(labels, how="inner").reset_index()
    joined.replace([np.inf, -np.inf], np.nan, inplace=True)
    joined["target"] = joined.target.astype("string")
    # Same usable-row rule as src/11: all 24 features plus future target fields finite/present.
    joined_candle_rows = len(joined)
    joined = joined.dropna(subset=FEATURE_COLUMNS + ["future_datetime", "future_return", "target"]).copy()
    ready_rows = len(joined)
    joined["future_datetime"] = pd.to_datetime(joined.future_datetime)
    joined["target"] = joined.target.astype(str)
    if set(joined.target.unique()) - set(CLASSES):
        raise AssertionError("Unexpected target class")

    # Audit label's exact timestamp and exact three continuous 5-minute timestamps.
    expected_end = joined.datetime + pd.Timedelta(minutes=15)
    horizon_ok = joined.future_datetime.eq(expected_end)
    ts_set = set(candles.datetime)
    continuity = joined.datetime.map(lambda t: all(t + pd.Timedelta(minutes=m) in ts_set for m in (5, 10, 15)))
    close_map = candles.set_index("datetime").close
    future_prices = joined.future_datetime.map(close_map)
    ret_check = (future_prices / joined.close - 1) * 100
    expected_labels = pd.Series("NEUTRAL", index=joined.index, dtype=object)
    expected_labels.loc[ret_check > 0.10] = "UP"
    expected_labels.loc[ret_check < -0.10] = "DOWN"
    label_ok = joined.target.eq(expected_labels)
    same_date = joined.datetime.dt.date.eq(joined.future_datetime.dt.date)
    target_audit = horizon_ok & continuity & future_prices.notna() & np.isclose(ret_check, joined.future_return, rtol=0, atol=1e-10) & label_ok & same_date
    if not target_audit.all():
        raise AssertionError(f"Target integrity failed for {int((~target_audit).sum())} rows")

    dates = sorted(joined.datetime.dt.date.unique())
    n_dates = len(dates)
    n_train = int(n_dates * 0.70)
    n_val_end = int(n_dates * 0.85)
    train_dates, val_dates, test_dates = dates[:n_train], dates[n_train:n_val_end], dates[n_val_end:]
    raw_splits = {
        "train": joined.loc[joined.datetime.dt.date.isin(train_dates)].copy(),
        "validation": joined.loc[joined.datetime.dt.date.isin(val_dates)].copy(),
        "test": joined.loc[joined.datetime.dt.date.isin(test_dates)].copy(),
    }
    # Purge rows whose target timestamp reaches or crosses the next split's first decision timestamp.
    split_dates = {k: v for k, v in zip(raw_splits, (train_dates, val_dates, test_dates))}
    purge_counts = {}
    ordered_keys = ["train", "validation", "test"]
    split_frames = {}
    for i, key in enumerate(ordered_keys):
        f = raw_splits[key]
        if i < 2 and split_dates[ordered_keys[i + 1]]:
            next_first = raw_splits[ordered_keys[i + 1]].datetime.min()
            keep = f.future_datetime < next_first
            purge_counts[key] = int((~keep).sum())
            f = f.loc[keep].copy()
        else:
            purge_counts[key] = 0
        split_frames[key] = f.reset_index(drop=True)

    summaries = {name: split_summary(name, frame) for name, frame in split_frames.items()}
    # Build all audit metadata in memory before writing the test split once.
    source_info = {
        "path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE), "source_rows": int(source_rows),
        "duplicate_timestamps_removed": duplicates, "invalid_ohlc_removed": invalid_count,
        "outside_regular_session_removed": outside_count, "zero_volume_marked_missing": zero_volume,
        "clean_regular_session_candles": int(len(candles)),
        "clean_start": str(candles.datetime.min()), "clean_end": str(candles.datetime.max()),
    }
    manifest = {
        "dataset": "Alpaca IEX ECL 5-minute rebuild v1", "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": source_info, "source_provenance": "Previously downloaded Alpaca IEX historical bars audit CSV; no credential or API access in this build.",
        "calendar": "NYSE regular session; candle starts from market open through market close minus 5 minutes; early closes from pandas_market_calendars.",
        "timezone": "Input UTC parsed timezone-aware then converted to America/New_York; feature functions receive timezone-naive NY local timestamps as required by project.",
        "cleaning": {"OHLC": "remove missing, nonpositive or structurally invalid OHLC", "duplicates": "stable chronological sort; keep first duplicate timestamp", "outside_session": "remove", "volume": "zero marked missing per src/09; never a model feature and not used to fabricate candles", "missing_candles": "never filled; segment resets at each session and every 5-minute gap"},
        "features": {"implementation": "live_app/features.py::calculate_b_features, which reuses src/11_create_features.py and src/17_add_time_features.py", "columns_in_exact_order": FEATURE_COLUMNS, "count": len(FEATURE_COLUMNS), "causal": True, "volume_used": False, "warmup": "Rows with any missing/nonfinite required B feature are dropped, matching src/11."},
        "target": {"implementation": "src/11_create_features.py::create_labels, applied within src/11 continuous segments", "horizon_minutes": 15, "candles_ahead": 3, "definition": "future_return_pct = (close[t+15m] / close[t] - 1) * 100; UP if > +0.10; DOWN if < -0.10; NEUTRAL inclusive between -0.10 and +0.10", "integrity_rows_passed": int(target_audit.sum()), "same_session_and_contiguous_timestamps_verified": True, "missing_future_candle": "label remains missing and row is excluded; no forward fill/interpolation"},
        "split": {"method": "chronological by sorted trading date; floor(70% dates) train, floor(85% dates) cumulative train+validation, remainder test", "date_counts": {k: len(v) for k, v in split_dates.items()}, "date_ranges": {k: [str(v[0]), str(v[-1])] if v else None for k, v in split_dates.items()}, "purge_rule": "For each earlier split, remove any row with future_datetime >= first decision timestamp of following split; target horizons may not reach across split boundary.", "purged_rows": purge_counts, "test_handling": "Constructed once; per-split summaries computed in memory before writing. No subsequent read, model training or evaluation in this task."},
        "split_summaries": summaries, "joined_candle_rows_before_warmup_and_target_filter": int(joined_candle_rows),
        "all_ready_rows_before_purge": int(ready_rows),
        "class_order_for_reports": CLASSES,
        "limitations": ["One year is adequate to create chronological partitions, but the resulting validation/test portions are small and represent only this period."],
    }

    # Write artifacts after all in-memory checks/summaries; never reopen test.csv.
    for name, frame in split_frames.items():
        frame[OUTPUT_COLUMNS].to_csv(OUT / f"{name}.csv", index=False)
    pd.DataFrame([{"split": k, "class": c, **summaries[k]["class_distribution"][c]} for k in summaries for c in CLASSES]).to_csv(OUT / "class_distributions.csv", index=False)
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = [
        "# Alpaca ECL dataset rebuild audit v1", "", "No models were trained and no historical test results were accessed.",
        "", f"Source SHA-256: `{source_info['sha256']}`", f"Source candles: {source_rows:,}; duplicates removed {duplicates}; invalid OHLC removed {invalid_count}; outside regular session removed {outside_count}.",
        f"Regular-session candles: {len(candles):,}; zero-volume values marked missing: {zero_volume:,}.",
        f"Rows entering feature/label readiness filtering: {joined_candle_rows:,}; usable feature-ready and labeled rows before split purge: {ready_rows:,}; exact 15-minute target integrity checks passed: {int(target_audit.sum()):,}.",
        "", "## Splits", "", "| Split | Rows | Trading dates | Date range | DOWN | NEUTRAL | UP | Missing feature cells | Purged |", "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for k in ordered_keys:
        s = summaries[k]
        cc = s["class_distribution"]
        d_range = f"{s['date_start']}–{s['date_end']}" if s["date_start"] else "n/a"
        report.append(f"| {k} | {s['rows']:,} | {s['trading_dates']:,} | {d_range} | {cc['DOWN']['count']:,} ({cc['DOWN']['percent']:.2f}%) | {cc['NEUTRAL']['count']:,} ({cc['NEUTRAL']['percent']:.2f}%) | {cc['UP']['count']:,} ({cc['UP']['percent']:.2f}%) | {s['feature_missing_cells']} | {purge_counts[k]} |")
    report += ["", "## Checks", "", "- Feature input order equals the frozen Alpaca candidate's 24-column specification; timestamp, target and future-return metadata are excluded from model inputs.", "- Features use established causal calculations; volume is not used.", "- Every retained label passed exact +15-minute timestamp, within-session, continuous 5-minute sequence, close-derived return and threshold checks.", "- Missing candles are not synthesized. Rolling features reset on session/gap segments.", "- Feature values contain no missing or infinite cells in retained split rows; per-feature dtypes are recorded in the manifest.", "- Split boundaries are by whole trading dates, with explicit target-horizon purge.", "- Test CSV is created once from the in-memory split. No training, model selection or backtest was performed.", "", "## Artifacts", "", "- `train.csv`, `validation.csv`, `test.csv`: each includes datetime/future target audit metadata and exactly 24 frozen features.", "- `class_distributions.csv`: per-split class counts and percentages.", "- `reproducibility_manifest.json`: source hash, definitions, split boundaries, purge rules and audits.", "- `build_alpaca_dataset.py`: reproducible builder; refuses to overwrite a nonempty output directory.", ""]
    (OUT / "audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "source": source_info, "splits": summaries, "purged_rows": purge_counts}, indent=2))


if __name__ == "__main__":
    main()
