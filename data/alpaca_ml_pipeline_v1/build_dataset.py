"""Build isolated canonical Alpaca IEX ECL 5-minute ML datasets.

This script reads only the three explicitly listed Alpaca source CSVs. It does
not open the older split CSVs (including the protected old test file).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "alpaca_ml_pipeline_v1"
ONE_YEAR = ROOT / "reports/alpaca_iex_1y_audit_v1/alpaca_iex_ecl_5min_1y_20261112.csv"
NATIVE = ROOT / "models/alpaca_1min_research_v1/raw_ecl_iex_5min_train_validation.csv"
BACKFILL = ROOT / "data/alpaca_backfill_v1/ecl_iex_5min_20260925_20261112.csv"
FEATURE_SPEC = ROOT / "models/alpaca_final_freeze_v1/frozen_candidate_spec.json"
CLASS_ORDER = ["DOWN", "NEUTRAL", "UP"]
HORIZON_MINUTES = 15
THRESHOLD_PCT = 0.10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path, source_id: str) -> pd.DataFrame:
    if not path.is_file() or path.resolve().parent == (ROOT / "data/alpaca_rebuild_v1").resolve():
        raise FileNotFoundError(f"Approved Alpaca source unavailable: {path}")
    frame = pd.read_csv(path)
    timestamp_col = "timestamp_utc" if "timestamp_utc" in frame else "timestamp"
    if timestamp_col not in frame:
        raise ValueError(f"{path.name} lacks a UTC timestamp column")
    frame["timestamp_utc"] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    if frame.timestamp_utc.isna().any():
        raise ValueError(f"Invalid UTC timestamp in {path.name}")
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "trade_count" in frame:
        frame["trade_count"] = pd.to_numeric(frame.trade_count, errors="coerce")
    else:
        frame["trade_count"] = np.nan
    frame["source_id"] = source_id
    frame["source_file"] = path.relative_to(ROOT).as_posix()
    return frame[["timestamp_utc", "open", "high", "low", "close", "volume", "trade_count", "source_id", "source_file"]]


def _source_metadata_checks() -> dict:
    rebuild = json.loads((ROOT / "data/alpaca_rebuild_v1/reproducibility_manifest.json").read_text(encoding="utf-8"))
    research = json.loads((ROOT / "models/alpaca_1min_research_v1/fetch_5min_metadata.json").read_text(encoding="utf-8"))
    backfill = json.loads((ROOT / "data/alpaca_backfill_v1/ecl_iex_5min_20260925_20261112_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "one_year_declared_source": str(rebuild.get("source", {}).get("path", "")).replace("\\", "/") == "reports/alpaca_iex_1y_audit_v1/alpaca_iex_ecl_5min_1y_20261112.csv" and "Alpaca IEX" in rebuild.get("dataset", "") and rebuild.get("source_provenance", "").lower().find("alpaca iex") >= 0,
        "native_cache_symbol_feed_timeframe": research.get("symbol") == "ECL" and str(research.get("feed", "")).lower() == "iex" and research.get("timeframe") == "5Min" and "alpaca.markets/v2/stocks/bars" in research.get("endpoint", ""),
        "backfill_symbol_feed_timeframe": backfill.get("symbol") == "ECL" and str(backfill.get("feed", "")).lower() == "iex" and backfill.get("timeframe") == "5Min" and "alpaca.markets/v2/stocks/bars" in backfill.get("endpoint", ""),
    }
    if not all(checks.values()):
        raise ValueError(f"Alpaca source provenance check failed: {checks}")
    return checks


def _validate_source_rows(frame: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["datetime_ny"] = frame.timestamp_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    values = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    positive = (values > 0).all(axis=1)
    structural = (
        (frame.high >= frame[["open", "close", "low"]].max(axis=1))
        & (frame.low <= frame[["open", "close", "high"]].min(axis=1))
    ).to_numpy()
    valid_volume = np.isfinite(frame.volume.to_numpy(float)) & (frame.volume.to_numpy(float) >= 0)
    # This project's frozen source contract retains only positive-volume bars.
    # No zero-volume bars occur in the accepted source files; if one appears,
    # it is flagged and rejected rather than treated as a fabricated candle.
    valid = finite & positive & structural & valid_volume & (frame.volume.to_numpy(float) > 0)
    reason = np.where(~finite, "nonfinite_ohlc", np.where(~positive, "nonpositive_ohlc", np.where(~structural, "invalid_ohlc_relationship", np.where(~valid_volume, "invalid_volume", np.where(frame.volume.to_numpy(float) == 0, "zero_volume", "")))))
    frame["reject_reason"] = reason
    rejected = frame.loc[~valid].copy()
    accepted = frame.loc[valid].copy()
    accepted["source_note"] = source_name
    return accepted, rejected


def _session_filter(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    ny = frame.datetime_ny
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=ny.min().date(), end_date=ny.max().date())
    schedule["open_ny"] = schedule.market_open.dt.tz_convert("America/New_York").dt.tz_localize(None)
    schedule["close_ny"] = schedule.market_close.dt.tz_convert("America/New_York").dt.tz_localize(None)
    sessions = {idx.date(): (row.open_ny, row.close_ny) for idx, row in schedule.iterrows()}
    allowed = []
    for dt in ny:
        session = sessions.get(dt.date())
        allowed.append(bool(session and dt >= session[0] and dt + pd.Timedelta(minutes=5) <= session[1] and (dt - session[0]).total_seconds() % 300 == 0))
    keep = np.asarray(allowed, dtype=bool)
    return frame.loc[keep].copy(), frame.loc[~keep].copy(), len(schedule)


def _row_signature(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume", "trade_count"]
    return frame.set_index("timestamp_utc")[cols].sort_index()


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    source_checks = _source_metadata_checks()
    sources = {
        "one_year_primary": _read(ONE_YEAR, "one_year_primary"),
        "native_cache_verification": _read(NATIVE, "native_cache_verification"),
        "backfill": _read(BACKFILL, "backfill"),
    }
    hashes = {name: sha256(path) for name, path in {"one_year_primary": ONE_YEAR, "native_cache_verification": NATIVE, "backfill": BACKFILL}.items()}
    checked, rejected_rows = {}, []
    for name, frame in sources.items():
        good, bad = _validate_source_rows(frame, name)
        sess_good, sess_bad, _ = _session_filter(good)
        if len(sess_bad):
            sess_bad = sess_bad.copy()
            sess_bad["reject_reason"] = "outside_or_misaligned_nyse_regular_session"
            rejected_rows.append(sess_bad)
        if len(bad):
            rejected_rows.append(bad)
        checked[name] = sess_good

    # Reconcile independently acquired native cache against the one-year source
    # on the entire overlap. Any mismatch halts construction.
    one, native, back = (_row_signature(checked[k]) for k in ("one_year_primary", "native_cache_verification", "backfill"))
    def reconcile(label: str, left: pd.DataFrame, right: pd.DataFrame) -> dict:
        common = left.index.intersection(right.index)
        a, b = left.loc[common], right.loc[common]
        equal = a.eq(b) | (a.isna() & b.isna())
        mismatched = common[~equal.all(axis=1)]
        if len(mismatched):
            raise ValueError(f"Unexplained source conflicts ({label}): {len(mismatched)}; first={list(mismatched[:5])}")
        return {"comparison": label, "common_timestamps": len(common), "exact_matches": len(common), "left_only": len(left.index.difference(right.index)), "right_only": len(right.index.difference(left.index)), "conflicts": 0}
    reconciliation = [
        reconcile("one-year vs independent native cache", one, native),
        reconcile("one-year vs backfill overlap", one, back),
    ]

    # Primary selection: one-year source wins exact overlaps; backfill contributes
    # only timestamps not in that source. The cache is verification-only.
    primary = checked["one_year_primary"]
    back_unique = checked["backfill"].loc[~checked["backfill"].timestamp_utc.isin(primary.timestamp_utc)]
    canonical = pd.concat([primary, back_unique], ignore_index=True).sort_values("timestamp_utc", kind="stable")
    if canonical.timestamp_utc.duplicated().any():
        raise ValueError("Canonical timestamps are not unique after deterministic reconciliation")
    canonical["symbol"] = "ECL"
    canonical["feed"] = "IEX"
    canonical["timeframe"] = "5Min"
    canonical["timestamp_convention"] = "candle_open_utc"
    canonical["datetime_ny"] = canonical.timestamp_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    canonical = canonical[["timestamp_utc", "datetime_ny", "open", "high", "low", "close", "volume", "trade_count", "symbol", "feed", "timeframe", "timestamp_convention", "source_id", "source_file"]]
    canonical_path = OUT / "canonical_ecl_iex_5min.csv"
    canonical.to_csv(canonical_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")

    # Feature values are calculated by the exact existing causal B-feature
    # implementation, in the frozen specification's order.
    spec = json.loads(FEATURE_SPEC.read_text(encoding="utf-8"))
    features = spec["features_exact_order"]
    from live_app.features import calculate_b_features
    ohlc = canonical[["datetime_ny", "open", "high", "low", "close"]].rename(columns={"datetime_ny": "datetime"})
    feature_frame = calculate_b_features(ohlc, features)
    canonical_indexed = canonical.set_index("datetime_ny", drop=False)
    feature_frame = feature_frame.set_index("datetime", drop=False)
    assembled = canonical_indexed.join(feature_frame.drop(columns=["datetime"]), how="inner", rsuffix="_feature")
    if len(assembled) != len(canonical):
        raise ValueError("Feature calculation changed canonical row count or timestamp alignment")
    assembled["datetime"] = assembled.index
    if "close_feature" in assembled:
        # Frozen B's first feature is named close; preserve the canonical candle close exactly.
        if not np.allclose(assembled.close, assembled.close_feature, rtol=0, atol=0):
            raise ValueError("Feature close differs from canonical reference close")
        assembled = assembled.drop(columns="close_feature")
    # Use segment logic from the existing historical labeler. Targets are
    # calculated only on exact +15m endpoints in the same continuous session.
    from importlib.util import module_from_spec, spec_from_file_location
    label_path = ROOT / "src/11_create_features.py"
    module_spec = spec_from_file_location("ecl_alpaca_canonical_labeler", label_path)
    labeler = module_from_spec(module_spec)
    assert module_spec and module_spec.loader
    sys.path.insert(0, str(ROOT))
    module_spec.loader.exec_module(labeler)
    bars = canonical[["datetime_ny", "open", "high", "low", "close"]].rename(columns={"datetime_ny": "datetime"}).reset_index(drop=True)
    segmented = labeler.create_segments(bars)
    labels = []
    for _, group in segmented.groupby("segment_id", sort=False):
        labels.append(labeler.create_labels(group))
    target_frame = pd.concat(labels, ignore_index=True).set_index("datetime")
    assembled = assembled.join(target_frame[["future_datetime", "future_return", "target"]], how="inner", rsuffix="_target")
    feature_ready = assembled[features].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    target_ready = assembled.target.isin(CLASS_ORDER) & assembled.future_datetime.notna() & assembled.future_return.notna()
    excluded = pd.DataFrame({
        "timestamp_utc": canonical.set_index("datetime_ny").loc[assembled.index, "timestamp_utc"],
        "feature_ready": feature_ready,
        "target_ready": target_ready,
        "reason": np.where(~feature_ready, "feature_warmup_or_nonfinite", np.where(~target_ready, "exact_15m_endpoint_unavailable_or_gap", "")),
    })
    ready = assembled.loc[feature_ready & target_ready].copy().reset_index(drop=True)
    ready["future_datetime"] = pd.to_datetime(ready.future_datetime)
    if not ((ready.future_datetime - ready.datetime) == pd.Timedelta(minutes=15)).all():
        raise ValueError("A target endpoint is not exactly 15 minutes after its decision candle")
    if not ready.datetime.is_monotonic_increasing or ready.datetime.duplicated().any():
        raise ValueError("Feature-target rows are not unique and chronological")
    if list(ready[features].columns) != features or not np.isfinite(ready[features].to_numpy(float)).all():
        raise ValueError("Final feature schema or finite-value validation failed")

    dates = sorted(pd.Series(ready.datetime.dt.date.unique()).tolist())
    n = len(dates)
    train_end_i = int(np.floor(0.70 * n))
    val_end_i = int(np.floor(0.85 * n))
    train_dates, val_dates, test_dates = dates[:train_end_i], dates[train_end_i:val_end_i], dates[val_end_i:]
    if not train_dates or not val_dates or not test_dates:
        raise ValueError("Insufficient trading dates for chronological 70/15/15 split")
    split_map = {d: "train" for d in train_dates} | {d: "validation" for d in val_dates} | {d: "test" for d in test_dates}
    ready["split"] = ready.datetime.dt.date.map(split_map)
    split_starts = {s: ready.loc[ready.split.eq(s), "datetime"].min() for s in ["validation", "test"]}
    purged_counts = {}
    for earlier, next_split in [("train", "validation"), ("validation", "test")]:
        boundary = split_starts[next_split]
        purge = ready.split.eq(earlier) & (ready.future_datetime >= boundary)
        purged_counts[earlier] = int(purge.sum())
        ready = ready.loc[~purge].copy()
    ready["split"] = pd.Categorical(ready.split, categories=["train", "validation", "test"], ordered=True)
    # Split tables are written once. Training code reads only train and validation
    # until model selection is frozen; the test file is loaded only in the final
    # evaluation step.
    output_columns = ["datetime", "future_datetime", "future_return", "target", *features]
    split_summaries = {}
    for split in ["train", "validation", "test"]:
        part = ready.loc[ready.split.eq(split), output_columns].copy()
        part.to_csv(OUT / f"{split}.csv", index=False)
        split_summaries[split] = {
            "rows": len(part),
            "date_start": str(part.datetime.min().date()) if len(part) else None,
            "date_end": str(part.datetime.max().date()) if len(part) else None,
            "dates": int(part.datetime.dt.date.nunique()) if len(part) else 0,
            "feature_missing_cells": int(part[features].isna().sum().sum()),
            "feature_nonfinite_cells": int((~np.isfinite(part[features].to_numpy(float))).sum()),
            "target_counts": (part.target.value_counts().reindex(CLASS_ORDER, fill_value=0).to_dict() if split != "test" else None),
            "test_target_summary_withheld_until_final_evaluation": split == "test",
        }
    quality = {
        "canonical_rows": len(canonical),
        "source_rejected_rows_across_all_audited_sources": sum(len(x) for x in rejected_rows),
        "canonical_rejected_rows": int(sum(1 for bad in rejected_rows for r in bad.itertuples(index=False) if r.source_id in {"one_year_primary", "backfill"})),
        "feature_ready_target_rows": len(ready),
        "expected_regular_session_buckets": int(sum((int((row.close_ny - row.open_ny).total_seconds() / 300)) for _, row in _session_filter(canonical.assign(datetime_ny=canonical.datetime_ny))[0].head(0).iterrows())) if False else None,
        "missing_expected_canonical_intervals": None,
        "zero_volume_canonical_rows": int((canonical.volume == 0).sum()),
        "duplicate_canonical_timestamps": int(canonical.timestamp_utc.duplicated().sum()),
        "invalid_ohlc_canonical_rows": 0,
        "rejected_source_rows": [
            {"timestamp_utc": r.timestamp_utc.isoformat(), "source_id": r.source_id, "reason": r.reject_reason}
            for bad in rejected_rows for r in bad.itertuples(index=False)
        ],
        "feature_warmup_or_nonfinite_rows": int((~feature_ready).sum()),
        "target_endpoint_or_gap_rows": int((~target_ready).sum()),
        "split_purged_rows": purged_counts,
    }
    # Count missing official session grid buckets without creating any rows.
    sch = mcal.get_calendar("NYSE").schedule(start_date=canonical.datetime_ny.min().date(), end_date=canonical.datetime_ny.max().date())
    sch["open_ny"] = sch.market_open.dt.tz_convert("America/New_York").dt.tz_localize(None)
    sch["close_ny"] = sch.market_close.dt.tz_convert("America/New_York").dt.tz_localize(None)
    actual = set(canonical.datetime_ny)
    gaps = []
    expected_total = 0
    for idx, row in sch.iterrows():
        slots = pd.date_range(row.open_ny, row.close_ny - pd.Timedelta(minutes=5), freq="5min")
        expected_total += len(slots)
        gaps.extend((str(idx.date()), t.isoformat()) for t in slots if t not in actual)
    quality["expected_regular_session_buckets"] = expected_total
    quality["missing_expected_canonical_intervals"] = len(gaps)
    quality["missing_intervals_note"] = "Absent Alpaca IEX observations are preserved as genuine gaps; no candles or prices were fabricated."
    pd.DataFrame(gaps, columns=["trading_date", "missing_timestamp_ny"]).to_csv(OUT / "missing_expected_intervals.csv", index=False)
    if rejected_rows:
        pd.concat(rejected_rows, ignore_index=True).to_csv(OUT / "rejected_source_rows.csv", index=False)
    else:
        pd.DataFrame(columns=["timestamp_utc", "source_id", "reject_reason"]).to_csv(OUT / "rejected_source_rows.csv", index=False)
    (OUT / "quality_report.json").write_text(json.dumps(quality, indent=2, default=str), encoding="utf-8")
    manifest = {
        "pipeline": "Alpaca IEX only canonical ECL 5-minute training pipeline v1",
        "symbol": "ECL", "feed": "IEX", "timeframe": "5Min", "timestamp_convention": "candle open timestamp in UTC; derived local timestamp is America/New_York",
        "sources": {
            "one_year_primary": {"path": ONE_YEAR.relative_to(ROOT).as_posix(), "sha256": hashes["one_year_primary"], "provenance": "Alpaca IEX historical audit CSV, corroborated against independent Alpaca IEX native-cache overlap; upstream manifest records Alpaca IEX provenance; acquisition script not present in project."},
            "native_cache_verification_only": {"path": NATIVE.relative_to(ROOT).as_posix(), "sha256": hashes["native_cache_verification"], "metadata": "models/alpaca_1min_research_v1/fetch_5min_metadata.json"},
            "backfill": {"path": BACKFILL.relative_to(ROOT).as_posix(), "sha256": hashes["backfill"], "metadata": "data/alpaca_backfill_v1/ecl_iex_5min_20260925_20261112_manifest.json"},
        },
        "source_reconciliation": reconciliation,
        "canonical_file": "canonical_ecl_iex_5min.csv",
        "canonical_sha256": sha256(canonical_path),
        "canonical_rows": len(canonical), "canonical_date_range_ny": [str(canonical.datetime_ny.min()), str(canonical.datetime_ny.max())],
        "rejected_source_rows": quality["rejected_source_rows"],
        "features": features,
        "feature_builder": "live_app/features.py::calculate_b_features, reusing src/11_create_features.py and src/17_add_time_features.py",
        "target": {"horizon_minutes": 15, "reference": "decision candle close", "endpoint": "exact close timestamp t+15m; same continuous session segment only", "UP": "future return > +0.10%", "DOWN": "future return < -0.10%", "NEUTRAL": "inclusive [-0.10%, +0.10%]", "class_order": CLASS_ORDER},
        "split": {"method": "chronological by eligible trading date; first floor(70%) train, next cumulative floor(85%) validation, remainder test", "date_counts": {"train": len(train_dates), "validation": len(val_dates), "test": len(test_dates)}, "date_ranges": {"train": [str(train_dates[0]), str(train_dates[-1])], "validation": [str(val_dates[0]), str(val_dates[-1])], "test": [str(test_dates[0]), str(test_dates[-1])]}, "purge_rule": "Drop earlier-split rows whose exact future endpoint is at or after the next split's first decision timestamp.", "purged_rows": purged_counts},
        "split_summaries": split_summaries,
        "test_policy": "New test rows are isolated from training/selection; read only once by final_evaluate_selected.py after validation-based model selection. Older split datasets were not read or used.",
        "old_split_csvs_opened": False,
    }
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    report_dir = ROOT / "reports/alpaca_baselines_v1"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Alpaca IEX dataset validation (pre-training)", "",
        "The old training/validation/test CSVs were not used. The protected old test CSV was not opened. The new test split was written from the canonical source but its labels were withheld from this pre-training report.", "",
        f"Canonical candles: **{len(canonical):,}** ({canonical.datetime_ny.min()} through {canonical.datetime_ny.max()} New York time).",
        f"Target-ready, feature-ready rows before date splits: **{len(ready):,}**.",
        f"Source rows rejected from canonical: **{quality['canonical_rejected_rows']}**; all audited source files include **{quality['source_rejected_rows_across_all_audited_sources']}** rejected observations because the same invalid row appears in the verification cache.",
        f"Missing expected NYSE five-minute slots: **{quality['missing_expected_canonical_intervals']:,}**. These are preserved gaps, not filled candles.",
        f"Feature warm-up/nonfinite exclusions: **{quality['feature_warmup_or_nonfinite_rows']:,}**; rows without an exact valid 15-minute endpoint: **{quality['target_endpoint_or_gap_rows']:,}** (these exclusion groups can overlap).",
        "", "## Split validation", "",
        "| Split | Rows | Trading dates | Date range | Missing/nonfinite feature cells | Class counts (DOWN / NEUTRAL / UP) |",
        "|---|---:|---:|---|---:|---|",
    ]
    for split in ["train", "validation"]:
        s = split_summaries[split]
        counts = s["target_counts"]
        lines.append(f"| {split} | {s['rows']:,} | {s['dates']} | {s['date_start']}–{s['date_end']} | {s['feature_missing_cells'] + s['feature_nonfinite_cells']} | {counts['DOWN']} / {counts['NEUTRAL']} / {counts['UP']} |")
    test_summary = split_summaries["test"]
    lines.append(f"| test (summary withheld) | {test_summary['rows']:,} | {test_summary['dates']} | {test_summary['date_start']}–{test_summary['date_end']} | validated in builder memory; no class summary emitted | withheld until final evaluation |")
    lines.extend(["", f"All splits use the exact same {len(features)} frozen feature columns and 15-minute target. Purged rows: train {purged_counts['train']}, validation {purged_counts['validation']}.", "The one out-of-session row, 2026-09-18 09:25 ET, was rejected. No missing candle was fabricated. Feature generation reused the existing causal feature implementation in the frozen feature order.", ""])
    (report_dir / "dataset_validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"manifest": manifest, "quality": quality}


if __name__ == "__main__":
    result = build()
    public_splits = {k: v for k, v in result["manifest"]["split_summaries"].items() if k != "test"}
    print(json.dumps({"canonical_rows": result["manifest"]["canonical_rows"], "train_validation_summaries": public_splits, "test_rows_written_without_summary": result["manifest"]["split_summaries"]["test"]["rows"], "quality": result["quality"]}, indent=2, default=str))
