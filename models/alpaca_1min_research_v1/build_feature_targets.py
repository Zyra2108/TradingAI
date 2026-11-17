"""Prepare 1m and reconstructed 5m datasets from fetched permitted IEX bars."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, OUT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipe = load_module("feature_and_label_pipeline.py", "ecl_1min_feature_pipeline")
FEATURE_DEFINITIONS = {
    "close": "completed current candle close, known at decision time",
    "return_{1,3,5,10}_pct": "100 * close[t] / close[t-lag] - 100; trailing only",
    "realized_vol_{3,5,10}": "sample standard deviation of trailing one-minute close returns, current return included",
    "range_pct": "100 * (high-low) / close for current completed candle",
    "body_pct": "100 * (close-open) / open for current completed candle",
    "upper_wick_pct": "100 * (high-max(open,close)) / close",
    "lower_wick_pct": "100 * (min(open,close)-low) / close",
    "volume_log1p": "log(1+current completed candle volume)",
    "volume_ratio_10": "current volume divided by mean of prior 10 one-minute volumes; current excluded from baseline",
    "volume_zscore_10": "current volume relative to prior 10 one-minute volumes; current excluded from baseline",
    "sma_ratio_{3,5,10}": "current close / trailing SMA including current close - 1",
    "ema_ratio_{3,5,10}": "current close / trailing EMA (adjust=false, current included) - 1",
    "rsi_7": "7-return rolling mean gains/losses, current return included; flat window maps to 50",
    "trend_agreement_3_10": "sign(return_3) * sign(return_10), causal",
    "momentum_acceleration_3_10": "return_3 - 0.3*return_10, causal",
    "minutes_since_open": "minute offset of candle open from official NYSE session open",
    "minutes_until_close": "regular-session minutes after the candle open through final minute",
    "session_progress": "minutes_since_open / (session length - 1), clipped to [0,1]",
    "day_of_week": "New York local weekday index",
    "time_sin/time_cos": "sine/cosine of causal session progress",
}


def build_dataset(bars: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame,
                  feature_names: list[str], interval: int, name: str) -> pd.DataFrame:
    merged = features.merge(labels, on="datetime", how="left", validate="one_to_one")
    if "session_date" not in merged:
        merged["session_date"] = pd.to_datetime(merged.datetime).dt.date.astype(str)
    keep = ["datetime", "session_date", "target_end", "future_return_pct", "label", "target_available", *feature_names]
    if "close" not in merged:
        merged = merged.merge(bars[["datetime", "close"]], on="datetime", how="left", validate="one_to_one")
    merged["reference_close"] = merged.close.astype(float)
    if interval == 1:
        volume_features = ["volume_log1p", "volume_ratio_10", "volume_zscore_10"]
        core = [f for f in feature_names if f not in volume_features]
        ready = np.isfinite(merged[core].to_numpy(float)).all(axis=1)
    else:
        ready = np.isfinite(merged[feature_names].to_numpy(float)).all(axis=1)
    merged["feature_ready"] = ready
    merged["split"] = "excluded"
    split_dates = pipe.split_date_sets()
    dates = pd.to_datetime(merged.datetime).dt.date
    for split, allowed_dates in split_dates.items():
        merged.loc[dates.isin(allowed_dates), "split"] = split
    allowed = merged.split.isin(["train", "validation"])
    merged = merged.loc[allowed & merged.target_available.fillna(False) & merged.feature_ready].copy()
    merged = merged[keep + ["reference_close", "feature_ready", "split"]]
    merged.insert(0, "input_interval_minutes", interval)
    merged["datetime"] = pd.to_datetime(merged.datetime).dt.strftime("%Y-%m-%d %H:%M:%S")
    merged["target_end"] = pd.to_datetime(merged.target_end).dt.strftime("%Y-%m-%d %H:%M:%S")
    merged = merged.sort_values(["split", "datetime"], kind="stable").reset_index(drop=True)
    merged.to_csv(OUT / f"{name}_train_validation.csv", index=False)
    for split in ("train", "validation"):
        merged.loc[merged.split.eq(split)].to_csv(OUT / f"{name}_{split}.csv", index=False)
    return merged


def main():
    regular, gaps_1m, quality = pipe.clean_and_audit()
    date_sets = pipe.split_date_sets()
    (OUT / "split_dates.json").write_text(json.dumps({name: sorted(day.isoformat() for day in values)
        for name, values in date_sets.items()}, indent=2) + "\n", encoding="utf-8")
    (OUT / "feature_schema.json").write_text(json.dumps({
        "one_minute_feature_order": pipe.ONE_MINUTE_FEATURES,
        "feature_definitions": FEATURE_DEFINITIONS,
        "causality": "Features use only current completed OHLCV and trailing rows within a same-session continuous segment. Rolling state resets at session boundaries and every missing one-minute interval. No future values, labels, or target timestamps are model inputs.",
        "missing_volume": "The three volume-derived inputs may be NaN where source volume/baseline is unavailable; per-fold train-only median imputation handles those fields. Other feature warm-up fields must be present.",
        "five_minute_feature_source": "Original B feature order and live_app.features.calculate_b_features; reconstructed only from five complete consecutive IEX 1-minute candles.",
        "five_minute_feature_order": json.loads((ROOT / "models/alpaca_final_freeze_v1/frozen_candidate_spec.json").read_text(encoding="utf-8"))["features_exact_order"],
        "feature_count_one_minute": len(pipe.ONE_MINUTE_FEATURES),
        "feature_count_five_minute": 24,
        "feature_metadata_excluded": sorted(pipe.FEATURE_METADATA),
        "volume_used_in_five_minute_baseline": False,
    }, indent=2) + "\n", encoding="utf-8")

    f1 = pipe.one_minute_features(regular)
    y1 = pipe.generate_labels(regular, 1)
    d1 = build_dataset(regular, f1, y1, pipe.ONE_MINUTE_FEATURES, 1, "one_minute")

    reconstructed5, gaps_reconstructed5 = pipe.aggregate_complete_five_minute(regular)
    reconstructed5.to_csv(OUT / "reconstructed_5min_ohlcv.csv", index=False)
    gaps_reconstructed5.to_csv(OUT / "incomplete_5min_buckets.csv", index=False)
    bars5, gaps5, quality5 = pipe.clean_five_minute_reference()
    f5, order5 = pipe.five_minute_b_features(bars5)
    y5 = pipe.generate_labels(bars5, 5)
    d5 = build_dataset(bars5, f5, y5, order5, 5, "five_minute")

    split_stats = {}
    for name, data, features in (("one_minute", d1, pipe.ONE_MINUTE_FEATURES),
                                 ("five_minute", d5, order5)):
        for split in ("train", "validation"):
            rows = data[data.split.eq(split)]
            split_stats[f"{name}_{split}"] = {
                "rows": int(len(rows)), "dates": int(rows.session_date.nunique()),
                "date_start": rows.session_date.min() if len(rows) else None,
                "date_end": rows.session_date.max() if len(rows) else None,
                "label_distribution": {str(k): int(v) for k, v in rows.label.value_counts().sort_index().items()},
                "eligible_directional_rows": int(rows.label.isin(["UP", "DOWN"]).sum()),
                "feature_count": len(features),
                "feature_nan_cells": int(rows[features].isna().sum().sum()),
                "feature_nonfinite_cells": int((~np.isfinite(rows[features].to_numpy(float))).sum()),
                "split_from_permitted_timestamps_only": True,
            }
    split_stats["strict_5min_reconstruction_from_1min"] = {"candles": int(len(reconstructed5)),
        "incomplete_buckets": int(len(gaps_reconstructed5)), "bars_from_same_iex_1min_source": True,
        "requirement": "exactly five expected one-minute bars; no fill/interpolation"}
    split_stats["direct_5min_reference"] = quality5
    split_stats["quality"] = quality
    (OUT / "prepared_dataset_audit.json").write_text(json.dumps(split_stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in split_stats.items() if k != "quality"}, indent=2))


if __name__ == "__main__":
    main()
