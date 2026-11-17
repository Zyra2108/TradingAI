from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT / "reports" / "alpaca_sequence_research_v1"
DATA_DIR = PROJECT / "data" / "alpaca_ml_pipeline_v1"
SOURCE_CSV = PROJECT / "models" / "alpaca_1min_research_v1" / "raw_ecl_iex_5min_train_validation.csv"
SOURCE_METADATA = SOURCE_CSV.with_name("fetch_5min_metadata.json")
TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
FOLDS_CSV = PROJECT / "reports" / "alpaca_binary_research_v3" / "walk_forward_folds.csv"
BASELINE_OOF_CSV = PROJECT / "reports" / "alpaca_binary_research_v3" / "walk_forward_oof_predictions.csv"
BASELINE_VALIDATION_CSV = PROJECT / "reports" / "alpaca_binary_research_v3" / "validation_predictions.csv"
BASELINE_CONFIG = PROJECT / "models" / "alpaca_binary_research_v3" / "configuration.json"
SEQUENCE_LENGTHS = (6, 12, 24, 36)
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
CLASSES = ("DOWN", "UP")
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_approved_path(path: Path) -> Path:
    allowed = {
        p.resolve()
        for p in (
            SOURCE_CSV,
            SOURCE_METADATA,
            TRAIN_CSV,
            VALIDATION_CSV,
            FOLDS_CSV,
            BASELINE_OOF_CSV,
            BASELINE_VALIDATION_CSV,
            BASELINE_CONFIG,
        )
    }
    resolved = path.resolve()
    if resolved not in allowed:
        raise ValueError(f"Unapproved research input path: {resolved}")
    return resolved


def read_approved_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(assert_approved_path(path), **kwargs)


def read_approved_json(path: Path) -> dict[str, Any]:
    return json.loads(assert_approved_path(path).read_text(encoding="utf-8"))


def load_and_audit_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metadata = read_approved_json(SOURCE_METADATA)
    if (metadata.get("symbol", "").upper(), metadata.get("feed", "").lower(), metadata.get("timeframe", "")) != (
        "ECL",
        "iex",
        "5Min",
    ):
        raise ValueError("The approved raw cache metadata is not ECL Alpaca IEX 5Min.")
    if metadata.get("permitted_new_york_date_range") != ["2025-11-13", "2026-09-24"]:
        raise ValueError("The raw cache metadata does not match the permitted pre-test range.")
    source_hash = sha256_file(SOURCE_CSV)
    if source_hash != metadata.get("raw_csv_sha256"):
        raise ValueError("The permitted raw cache hash does not match its fetch metadata.")

    raw = read_approved_csv(SOURCE_CSV)
    required_raw = {"timestamp_utc", *OHLCV_COLUMNS}
    if not required_raw.issubset(raw.columns):
        raise ValueError(f"Raw cache is missing columns: {sorted(required_raw - set(raw.columns))}")
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="raise")
    raw = raw.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    raw["datetime"] = raw["timestamp_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    raw["session_date"] = raw["datetime"].dt.date
    if raw["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps in approved raw cache.")
    if raw["timestamp_utc"].min() < pd.Timestamp("2025-11-13", tz="UTC"):
        raise ValueError("Raw cache starts before permitted range.")
    if raw["timestamp_utc"].max() > pd.Timestamp("2026-09-24 23:59:59", tz="UTC"):
        raise ValueError("Raw cache contains observations after the permitted end date.")
    numeric = raw[list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite OHLCV values found in raw cache.")
    invalid_ohlc = (
        (numeric["low"] > numeric[["open", "close"]].min(axis=1))
        | (numeric["high"] < numeric[["open", "close"]].max(axis=1))
        | (numeric["low"] > numeric["high"])
        | (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (numeric["volume"] < 0)
    )
    if invalid_ohlc.any():
        raise ValueError(f"Invalid raw OHLCV rows found: {int(invalid_ohlc.sum())}")
    raw[list(OHLCV_COLUMNS)] = numeric

    train = read_approved_csv(TRAIN_CSV)
    validation = read_approved_csv(VALIDATION_CSV)
    folds = read_approved_csv(FOLDS_CSV)
    for split_name, frame, lo, hi in (
        ("train", train, "2025-11-13", "2026-08-05"),
        ("validation", validation, "2026-08-06", "2026-09-24"),
    ):
        needed = {"datetime", "future_datetime", "target"}
        if not needed.issubset(frame.columns):
            raise ValueError(f"{split_name} split is missing label/timestamp columns.")
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        frame["future_datetime"] = pd.to_datetime(frame["future_datetime"], errors="raise")
        dates = frame["datetime"].dt.strftime("%Y-%m-%d")
        if dates.min() < lo or dates.max() > hi:
            raise ValueError(f"{split_name} split has rows outside the approved range.")
        if not set(frame["target"].unique()).issubset({"DOWN", "NEUTRAL", "UP"}):
            raise ValueError(f"Unexpected target class in {split_name} split.")
        if frame["datetime"].duplicated().any() or not frame["datetime"].is_monotonic_increasing:
            raise ValueError(f"{split_name} timestamps are not unique and chronological.")
        actual_horizon = (frame["future_datetime"] - frame["datetime"]).dt.total_seconds() / 60
        if not actual_horizon.eq(15).all():
            raise ValueError(f"{split_name} target horizon is not exactly 15 minutes.")
        if (frame["datetime"].dt.date != frame["future_datetime"].dt.date).any():
            raise ValueError(f"{split_name} contains a target crossing the New York session date.")
        raw_time_set = set(raw["datetime"])
        if not frame["datetime"].isin(raw_time_set).all() or not frame["future_datetime"].isin(raw_time_set).all():
            raise ValueError(f"{split_name} has a decision or exact target endpoint absent from the approved raw cache.")
        if "future_return" not in frame:
            raise ValueError(f"{split_name} has no saved future_return for label verification.")
        expected = np.where(frame["future_return"] > 0.10, "UP", np.where(frame["future_return"] < -0.10, "DOWN", "NEUTRAL"))
        if not np.array_equal(expected, frame["target"].to_numpy(dtype=str)):
            raise ValueError(f"{split_name} target labels do not match the existing strict +/-0.10% definition.")

    if len(folds) != 4:
        raise ValueError("Expected the established four chronological folds.")
    return raw, train, validation, {"metadata": metadata, "folds": folds, "invalid_ohlc": int(invalid_ohlc.sum())}


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Causally normalize OHLC to the current close and compress volume."""
    values = np.asarray(sequence, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("Each sequence must have shape (time, 5 OHLCV channels).")
    last_close = values[-1, 3]
    if not np.isfinite(last_close) or last_close <= 0:
        raise ValueError("Sequence endpoint close must be finite and positive.")
    values[:, :4] = 100.0 * (values[:, :4] / last_close - 1.0)
    if np.any(values[:, 4] < 0) or not np.isfinite(values).all():
        raise ValueError("Sequence has invalid volume or non-finite values.")
    values[:, 4] = np.log1p(values[:, 4])
    return values.astype(np.float32)


def build_sequence_samples(raw: pd.DataFrame, labels: pd.DataFrame, length: int) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Build windows ending exactly at each labeled candle; reset at dates/gaps."""
    if length not in SEQUENCE_LENGTHS:
        raise ValueError(f"Unsupported sequence length: {length}")
    raw_times = pd.DatetimeIndex(raw["datetime"])
    raw_values = raw[list(OHLCV_COLUMNS)].to_numpy(dtype=np.float64)
    positions = {timestamp: pos for pos, timestamp in enumerate(raw_times)}
    meta = labels[["datetime", "future_datetime", "target"]].copy().reset_index(drop=True)
    windows: list[np.ndarray] = []
    keep: list[int] = []
    counts = {"total_labeled_rows": len(meta), "missing_raw_timestamp": 0, "insufficient_contiguous_session_warmup": 0, "eligible_rows": 0}
    for row, timestamp in enumerate(meta["datetime"]):
        end = positions.get(timestamp)
        if end is None:
            counts["missing_raw_timestamp"] += 1
            continue
        start = end - length + 1
        if start < 0:
            counts["insufficient_contiguous_session_warmup"] += 1
            continue
        times = raw_times[start : end + 1]
        if len(times) != length or times[-1] != timestamp:
            counts["insufficient_contiguous_session_warmup"] += 1
            continue
        if times[0].date() != timestamp.date() or any(
            times[i] - times[i - 1] != pd.Timedelta(minutes=5) or times[i].date() != timestamp.date()
            for i in range(1, length)
        ):
            counts["insufficient_contiguous_session_warmup"] += 1
            continue
        try:
            normalized = normalize_sequence(raw_values[start : end + 1])
        except ValueError:
            counts["insufficient_contiguous_session_warmup"] += 1
            continue
        keep.append(row)
        windows.append(normalized)
    counts["eligible_rows"] = len(keep)
    result = meta.iloc[keep].reset_index(drop=True)
    sequences = np.stack(windows) if windows else np.empty((0, length, 5), dtype=np.float32)
    return result, sequences, counts


def binary_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    p_up: np.ndarray,
    all_targets: np.ndarray | None = None,
    all_predictions: np.ndarray | None = None,
    possible_directional_rows: int | None = None,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=str)
    predicted = np.asarray(predicted, dtype=str)
    p_up = np.asarray(p_up, dtype=float)
    if len(actual) == 0:
        return {"n_directional": 0}
    p = np.column_stack([1.0 - p_up, p_up])
    cm = confusion_matrix(actual, predicted, labels=list(CLASSES))
    classes = {}
    for i, name in enumerate(CLASSES):
        classes[name] = {
            "precision": float(precision_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[i]),
            "recall": float(recall_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[i]),
            "f1": float(f1_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[i]),
            "support": int(np.sum(actual == name)),
        }
    out: dict[str, Any] = {
        "n_directional": int(len(actual)),
        "actual_down": int(np.sum(actual == "DOWN")),
        "actual_up": int(np.sum(actual == "UP")),
        "prediction_down": int(np.sum(predicted == "DOWN")),
        "prediction_up": int(np.sum(predicted == "UP")),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=list(CLASSES), average="macro", zero_division=0)),
        "directional_macro_precision": float(precision_score(actual, predicted, labels=list(CLASSES), average="macro", zero_division=0)),
        "log_loss": float(log_loss(actual, p, labels=list(CLASSES))),
        "confusion_matrix_actual_rows_predicted_columns_down_up": cm.tolist(),
        "class_metrics": classes,
    }
    if all_targets is not None:
        all_targets = np.asarray(all_targets, dtype=str)
        out["all_label_rows_included"] = int(len(all_targets))
        out["neutral_rows"] = int(np.sum(all_targets == "NEUTRAL"))
        denominator = int(possible_directional_rows) if possible_directional_rows is not None else int(np.sum(np.isin(all_targets, CLASSES)))
        out["possible_directional_rows_in_source_partition"] = denominator
        out["sequence_eligibility_coverage_of_directional_rows"] = float(len(actual) / denominator) if denominator else 0.0
        out["emitted_signal_coverage_of_eligible_directional_rows"] = 1.0
        if all_predictions is None or len(all_predictions) != len(all_targets):
            raise ValueError("all_predictions matching all_targets is required for all-candle operational accuracy.")
        all_predictions = np.asarray(all_predictions, dtype=str)
        out["all_candle_operational_accuracy_neutral_wrong"] = float(np.sum((all_targets != "NEUTRAL") & (all_predictions == all_targets)) / len(all_targets)) if len(all_targets) else 0.0
    return out


def fit_channel_scaler(train_x: np.ndarray) -> StandardScaler:
    if train_x.ndim != 3 or len(train_x) == 0:
        raise ValueError("Scaler needs a non-empty sequence training array.")
    scaler = StandardScaler()
    scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
    return scaler


def chronological_fold_masks(frame: pd.DataFrame, train_end: Any, validation_start: Any, validation_end: Any) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Return expanding train/evaluation masks with the forward horizon purged."""
    train_end = pd.Timestamp(train_end).date()
    validation_start = pd.Timestamp(validation_start).date()
    validation_end = pd.Timestamp(validation_end).date()
    dates = frame["datetime"].dt.date
    validation_mask = (dates >= validation_start) & (dates <= validation_end)
    if not validation_mask.any():
        raise ValueError("Chronological fold has no eligible evaluation samples.")
    first_validation_time = frame.loc[validation_mask, "datetime"].min()
    train_mask = (dates <= train_end) & (frame["future_datetime"] < first_validation_time) & frame["target"].isin(CLASSES)
    return train_mask.to_numpy(), validation_mask.to_numpy(), first_validation_time


def transform_sequences(scaler: StandardScaler, sequences: np.ndarray) -> np.ndarray:
    if len(sequences) == 0:
        return sequences.astype(np.float32)
    shape = sequences.shape
    return scaler.transform(sequences.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)


def _tensorflow():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    return tf


def build_cnn(length: int, seed: int):
    tf = _tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    l2 = tf.keras.regularizers.l2(0.01)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(length, len(OHLCV_COLUMNS))),
            tf.keras.layers.Conv1D(8, 3, padding="causal", activation="relu", kernel_regularizer=l2),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=l2),
            tf.keras.layers.Dropout(0.20, seed=seed),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy")
    return model


def train_cnn(length: int, x: np.ndarray, y: np.ndarray, seed: int):
    model = build_cnn(length, seed)
    y_numeric = np.asarray(y == "UP", dtype=np.float32)
    start = time.perf_counter()
    model.fit(x, y_numeric, epochs=20, batch_size=32, shuffle=False, verbose=0)
    return model, time.perf_counter() - start


def evaluate(pred_frame: pd.DataFrame, group_cols: list[str], output_path: Path) -> pd.DataFrame:
    rows = []
    for keys, group in pred_frame.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        all_rows = group
        directional = all_rows[all_rows["actual_binary"].isin(CLASSES)]
        if directional.empty:
            continue
        metric = binary_metrics(
            directional["actual_binary"].to_numpy(),
            directional["prediction"].to_numpy(),
            directional["p_up"].to_numpy(),
            all_targets=all_rows["actual_target"].to_numpy(),
            all_predictions=all_rows["prediction"].to_numpy(),
            possible_directional_rows=int(all_rows["possible_directional_rows"].iloc[0]) if "possible_directional_rows" in all_rows else None,
        )
        rows.append({**dict(zip(group_cols, keys)), **metric})
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def _make_sequence_key(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")


def run() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    raw, train, validation, audit = load_and_audit_inputs()
    train_labels = train[["datetime", "future_datetime", "target"]].copy()
    validation_labels = validation[["datetime", "future_datetime", "target"]].copy()
    fold_table: pd.DataFrame = audit["folds"]
    folds = []
    for row in fold_table.to_dict("records"):
        folds.append({
            "fold": int(row["fold"]),
            "train_end": pd.Timestamp(row["train_end"]).date(),
            "validation_start": pd.Timestamp(row["validation_start"]).date(),
            "validation_end": pd.Timestamp(row["validation_end"]).date(),
        })

    source_cache_by_len: dict[int, dict[str, Any]] = {}
    eligibility_rows = []
    split_sequences: dict[tuple[str, int], tuple[pd.DataFrame, np.ndarray]] = {}
    for split_name, frame in (("train", train_labels), ("validation", validation_labels)):
        for length in SEQUENCE_LENGTHS:
            eligible, sequences, counts = build_sequence_samples(raw, frame, length)
            eligible["split"] = split_name
            split_sequences[(split_name, length)] = (eligible, sequences)
            source_cache_by_len.setdefault(length, {})[split_name] = counts
            directional_total = int(frame["target"].isin(CLASSES).sum())
            directional_eligible = int(eligible["target"].isin(CLASSES).sum())
            eligibility_rows.append({
                "split": split_name,
                "sequence_length": length,
                **counts,
                "directional_total": directional_total,
                "directional_eligible": directional_eligible,
                "eligible_directional_coverage": directional_eligible / directional_total if directional_total else 0.0,
                "eligible_dates": int(eligible.loc[eligible.target.isin(CLASSES), "datetime"].dt.date.nunique()),
                "date_min": eligible.datetime.min().date().isoformat() if len(eligible) else None,
                "date_max": eligible.datetime.max().date().isoformat() if len(eligible) else None,
            })
    pd.DataFrame(eligibility_rows).to_csv(REPORT_DIR / "sequence_eligibility.csv", index=False)
    local_minutes = raw.datetime.dt.hour * 60 + raw.datetime.dt.minute
    if not local_minutes.between(9 * 60 + 30, 16 * 60 - 5).all():
        raise ValueError("Raw cache has a timestamp outside the NYSE regular five-minute candle-open interval.")
    raw_gaps = raw.groupby("session_date", sort=True).timestamp_utc.diff().dropna()
    gap_rows = raw_gaps[raw_gaps > pd.Timedelta(minutes=5)]
    source_audit = {
        "source_path": str(SOURCE_CSV.relative_to(PROJECT)),
        "symbol": "ECL",
        "feed": "IEX",
        "timeframe": "5Min",
        "rows": int(len(raw)),
        "first_timestamp_utc": raw.timestamp_utc.min().isoformat(),
        "last_timestamp_utc": raw.timestamp_utc.max().isoformat(),
        "new_york_trading_dates": int(raw.session_date.nunique()),
        "duplicate_timestamps": int(raw.timestamp_utc.duplicated().sum()),
        "invalid_ohlc_rows": int(audit["invalid_ohlc"]),
        "zero_volume_rows": int((raw.volume == 0).sum()),
        "out_of_regular_session_rows": int((~local_minutes.between(9 * 60 + 30, 16 * 60 - 5)).sum()),
        "within_session_gaps_over_five_minutes": int(len(gap_rows)),
        "gap_minutes_distribution": (gap_rows.dt.total_seconds().div(60).value_counts().sort_index().to_dict()),
        "gaps_are_preserved_as_sequence_resets": True,
        "source_sha256": sha256_file(SOURCE_CSV),
    }
    (REPORT_DIR / "source_data_audit.json").write_text(json.dumps(source_audit, indent=2, default=str), encoding="utf-8")

    # Read only saved historical binary baseline predictions; align by exact timestamp.
    base_oof = read_approved_csv(BASELINE_OOF_CSV)
    base_val = read_approved_csv(BASELINE_VALIDATION_CSV)
    for frame in (base_oof, base_val):
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    baseline_predictions: list[dict[str, Any]] = []
    oof_predictions: list[dict[str, Any]] = []
    fold_train_times: list[dict[str, Any]] = []

    for length in SEQUENCE_LENGTHS:
        seq_train, x_all_train = split_sequences[("train", length)]
        for fold in folds:
            fold_possible_directional = int(
                train.loc[
                    train.datetime.dt.date.between(fold["validation_start"], fold["validation_end"]), "target"
                ].isin(CLASSES).sum()
            )
            train_mask, val_mask, first_val_time = chronological_fold_masks(
                seq_train, fold["train_end"], fold["validation_start"], fold["validation_end"]
            )
            fold_val = seq_train.loc[val_mask].reset_index(drop=True)
            val_pos = np.flatnonzero(val_mask)
            x_val_all = x_all_train[val_pos]
            fold_train = seq_train.loc[train_mask].reset_index(drop=True)
            train_pos = np.flatnonzero(train_mask)
            if len(fold_train) == 0 or not (fold_train.datetime.max() < first_val_time):
                raise ValueError(f"Empty or non-chronological fold training set, fold {fold['fold']} length {length}.")
            if not (fold_train.future_datetime < first_val_time).all():
                raise ValueError(f"Unpurged target horizon in fold {fold['fold']} length {length}.")
            x_fit_raw = x_all_train[train_pos]
            y_fit = fold_train.target.to_numpy(dtype=str)
            scaler = fit_channel_scaler(x_fit_raw)
            x_fit = transform_sequences(scaler, x_fit_raw)
            x_eval = transform_sequences(scaler, x_val_all)
            model, train_seconds = train_cnn(length, x_fit, y_fit, SEED + int(fold["fold"]) * 100 + length)
            start = time.perf_counter()
            p_up_all = model.predict(x_eval, verbose=0).reshape(-1)
            infer_seconds = time.perf_counter() - start
            prediction_all = np.where(p_up_all >= 0.5, "UP", "DOWN")
            for i, row in fold_val.iterrows():
                p_up = float(p_up_all[i])
                oof_predictions.append({
                    "model": "causal_1d_cnn",
                    "sequence_length": length,
                    "fold": int(fold["fold"]),
                    "datetime": row.datetime,
                    "actual_target": row.target,
                    "actual_binary": row.target if row.target in CLASSES else "",
                    "possible_directional_rows": fold_possible_directional,
                    "prediction": str(prediction_all[i]),
                    "p_down": float(1.0 - p_up),
                    "p_up": p_up,
                })
            fold_train_times.append({
                "model": "causal_1d_cnn",
                "sequence_length": length,
                "fold": int(fold["fold"]),
                "train_start": fold_train.datetime.min().date().isoformat(),
                "train_end": fold_train.datetime.max().date().isoformat(),
                "validation_start": fold_val.datetime.min().date().isoformat(),
                "validation_end": fold_val.datetime.max().date().isoformat(),
                "training_directional_rows": int(len(fold_train)),
                "validation_sequence_rows_all_classes": int(len(fold_val)),
                "validation_directional_rows": int(fold_val.target.isin(CLASSES).sum()),
                "validation_neutral_rows": int((fold_val.target == "NEUTRAL").sum()),
                "purged_training_rows": int(((seq_train.datetime.dt.date <= fold["train_end"]) & (seq_train.future_datetime >= first_val_time)).sum()),
                "training_seconds": train_seconds,
                "inference_seconds": infer_seconds,
            })
            del model

            # Compare the already saved RF/XGBoost OOF predictions on exactly these timestamps.
            oof_key = fold_val[["datetime", "target"]].copy()
            oof_key["datetime_key"] = _make_sequence_key(oof_key)
            for family, config in (("random_forest", "rf_compact_unweighted"), ("xgboost", "xgb_shallow_unweighted")):
                base = base_oof[(base_oof.family == family) & (base_oof.configuration == config) & (base_oof.fold == int(fold["fold"]))].copy()
                base["datetime_key"] = _make_sequence_key(base)
                aligned = oof_key.merge(base[["datetime_key", "actual_three_class", "prediction", "p_down", "p_up"]], on="datetime_key", how="left", validate="one_to_one")
                if aligned["prediction"].isna().any():
                    raise ValueError(f"Saved {family} OOF predictions do not cover all matched timestamps in fold {fold['fold']}.")
                if not np.array_equal(aligned["target"].to_numpy(), fold_val.target.to_numpy()):
                    raise ValueError("Baseline labels are not aligned to the sequence sample timestamps.")
                if not np.array_equal(aligned["actual_three_class"].to_numpy(), fold_val.target.to_numpy()):
                    raise ValueError("Saved baseline actual labels differ from the approved split labels.")
                for i, row in fold_val.iterrows():
                    baseline_predictions.append({
                        "model": family,
                        "configuration": config,
                        "sequence_length": length,
                        "fold": int(fold["fold"]),
                        "datetime": row.datetime,
                        "actual_target": row.target,
                        "actual_binary": row.target if row.target in CLASSES else "",
                        "possible_directional_rows": fold_possible_directional,
                        "prediction": str(aligned.iloc[i].prediction),
                        "p_down": float(aligned.iloc[i].p_down),
                        "p_up": float(aligned.iloc[i].p_up),
                    })

    oof = pd.DataFrame(oof_predictions)
    baselines = pd.DataFrame(baseline_predictions)
    oof.to_csv(REPORT_DIR / "walk_forward_sequence_predictions.csv", index=False)
    baselines.to_csv(REPORT_DIR / "walk_forward_matched_baseline_predictions.csv", index=False)
    combined_oof = pd.concat([oof, baselines], ignore_index=True, sort=False)
    fold_metrics = evaluate(combined_oof, ["model", "sequence_length", "fold"], REPORT_DIR / "walk_forward_fold_metrics.csv")

    overall_rows = []
    for (model_name, length), group in combined_oof.groupby(["model", "sequence_length"], sort=True):
        directional = group[group.actual_binary.isin(CLASSES)]
        possible = int(group[["fold", "possible_directional_rows"]].drop_duplicates().possible_directional_rows.sum())
        overall_rows.append({
            "model": model_name,
            "sequence_length": int(length),
            **binary_metrics(
                directional.actual_binary.to_numpy(),
                directional.prediction.to_numpy(),
                directional.p_up.to_numpy(),
                all_targets=group.actual_target.to_numpy(),
                all_predictions=group.prediction.to_numpy(),
                possible_directional_rows=possible,
            ),
            "mean_fold_accuracy": float(fold_metrics[(fold_metrics.model == model_name) & (fold_metrics.sequence_length == length)].accuracy.mean()),
            "fold_accuracy_sd": float(fold_metrics[(fold_metrics.model == model_name) & (fold_metrics.sequence_length == length)].accuracy.std(ddof=1)),
        })
    overall_metrics = pd.DataFrame(overall_rows)
    overall_metrics.to_csv(REPORT_DIR / "walk_forward_overall_metrics.csv", index=False)

    # Per-date stability is descriptive; it is not a tuning target.
    date_frame = combined_oof.assign(date=pd.to_datetime(combined_oof.datetime).dt.date.astype(str)).copy()
    all_research_labels = pd.concat([train_labels, validation_labels], ignore_index=True)
    all_research_labels["date"] = all_research_labels.datetime.dt.date.astype(str)
    possible_by_date = all_research_labels.groupby("date").target.apply(lambda labels: int(labels.isin(CLASSES).sum()))
    date_frame["possible_directional_rows"] = date_frame.date.map(possible_by_date)
    date_metrics = evaluate(date_frame, ["model", "sequence_length", "date"], REPORT_DIR / "date_stability_metrics.csv")
    date_summary_rows = []
    for (model_name, length), group in date_metrics.groupby(["model", "sequence_length"], sort=True):
        sufficiently_sampled = group[group.n_directional >= 10]
        date_summary_rows.append({
            "model": model_name,
            "sequence_length": int(length),
            "dates_with_at_least_10_directional": int(len(sufficiently_sampled)),
            "median_daily_accuracy": float(sufficiently_sampled.accuracy.median()) if len(sufficiently_sampled) else None,
            "min_daily_accuracy": float(sufficiently_sampled.accuracy.min()) if len(sufficiently_sampled) else None,
            "max_daily_accuracy": float(sufficiently_sampled.accuracy.max()) if len(sufficiently_sampled) else None,
            "fraction_dates_accuracy_gt_50": float((sufficiently_sampled.accuracy > 0.50).mean()) if len(sufficiently_sampled) else None,
            "fraction_dates_accuracy_gt_55": float((sufficiently_sampled.accuracy > 0.55).mean()) if len(sufficiently_sampled) else None,
        })
    date_stability_summary = pd.DataFrame(date_summary_rows)
    date_stability_summary.to_csv(REPORT_DIR / "date_stability_summary.csv", index=False)

    # Final training fit for each predeclared length and one validation inference per model.
    validation_predictions: list[dict[str, Any]] = []
    validation_training_records = []
    for length in SEQUENCE_LENGTHS:
        tr_frame, tr_x_raw = split_sequences[("train", length)]
        va_frame, va_x_raw = split_sequences[("validation", length)]
        fit_mask = tr_frame.target.isin(CLASSES).to_numpy()
        x_fit_raw = tr_x_raw[fit_mask]
        y_fit = tr_frame.loc[fit_mask, "target"].to_numpy(dtype=str)
        scaler = fit_channel_scaler(x_fit_raw)
        x_fit = transform_sequences(scaler, x_fit_raw)
        x_val = transform_sequences(scaler, va_x_raw)
        model, train_seconds = train_cnn(length, x_fit, y_fit, SEED + length)
        start = time.perf_counter()
        p_up_all = model.predict(x_val, verbose=0).reshape(-1)
        infer_seconds = time.perf_counter() - start
        prediction_all = np.where(p_up_all >= 0.5, "UP", "DOWN")
        for i, row in va_frame.iterrows():
            p_up = float(p_up_all[i])
            validation_predictions.append({
                "model": "causal_1d_cnn",
                "sequence_length": length,
                "datetime": row.datetime,
                "actual_target": row.target,
                "actual_binary": row.target if row.target in CLASSES else "",
                "possible_directional_rows": int(validation.target.isin(CLASSES).sum()),
                "prediction": str(prediction_all[i]),
                "p_down": float(1.0 - p_up),
                "p_up": p_up,
            })
        model_path = MODEL_DIR / f"causal_1d_cnn_len{length}.keras"
        model.save(model_path, include_optimizer=False)
        scaler_path = MODEL_DIR / f"channel_scaler_len{length}.joblib"
        joblib.dump(scaler, scaler_path)
        validation_training_records.append({
            "model": "causal_1d_cnn",
            "sequence_length": length,
            "training_directional_rows": int(len(y_fit)),
            "validation_sequence_rows_all_classes": int(len(va_frame)),
            "validation_directional_rows": int(va_frame.target.isin(CLASSES).sum()),
            "validation_neutral_rows": int((va_frame.target == "NEUTRAL").sum()),
            "training_seconds": train_seconds,
            "inference_seconds": infer_seconds,
            "model_path": str(model_path.relative_to(PROJECT)),
            "scaler_path": str(scaler_path.relative_to(PROJECT)),
        })
        del model

    val_seq = pd.DataFrame(validation_predictions)
    val_base_predictions: list[dict[str, Any]] = []
    val_key_base = base_val.copy()
    val_key_base["datetime_key"] = _make_sequence_key(val_key_base)
    for length in SEQUENCE_LENGTHS:
        va_frame, _ = split_sequences[("validation", length)]
        keys = va_frame[["datetime", "target"]].copy()
        keys["datetime_key"] = _make_sequence_key(keys)
        for family, pred_col, down_col, up_col in (
            ("random_forest", "random_forest_prediction", "random_forest_p_down", "random_forest_p_up"),
            ("xgboost", "xgboost_prediction", "xgboost_p_down", "xgboost_p_up"),
        ):
            aligned = keys.merge(
                val_key_base[["datetime_key", "actual_three_class", pred_col, down_col, up_col]],
                on="datetime_key", how="left", validate="one_to_one",
            )
            if aligned[pred_col].isna().any():
                raise ValueError(f"Saved validation {family} predictions do not cover matched sequence length {length} rows.")
            if not np.array_equal(aligned["actual_three_class"].to_numpy(), va_frame.target.to_numpy()):
                raise ValueError("Saved validation baseline actual labels differ from the approved split labels.")
            for i, row in va_frame.iterrows():
                val_base_predictions.append({
                    "model": family,
                    "sequence_length": length,
                    "datetime": row.datetime,
                    "actual_target": row.target,
                    "actual_binary": row.target if row.target in CLASSES else "",
                    "possible_directional_rows": int(validation.target.isin(CLASSES).sum()),
                    "prediction": str(aligned.iloc[i][pred_col]),
                    "p_down": float(aligned.iloc[i][down_col]),
                    "p_up": float(aligned.iloc[i][up_col]),
                })
    val_base = pd.DataFrame(val_base_predictions)
    val_seq.to_csv(REPORT_DIR / "validation_sequence_predictions.csv", index=False)
    val_base.to_csv(REPORT_DIR / "validation_matched_baseline_predictions.csv", index=False)
    combined_val = pd.concat([val_seq, val_base], ignore_index=True, sort=False)
    validation_metrics = evaluate(combined_val, ["model", "sequence_length"], REPORT_DIR / "validation_metrics.csv")

    # A single reproducibility manifest records only permitted inputs and generated artifacts.
    input_paths = (SOURCE_CSV, SOURCE_METADATA, TRAIN_CSV, VALIDATION_CSV, FOLDS_CSV, BASELINE_OOF_CSV, BASELINE_VALIDATION_CSV, BASELINE_CONFIG)
    manifest = {
        "experiment": "ECL Alpaca IEX raw OHLCV sequence research v1",
        "source_constraints": {
            "approved_input_paths": [str(p.relative_to(PROJECT)) for p in input_paths],
            "test_csv_accessed": False,
            "post_2026_09_24_data_accessed": False,
            "network_access": False,
            "live_app_or_frozen_model_modified": False,
            "source_symbol": "ECL",
            "source_feed": "IEX",
            "source_timeframe": "5Min",
            "source_timezone": "UTC event timestamps converted to America/New_York for session checks",
            "source_first_utc": raw.timestamp_utc.min().isoformat(),
            "source_last_utc": raw.timestamp_utc.max().isoformat(),
            "source_rows": int(len(raw)),
            "source_sha256": sha256_file(SOURCE_CSV),
            "source_metadata_sha256": sha256_file(SOURCE_METADATA),
        },
        "split_rows": {"train": int(len(train)), "validation": int(len(validation))},
        "feature_channels": list(OHLCV_COLUMNS),
        "sequence_lengths": list(SEQUENCE_LENGTHS),
        "folds": fold_table.to_dict("records"),
        "source_audit_invalid_ohlc_rows": int(audit["invalid_ohlc"]),
        "sequence_eligibility": eligibility_rows,
        "fold_training": fold_train_times,
        "validation_training": validation_training_records,
        "metrics_definition": {
            "directional_metrics": "NEUTRAL actual labels excluded; binary model emits UP/DOWN for every sequence-eligible row",
            "confusion_matrix_order": "actual rows, predicted columns; DOWN then UP",
            "sequence_eligibility_coverage": "eligible directional labels divided by all directional labels in the same split/fold",
            "all_candle_operational_accuracy": "correct directional predictions divided by all sequence-eligible labeled rows, so NEUTRAL actuals count as incorrect",
        },
        "seed": SEED,
        "tensorflow_version": _tensorflow().__version__,
        "models": [r["model_path"] for r in validation_training_records],
        "scalers": [r["scaler_path"] for r in validation_training_records],
        "outputs": [str(p.relative_to(PROJECT)) for p in sorted(REPORT_DIR.glob("*.csv"))],
        "selected_or_promoted_model": None,
    }
    (MODEL_DIR / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (MODEL_DIR / "feature_schema.json").write_text(json.dumps({
        "channels": list(OHLCV_COLUMNS),
        "formulas": {
            "open_high_low_close": "100 * (bar_price / current_sequence_end_close - 1) within each sample",
            "volume": "log1p(raw IEX volume), then channel-wise StandardScaler fitted on training windows only",
        },
        "continuity": "same New York session date and exact 5-minute timestamps for all bars in the selected sequence",
        "no_volume_fabrication": True,
        "future_candles_in_input": False,
    }, indent=2), encoding="utf-8")
    (MODEL_DIR / "fold_training_config.json").write_text(json.dumps({
        "epochs": 20,
        "batch_size": 32,
        "shuffle": False,
        "learning_rate": 0.001,
        "loss": "binary_crossentropy",
        "seed": SEED,
        "architecture": "causal Conv1D(8,3)+GlobalAveragePooling1D+Dense(8,L2=0.01)+Dropout(0.20)+sigmoid",
        "no_threshold_search": True,
        "class_weighting": "none; Directional UP/DOWN classes in the approved train split are close to balanced.",
    }, indent=2), encoding="utf-8")

    # concise machine-readable summary for report generation/inspection
    (REPORT_DIR / "run_summary.json").write_text(json.dumps({
        "walk_forward_metrics": fold_metrics.to_dict("records"),
        "validation_metrics": validation_metrics.to_dict("records"),
        "walk_forward_overall_metrics": overall_metrics.to_dict("records"),
        "date_stability_summary": date_stability_summary.to_dict("records"),
        "tf_version": manifest["tensorflow_version"],
        "sequence_eligibility": eligibility_rows,
    }, indent=2, default=str), encoding="utf-8")
    manifest["input_sha256"] = {str(p.relative_to(PROJECT)): sha256_file(p) for p in input_paths}
    artifact_paths = [
        MODEL_DIR / "configuration.json",
        MODEL_DIR / "feature_schema.json",
        MODEL_DIR / "fold_training_config.json",
        MODEL_DIR / "run_sequence_research.py",
        MODEL_DIR / "test_sequence_research.py",
        *[MODEL_DIR / f"causal_1d_cnn_len{length}.keras" for length in SEQUENCE_LENGTHS],
        *[MODEL_DIR / f"channel_scaler_len{length}.joblib" for length in SEQUENCE_LENGTHS],
    ]
    manifest["artifact_sha256"] = {str(p.relative_to(PROJECT)): sha256_file(p) for p in artifact_paths}
    manifest["report_outputs"] = [str(p.relative_to(PROJECT)) for p in sorted(REPORT_DIR.iterdir()) if p.is_file()]
    (MODEL_DIR / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("Sequence research completed.")
    print(f"Source rows: {len(raw):,}; train rows: {len(train):,}; validation rows: {len(validation):,}")
    print("Walk-forward aggregate and per-fold metrics:")
    if not fold_metrics.empty:
        summary = fold_metrics.groupby(["model", "sequence_length"], as_index=False).agg(
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            folds=("fold", "count"),
        )
        print(summary.to_string(index=False))
    print("Validation metrics (diagnostic only; not used to tune/select):")
    print(validation_metrics[["model", "sequence_length", "n_directional", "accuracy", "balanced_accuracy", "macro_f1", "sequence_eligibility_coverage_of_directional_rows", "all_candle_operational_accuracy_neutral_wrong"]].to_string(index=False))
    print(f"Reports: {REPORT_DIR}")
    print(f"Models/configuration: {MODEL_DIR}")


if __name__ == "__main__":
    run()
