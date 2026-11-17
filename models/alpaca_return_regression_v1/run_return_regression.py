from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


PROJECT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT / "reports" / "alpaca_return_regression_v1"
DATA_DIR = PROJECT / "data" / "alpaca_ml_pipeline_v1"
SOURCE = PROJECT / "models" / "alpaca_1min_research_v1" / "raw_ecl_iex_5min_train_validation.csv"
SOURCE_META = SOURCE.with_name("fetch_5min_metadata.json")
FEATURE_SCHEMA = PROJECT / "models" / "alpaca_binary_research_v3" / "feature_schema.json"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
FOLDS_CSV = PROJECT / "reports" / "alpaca_binary_research_v3" / "walk_forward_folds.csv"
BASELINE_OOF = PROJECT / "reports" / "alpaca_binary_research_v3" / "walk_forward_oof_predictions.csv"
BASELINE_VAL = PROJECT / "reports" / "alpaca_binary_research_v3" / "validation_predictions.csv"
BASELINE_CONFIG = PROJECT / "models" / "alpaca_binary_research_v3" / "configuration.json"
CLASSES = ("DOWN", "UP")
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def approved_paths() -> set[Path]:
    return {p.resolve() for p in (SOURCE, SOURCE_META, FEATURE_SCHEMA, TRAIN_CSV, VAL_CSV, FOLDS_CSV, BASELINE_OOF, BASELINE_VAL, BASELINE_CONFIG)}


def approved_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if path.resolve() not in approved_paths():
        raise ValueError(f"Unapproved research input path: {path}")
    return pd.read_csv(path, **kwargs)


def approved_read_json(path: Path) -> dict[str, Any]:
    if path.resolve() not in approved_paths():
        raise ValueError(f"Unapproved research input path: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def return_pct(reference_close: np.ndarray | pd.Series, future_close: np.ndarray | pd.Series) -> np.ndarray:
    reference = np.asarray(reference_close, dtype=float)
    future = np.asarray(future_close, dtype=float)
    if not np.isfinite(reference).all() or not np.isfinite(future).all() or np.any(reference <= 0) or np.any(future <= 0):
        raise ValueError("Return prices must be finite and positive.")
    return (future / reference - 1.0) * 100.0


def target_from_return(return_values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(return_values, dtype=float)
    return np.where(values > 0.10, "UP", np.where(values < -0.10, "DOWN", "NEUTRAL"))


def verify_target_alignment(frame: pd.DataFrame, close_by_time: pd.Series, source_times: set[pd.Timestamp], split_name: str) -> None:
    if not ((frame.future_datetime - frame.datetime) == pd.Timedelta(minutes=15)).all():
        raise ValueError(f"{split_name} has a target not exactly 15 minutes forward.")
    if (frame.datetime.dt.date != frame.future_datetime.dt.date).any():
        raise ValueError(f"{split_name} target crosses a New York session date.")
    if not frame.datetime.isin(source_times).all() or not frame.future_datetime.isin(source_times).all():
        raise ValueError(f"{split_name} decision or endpoint candle is missing from the permitted raw source.")
    # Existing builder defines targets only inside continuous five-minute segments.
    for offset in (5, 10, 15):
        if not (frame.datetime + pd.Timedelta(minutes=offset)).isin(source_times).all():
            raise ValueError(f"{split_name} target crosses a missing five-minute interval at +{offset}m.")
    close_t = frame.datetime.map(close_by_time)
    close_future = frame.future_datetime.map(close_by_time)
    recomputed = return_pct(close_t.to_numpy(dtype=float), close_future.to_numpy(dtype=float))
    if not np.allclose(recomputed, frame.future_return.to_numpy(dtype=float), rtol=0.0, atol=1e-9):
        raise ValueError(f"{split_name} saved return differs from raw endpoint close calculation.")
    if not np.array_equal(target_from_return(frame.future_return), frame.target.to_numpy(dtype=str)):
        raise ValueError(f"{split_name} labels do not match existing strict thresholds.")


def make_model_pipelines(seed: int = SEED) -> dict[str, Pipeline]:
    """Fixed, modest estimators; preprocessing is learned when each pipeline is fit."""
    return {
        "ridge": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "random_forest_regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, max_features=1.0, random_state=seed, n_jobs=1)),
        ]),
        "xgboost_regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(n_estimators=250, max_depth=2, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, objective="reg:squarederror", eval_metric="rmse", random_state=seed, n_jobs=1, verbosity=0)),
        ]),
        "hist_gradient_boosting_regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)),
        ]),
    }


def direction_from_return(predicted_return: np.ndarray | pd.Series) -> np.ndarray:
    """Fixed zero-boundary conversion; exact zero is a deterministic DOWN tie-break."""
    values = np.asarray(predicted_return, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predicted returns must be finite.")
    return np.where(values > 0.0, "UP", "DOWN")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    metadata = approved_read_json(SOURCE_META)
    if (metadata.get("symbol", "").upper(), metadata.get("feed", "").lower(), metadata.get("timeframe")) != ("ECL", "iex", "5Min"):
        raise ValueError("Approved source metadata is not ECL Alpaca IEX 5Min.")
    if metadata.get("permitted_new_york_date_range") != ["2025-11-13", "2026-09-24"]:
        raise ValueError("Approved raw cache is not limited to the permitted research period.")
    if sha256(SOURCE) != metadata.get("raw_csv_sha256"):
        raise ValueError("Approved source cache hash mismatch.")
    schema = approved_read_json(FEATURE_SCHEMA)
    features = schema.get("features", [])
    if len(features) != 24 or len(set(features)) != 24:
        raise ValueError("Expected exact 24-feature B schema.")

    raw = approved_read_csv(SOURCE)
    required = {"timestamp_utc", "open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Raw source missing OHLCV fields: {required - set(raw.columns)}")
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True, errors="raise")
    raw = raw.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    raw["datetime"] = raw.timestamp_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    if raw.timestamp_utc.duplicated().any():
        raise ValueError("Raw source contains duplicate timestamps.")
    if raw.timestamp_utc.max() > pd.Timestamp("2026-09-24 23:59:59", tz="UTC"):
        raise ValueError("Raw source contains data after the approved cutoff.")
    vals = raw[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    bad = (~np.isfinite(vals.to_numpy(dtype=float)).all(axis=1)) | (vals["low"] > vals[["open", "close"]].min(axis=1)) | (vals["high"] < vals[["open", "close"]].max(axis=1)) | (vals["low"] > vals["high"]) | (vals[["open", "high", "low", "close"]] <= 0).any(axis=1) | (vals["volume"] < 0)
    if bad.any():
        raise ValueError(f"Invalid OHLCV source rows: {int(bad.sum())}")
    # Keep assignment order explicit; a set-derived column list can permute OHLCV columns.
    raw[["open", "high", "low", "close", "volume"]] = vals[["open", "high", "low", "close", "volume"]]
    close_by_time = pd.Series(raw.close.to_numpy(dtype=float), index=pd.DatetimeIndex(raw.datetime))
    source_times = set(close_by_time.index)

    train = approved_read_csv(TRAIN_CSV)
    validation = approved_read_csv(VAL_CSV)
    for split_name, frame, date_bounds in (
        ("train", train, ("2025-11-13", "2026-08-05")),
        ("validation", validation, ("2026-08-06", "2026-09-24")),
    ):
        expected_columns = ["datetime", "future_datetime", "future_return", "target", *features]
        if list(frame.columns) != expected_columns:
            raise ValueError(f"{split_name} columns differ from the frozen ordered feature contract.")
        frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
        frame["future_datetime"] = pd.to_datetime(frame.future_datetime, errors="raise")
        dates = frame.datetime.dt.strftime("%Y-%m-%d")
        if dates.min() < date_bounds[0] or dates.max() > date_bounds[1]:
            raise ValueError(f"{split_name} contains rows outside approved dates.")
        if frame.datetime.duplicated().any() or not frame.datetime.is_monotonic_increasing:
            raise ValueError(f"{split_name} timestamps are not unique/chronological.")
        verify_target_alignment(frame, close_by_time, source_times, split_name)
        x = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError(f"{split_name} has missing or non-finite model features.")
    folds = approved_read_csv(FOLDS_CSV)
    baseline_oof = approved_read_csv(BASELINE_OOF)
    baseline_val = approved_read_csv(BASELINE_VAL)
    for f in (baseline_oof, baseline_val):
        f["datetime"] = pd.to_datetime(f.datetime, errors="raise")
    xgb=approved_read_json(BASELINE_CONFIG)
    if xgb.get("xgboost_version") not in (None, "3.4.1"):
        raise ValueError("Existing XGBoost baseline version is not the verified environment version.")
    return raw, train, validation, folds, baseline_oof, baseline_val, {"features": features, "metadata": metadata, "baseline_config": xgb}


def fold_masks(frame: pd.DataFrame, train_end: Any, validation_start: Any, validation_end: Any) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    train_end = pd.Timestamp(train_end).date()
    start = pd.Timestamp(validation_start).date()
    end = pd.Timestamp(validation_end).date()
    dates = frame.datetime.dt.date
    val_mask = (dates >= start) & (dates <= end)
    if not val_mask.any():
        raise ValueError("Fold has no evaluation rows.")
    first_val = frame.loc[val_mask, "datetime"].min()
    train_mask = (dates <= train_end) & (frame.future_datetime < first_val)
    return train_mask.to_numpy(), val_mask.to_numpy(), first_val


def regression_metrics(actual_return: np.ndarray, predicted_return: np.ndarray, actual_target: np.ndarray) -> dict[str, Any]:
    yret = np.asarray(actual_return, dtype=float)
    predret = np.asarray(predicted_return, dtype=float)
    yclass = np.asarray(actual_target, dtype=str)
    directional = np.isin(yclass, CLASSES)
    actual = yclass[directional]
    predicted = direction_from_return(predret[directional])
    pdown = precision_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[0] if len(actual) else 0.0
    pup = precision_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[1] if len(actual) else 0.0
    rdown = recall_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[0] if len(actual) else 0.0
    rup = recall_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[1] if len(actual) else 0.0
    corr = float(np.corrcoef(yret, predret)[0, 1]) if len(yret) > 1 and np.std(yret) > 0 and np.std(predret) > 0 else None
    return {
        "n_all_rows": int(len(yret)),
        "n_directional": int(directional.sum()),
        "actual_down": int(np.sum(actual == "DOWN")),
        "actual_up": int(np.sum(actual == "UP")),
        "prediction_down_directional": int(np.sum(predicted == "DOWN")),
        "prediction_up_directional": int(np.sum(predicted == "UP")),
        "prediction_up_share_directional": float(np.mean(predicted == "UP")) if len(predicted) else None,
        "directional_signal_coverage": 1.0 if len(actual) else 0.0,
        "directional_accuracy": float(accuracy_score(actual, predicted)) if len(actual) else None,
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)) if len(actual) else None,
        "down_precision": float(pdown), "down_recall": float(rdown), "up_precision": float(pup), "up_recall": float(rup),
        "directional_macro_f1": float(f1_score(actual, predicted, labels=list(CLASSES), average="macro", zero_division=0)) if len(actual) else None,
        "directional_confusion_matrix_actual_rows_predicted_columns_down_up": confusion_matrix(actual, predicted, labels=list(CLASSES)).tolist() if len(actual) else [[0, 0], [0, 0]],
        "mae_all_rows_pct_points": float(mean_absolute_error(yret, predret)),
        "rmse_all_rows_pct_points": float(mean_squared_error(yret, predret) ** 0.5),
        "return_correlation_all_rows": corr,
        "neutral_actual_rows": int(np.sum(yclass == "NEUTRAL")),
        "all_candle_directional_accuracy_neutral_wrong": float(np.sum((yclass != "NEUTRAL") & (direction_from_return(predret) == yclass)) / len(yclass)) if len(yclass) else None,
    }


def class_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=str)
    predicted = np.asarray(predicted, dtype=str)
    return {
        "prediction_down": int(np.sum(predicted == "DOWN")),
        "prediction_up": int(np.sum(predicted == "UP")),
        "prediction_up_share": float(np.mean(predicted == "UP")) if len(predicted) else None,
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=list(CLASSES), average="macro", zero_division=0)),
        "down_precision": float(precision_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[0]),
        "down_recall": float(recall_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[0]),
        "up_precision": float(precision_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[1]),
        "up_recall": float(recall_score(actual, predicted, labels=list(CLASSES), average=None, zero_division=0)[1]),
        "confusion_matrix_actual_rows_predicted_columns_down_up": confusion_matrix(actual, predicted, labels=list(CLASSES)).tolist(),
    }


def metric_rows(predictions: pd.DataFrame, group_cols: list[str], kind: str) -> pd.DataFrame:
    rows = []
    for keys, g in predictions.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y_all = g.actual_target.to_numpy(dtype=str)
        y_bin = y_all[np.isin(y_all, CLASSES)]
        pred_bin = g.loc[g.actual_target.isin(CLASSES), "prediction"].to_numpy(dtype=str)
        if kind == "regression":
            ret_metrics = regression_metrics(g.future_return.to_numpy(dtype=float), g.predicted_return.to_numpy(dtype=float), y_all)
            rows.append({**dict(zip(group_cols, keys)), **ret_metrics})
        else:
            rows.append({**dict(zip(group_cols, keys)), "n_directional": int(len(y_bin)), **class_metrics(y_bin, pred_bin)})
    return pd.DataFrame(rows)


def _baseline_aligned(frame: pd.DataFrame, baseline: pd.DataFrame, fold: int | None = None) -> pd.DataFrame:
    left = frame[["datetime", "target", "future_return"]].copy()
    left["datetime_key"] = left.datetime.dt.strftime("%Y-%m-%d %H:%M:%S")
    b = baseline.copy()
    b["datetime_key"] = b.datetime.dt.strftime("%Y-%m-%d %H:%M:%S")
    if fold is not None:
        b = b[b.fold == fold]
        b = b[(b.family == "random_forest") & (b.configuration == "rf_compact_unweighted") | (b.family == "xgboost") & (b.configuration == "xgb_shallow_unweighted")]
        keep = b[["datetime_key", "actual_three_class", "family", "prediction"]]
        aligned = left.merge(keep, on="datetime_key", how="left", validate="one_to_many")
        if aligned.actual_three_class.isna().any():
            raise ValueError(f"Saved OOF baseline predictions are missing matched fold {fold} rows.")
        return aligned
    keep = b[["datetime_key", "actual_three_class", "random_forest_prediction", "xgboost_prediction"]]
    aligned = left.merge(keep, on="datetime_key", how="left", validate="one_to_one")
    if aligned.actual_three_class.isna().any():
        raise ValueError("Saved validation baseline predictions are missing matched rows.")
    return aligned


def run() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw, train, validation, folds, baseline_oof, baseline_val, extra = load_inputs()
    features = extra["features"]
    x_train = train[features].to_numpy(dtype=float)
    y_train = train.future_return.to_numpy(dtype=float)
    x_val = validation[features].to_numpy(dtype=float)
    y_val = validation.future_return.to_numpy(dtype=float)

    # Source-level audit after exact endpoint/gap and return reconstruction checks.
    source_audit = {
        "source_path": str(SOURCE.relative_to(PROJECT)), "symbol": "ECL", "feed": "IEX", "timeframe": "5Min",
        "rows": int(len(raw)), "first_utc": raw.timestamp_utc.min().isoformat(), "last_utc": raw.timestamp_utc.max().isoformat(),
        "source_sha256": sha256(SOURCE), "target_return_verified_against_raw_close": True,
        "decision_endpoint_and_intervening_5m_timestamps_verified": True,
    }
    (REPORT_DIR / "source_target_audit.json").write_text(json.dumps(source_audit, indent=2), encoding="utf-8")

    fold_rows: list[dict[str, Any]] = []
    oof_pred_rows: list[dict[str, Any]] = []
    aligned_baseline_rows: list[dict[str, Any]] = []
    training_audit: list[dict[str, Any]] = []
    for f in folds.to_dict("records"):
        fold = int(f["fold"])
        train_mask, val_mask, first_val = fold_masks(train, f["train_end"], f["validation_start"], f["validation_end"])
        tr, ev = train.loc[train_mask].copy(), train.loc[val_mask].copy()
        if tr.empty or ev.empty or tr.future_datetime.max() >= first_val or tr.datetime.max() >= first_val:
            raise ValueError(f"Fold {fold} is empty, non-chronological or not purged.")
        baseline_aligned = _baseline_aligned(ev, baseline_oof, fold=fold)
        if not np.array_equal(baseline_aligned.target.to_numpy(dtype=str), baseline_aligned.actual_three_class.to_numpy(dtype=str)):
            raise ValueError(f"Baseline fold {fold} actual labels do not align with regression labels.")
        templates = make_model_pipelines(seed=SEED + fold)
        for model_name, pipeline in templates.items():
            start = time.perf_counter()
            pipeline.fit(tr[features], tr.future_return)
            train_seconds = time.perf_counter() - start
            start = time.perf_counter()
            pred_return = pipeline.predict(ev[features]).astype(float)
            infer_seconds = time.perf_counter() - start
            direction = direction_from_return(pred_return)
            record = pd.DataFrame({
                "model": model_name, "fold": fold, "datetime": ev.datetime.to_numpy(),
                "actual_target": ev.target.to_numpy(), "future_return": ev.future_return.to_numpy(dtype=float),
                "predicted_return": pred_return, "prediction": direction,
            })
            oof_pred_rows.extend(record.to_dict("records"))
            metrics = regression_metrics(record.future_return.to_numpy(), pred_return, record.actual_target.to_numpy())
            fold_rows.append({
                "model": model_name, "fold": fold, "train_start": tr.datetime.min().date().isoformat(), "train_end": tr.datetime.max().date().isoformat(),
                "evaluation_start": ev.datetime.min().date().isoformat(), "evaluation_end": ev.datetime.max().date().isoformat(),
                "train_rows_all_returns": int(len(tr)), "evaluation_rows_all_returns": int(len(ev)),
                "purged_training_rows": int(((train.datetime.dt.date <= pd.Timestamp(f["train_end"]).date()) & (train.future_datetime >= first_val)).sum()),
                "training_seconds": train_seconds, "inference_seconds": infer_seconds, **metrics,
            })
            model_path = MODEL_DIR / f"{model_name}_fold{fold}.joblib"
            joblib.dump(pipeline, model_path)
            training_audit.append({"model": model_name, "fold": fold, "artifact": str(model_path.relative_to(PROJECT)), "trained_rows": len(tr), "eval_rows": len(ev)})
        for _, row in baseline_aligned.iterrows():
            aligned_baseline_rows.append({
                "fold": fold, "datetime": row.datetime, "actual_target": row.target, "future_return": row.future_return,
                "model": str(row.family), "prediction": str(row.prediction),
            })

    oof = pd.DataFrame(oof_pred_rows)
    oof.to_csv(REPORT_DIR / "walk_forward_regression_predictions.csv", index=False)
    baselines_oof = pd.DataFrame(aligned_baseline_rows)
    baselines_oof.to_csv(REPORT_DIR / "walk_forward_matched_baseline_predictions.csv", index=False)
    baseline_fold_metrics = []
    for (family, fold_id), g in baselines_oof.groupby(["model", "fold"], sort=True):
        directional = g[g.actual_target.isin(CLASSES)]
        baseline_fold_metrics.append({"model": f"binary_baseline_{family}", "fold": int(fold_id), "n_directional": int(len(directional)), **class_metrics(directional.actual_target.to_numpy(dtype=str), directional.prediction.to_numpy(dtype=str))})
    pd.DataFrame(baseline_fold_metrics).to_csv(REPORT_DIR / "walk_forward_matched_baseline_metrics.csv", index=False)
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(REPORT_DIR / "walk_forward_fold_metrics.csv", index=False)

    overall_rows = []
    for model_name, g in oof.groupby("model", sort=True):
        overall_rows.append({"model": model_name, **regression_metrics(g.future_return.to_numpy(), g.predicted_return.to_numpy(), g.actual_target.to_numpy())})
    for model_name, g in baselines_oof.groupby("model", sort=True):
        y = g.loc[g.actual_target.isin(CLASSES), "actual_target"].to_numpy(dtype=str)
        p = g.loc[g.actual_target.isin(CLASSES), "prediction"].to_numpy(dtype=str)
        overall_rows.append({"model": f"binary_baseline_{model_name}", "n_directional": int(len(y)), **class_metrics(y, p)})
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(REPORT_DIR / "walk_forward_overall_metrics.csv", index=False)

    val_base_aligned = _baseline_aligned(validation, baseline_val)
    if not np.array_equal(val_base_aligned.target.to_numpy(dtype=str), val_base_aligned.actual_three_class.to_numpy(dtype=str)):
        raise ValueError("Saved validation baseline actual labels do not match the approved validation split.")
    validation_prediction_rows: list[dict[str, Any]] = []
    validation_metrics_rows: list[dict[str, Any]] = []
    full_models = make_model_pipelines(seed=SEED)
    for model_name, pipeline in full_models.items():
        start = time.perf_counter()
        pipeline.fit(train[features], train.future_return)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predicted_return = pipeline.predict(validation[features]).astype(float)
        infer_seconds = time.perf_counter() - start
        predicted_direction = direction_from_return(predicted_return)
        records = pd.DataFrame({
            "model": model_name, "datetime": validation.datetime,
            "actual_target": validation.target, "future_return": y_val,
            "predicted_return": predicted_return, "prediction": predicted_direction,
        })
        validation_prediction_rows.extend(records.to_dict("records"))
        validation_metrics_rows.append({"model": model_name, "training_seconds": train_seconds, "inference_seconds": infer_seconds, **regression_metrics(y_val, predicted_return, validation.target.to_numpy(dtype=str))})
        joblib.dump(pipeline, MODEL_DIR / f"{model_name}_full_train.joblib")

    val_preds = pd.DataFrame(validation_prediction_rows)
    val_preds.to_csv(REPORT_DIR / "validation_regression_predictions.csv", index=False)
    val_baseline_rows = []
    for family, col in (("random_forest", "random_forest_prediction"), ("xgboost", "xgboost_prediction")):
        g = val_base_aligned[["datetime", "target"]].copy()
        g["model"] = f"binary_baseline_{family}"
        g["prediction"] = val_base_aligned[col].to_numpy()
        val_baseline_rows.extend(g.to_dict("records"))
    pd.DataFrame(val_baseline_rows).to_csv(REPORT_DIR / "validation_matched_baseline_predictions.csv", index=False)
    validation_metrics = pd.DataFrame(validation_metrics_rows)
    validation_metrics.to_csv(REPORT_DIR / "validation_metrics.csv", index=False)
    baseline_val_metrics = []
    for model_name, g in pd.DataFrame(val_baseline_rows).groupby("model"):
        actual = g.loc[g.target.isin(CLASSES), "target"].to_numpy(dtype=str)
        pred = g.loc[g.target.isin(CLASSES), "prediction"].to_numpy(dtype=str)
        baseline_val_metrics.append({"model": model_name, **class_metrics(actual, pred)})
    pd.DataFrame(baseline_val_metrics).to_csv(REPORT_DIR / "validation_baseline_metrics.csv", index=False)

    # Majority-direction controls are computed without fitting a model.
    majority_rows = []
    for f in folds.to_dict("records"):
        fold = int(f["fold"])
        _, eval_mask, _ = fold_masks(train, f["train_end"], f["validation_start"], f["validation_end"])
        eval_frame = train.loc[eval_mask]
        train_mask, _, _ = fold_masks(train, f["train_end"], f["validation_start"], f["validation_end"])
        train_directional = train.loc[train_mask & train.target.isin(CLASSES), "target"]
        majority = str(train_directional.value_counts().idxmax())
        actual = eval_frame.loc[eval_frame.target.isin(CLASSES), "target"].to_numpy(dtype=str)
        for name, pred in (("training_majority", np.repeat(majority, len(actual))), ("always_up", np.repeat("UP", len(actual))), ("always_down", np.repeat("DOWN", len(actual)))):
            majority_rows.append({"fold": fold, "baseline": name, "training_majority_class": majority if name == "training_majority" else "", "n_directional": len(actual), **class_metrics(actual, pred)})
    majority = pd.DataFrame(majority_rows)
    majority.to_csv(REPORT_DIR / "walk_forward_majority_baselines.csv", index=False)
    val_actual = validation.loc[validation.target.isin(CLASSES), "target"].to_numpy(dtype=str)
    val_majority = str(train.loc[train.target.isin(CLASSES), "target"].value_counts().idxmax())
    val_majority_rows = []
    for name, pred in (("training_majority", np.repeat(val_majority, len(val_actual))), ("always_up", np.repeat("UP", len(val_actual))), ("always_down", np.repeat("DOWN", len(val_actual)))):
        val_majority_rows.append({"baseline": name, "training_majority_class": val_majority if name == "training_majority" else "", "n_directional": len(val_actual), **class_metrics(val_actual, pred)})
    pd.DataFrame(val_majority_rows).to_csv(REPORT_DIR / "validation_majority_baselines.csv", index=False)

    manifest = {
        "experiment": "ECL Alpaca IEX return regression v1",
        "approved_inputs": [str(p.relative_to(PROJECT)) for p in (SOURCE, SOURCE_META, FEATURE_SCHEMA, TRAIN_CSV, VAL_CSV, FOLDS_CSV, BASELINE_OOF, BASELINE_VAL, BASELINE_CONFIG)],
        "test_csv_accessed": False,
        "post_2026_09_24_data_accessed": False,
        "network_access": False,
        "symbol": "ECL", "feed": "IEX", "timeframe": "5Min", "horizon_minutes": 15,
        "target_units": "percentage points", "target_threshold_pct_for_label_reference": 0.10,
        "feature_order": features,
        "fold_schedule": folds.to_dict("records"),
        "xgboost_version": __import__("xgboost").__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "training_audit": training_audit,
        "input_sha256": {str(p.relative_to(PROJECT)): sha256(p) for p in (SOURCE, SOURCE_META, FEATURE_SCHEMA, TRAIN_CSV, VAL_CSV, FOLDS_CSV, BASELINE_OOF, BASELINE_VAL, BASELINE_CONFIG)},
        "artifact_sha256": {},
        "selected_or_promoted_model": None,
    }
    (MODEL_DIR / "source_feature_contract.json").write_text(json.dumps({"feature_order": features, "feature_count": len(features), "target_column": "future_return", "target_units": "percentage points", "excluded_columns": ["datetime", "future_datetime", "target"], "baseline_feature_model_inputs": "exactly the same 24 columns"}, indent=2), encoding="utf-8")
    (MODEL_DIR / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    outputs = [p for p in MODEL_DIR.iterdir() if p.is_file() and p.name != "reproducibility_manifest.json"] + [p for p in REPORT_DIR.iterdir() if p.is_file()]
    manifest["artifact_sha256"] = {str(p.relative_to(PROJECT)): sha256(p) for p in outputs}
    (MODEL_DIR / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("Regression experiment complete; no validation tuning was performed.")
    print(f"Approved source rows={len(raw):,}; train={len(train):,}; validation={len(validation):,}; features={len(features)}")
    print("Walk-forward metrics:")
    print(fold_metrics[["model", "fold", "n_directional", "directional_accuracy", "balanced_accuracy", "directional_macro_f1", "mae_all_rows_pct_points", "rmse_all_rows_pct_points", "return_correlation_all_rows", "prediction_up_share_directional"]].to_string(index=False))
    print("Validation metrics:")
    print(validation_metrics[["model", "n_directional", "directional_accuracy", "balanced_accuracy", "directional_macro_f1", "mae_all_rows_pct_points", "rmse_all_rows_pct_points", "return_correlation_all_rows", "prediction_up_share_directional"]].to_string(index=False))


if __name__ == "__main__":
    run()
