"""Isolated ECL 3-class research; reads only approved train/validation inputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, log_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports" / "alpaca_three_class_v1"
DATA_DIR = ROOT / "data" / "alpaca_rebuild_v1"
TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
FEATURE_SPEC = ROOT / "models" / "alpaca_final_freeze_v1" / "feature_columns.json"
FROZEN_SPEC = ROOT / "models" / "alpaca_final_freeze_v1" / "frozen_candidate_spec.json"
RAW_SOURCE = ROOT / "models" / "alpaca_1min_research_v1" / "raw_ecl_iex_5min_train_validation.csv"
RAW_SOURCE_META = ROOT / "models" / "alpaca_1min_research_v1" / "fetch_5min_metadata.json"
FOLD_CSV = ROOT / "reports" / "alpaca_binary_research_v3" / "walk_forward_folds.csv"
CLASS_ORDER = ["DOWN", "NEUTRAL", "UP"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
SEED = 42
THRESHOLDS = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]
OUTPUT_DIRS = [REPORT_DIR, MODEL_DIR / "estimators"]

# Fixed read allowlist. No test split path appears in this experiment.
APPROVED_CSVS = {
    TRAIN_CSV.resolve(), VALIDATION_CSV.resolve(), FOLD_CSV.resolve(),
    (ROOT / "reports/alpaca_baselines_v1/validation_predictions.csv").resolve(),
    (ROOT / "reports/alpaca_return_regression_v1/walk_forward_regression_predictions.csv").resolve(),
    (ROOT / "reports/alpaca_return_regression_v1/validation_regression_predictions.csv").resolve(),
    (ROOT / "reports/alpaca_binary_research_v3/walk_forward_oof_predictions.csv").resolve(),
    (ROOT / "reports/alpaca_binary_research_v3/validation_predictions.csv").resolve(),
    RAW_SOURCE.resolve(),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def approved_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    resolved = path.resolve()
    if resolved not in APPROVED_CSVS:
        raise ValueError(f"Unapproved experiment input: {path}")
    return pd.read_csv(resolved, **kwargs)


def approved_read_json(path: Path) -> Any:
    if path.resolve() not in {FEATURE_SPEC.resolve(), FROZEN_SPEC.resolve(), RAW_SOURCE_META.resolve()}:
        raise ValueError(f"Unapproved experiment specification: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def target_from_return(values: np.ndarray | pd.Series) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("Forward returns must be finite")
    return np.where(a > 0.10, "UP", np.where(a < -0.10, "DOWN", "NEUTRAL"))


def ridge_to_classes(predicted_return: np.ndarray | pd.Series, threshold: float) -> np.ndarray:
    a = np.asarray(predicted_return, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Predictions and threshold must be finite; threshold nonnegative")
    return np.where(a > threshold, "UP", np.where(a < -threshold, "DOWN", "NEUTRAL"))


def ridge_zero_sign(predicted_return: np.ndarray | pd.Series) -> np.ndarray:
    """Binary diagnostic only; exact zero deterministically maps to DOWN."""
    a = np.asarray(predicted_return, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("Predicted returns must be finite")
    return np.where(a > 0.0, "UP", "DOWN")


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  probabilities: np.ndarray | None = None) -> dict[str, Any]:
    yt, yp = np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str)
    if len(yt) != len(yp) or not set(yp).issubset(CLASS_ORDER) or not set(yt).issubset(CLASS_ORDER):
        raise ValueError("Class output or vector lengths are invalid")
    report = classification_report(yt, yp, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    actual_dir = np.isin(yt, ["DOWN", "UP"])
    predicted_dir = np.isin(yp, ["DOWN", "UP"])
    emit = predicted_dir
    out: dict[str, Any] = {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "per_class": {c: {"precision": float(report[c]["precision"]), "recall": float(report[c]["recall"]),
                          "f1": float(report[c]["f1-score"]), "support": int(report[c]["support"])} for c in CLASS_ORDER},
        "directional_macro_precision": float(np.mean([report[c]["precision"] for c in ("DOWN", "UP")])),
        "directional_macro_recall": float(np.mean([report[c]["recall"] for c in ("DOWN", "UP")])),
        "directional_macro_f1": float(np.mean([report[c]["f1-score"] for c in ("DOWN", "UP")])),
        "directional_prediction_coverage_all_rows": float(predicted_dir.mean()) if len(yt) else 0.0,
        "directional_prediction_coverage_actual_directional": float(predicted_dir[actual_dir].mean()) if actual_dir.any() else None,
        "directional_accuracy_actual_directional": float((yp[actual_dir] == yt[actual_dir]).mean()) if actual_dir.any() else None,
        "accuracy_among_directional_predictions": float((yp[emit] == yt[emit]).mean()) if emit.any() else None,
        "prediction_distribution": {c: int((yp == c).sum()) for c in CLASS_ORDER},
        "confusion_matrix_actual_rows_predicted_columns": confusion_matrix(yt, yp, labels=CLASS_ORDER).tolist(),
    }
    if probabilities is not None:
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (len(yt), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Multiclass probabilities invalid")
        out["log_loss"] = float(log_loss(yt, p, labels=CLASS_ORDER))
    else:
        out["log_loss"] = None
    return out


def threshold_guardrail(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    m = class_metrics(y_true, y_pred)
    actual_neutral = float(np.mean(np.asarray(y_true, dtype=str) == "NEUTRAL"))
    dist = m["prediction_distribution"]
    shares = {c: dist[c] / max(m["n"], 1) for c in CLASS_ORDER}
    min_directional = max(0.40, 1.0 - actual_neutral - 0.10)
    passed = (m["directional_prediction_coverage_all_rows"] >= min_directional
              and shares["NEUTRAL"] <= actual_neutral + 0.05
              and min(shares.values()) >= 0.05)
    return {"passed": bool(passed), "minimum_directional_coverage": min_directional,
            "actual_neutral_share": actual_neutral, "predicted_class_share": shares}


def select_ridge_threshold(y_oof: np.ndarray, return_oof: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        pred = ridge_to_classes(return_oof, threshold)
        score = class_metrics(y_oof, pred)
        guard = threshold_guardrail(y_oof, pred)
        rows.append({"threshold_percentage_points": threshold, **{k: score[k] for k in (
            "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "directional_macro_f1",
            "directional_prediction_coverage_all_rows", "prediction_distribution", "confusion_matrix_actual_rows_predicted_columns")},
            "guardrail": guard})
    feasible = [r for r in rows if r["guardrail"]["passed"]]
    pool = feasible or rows
    best = max(pool, key=lambda r: (r["macro_f1"], r["balanced_accuracy"], r["directional_macro_f1"],
                                    -r["guardrail"]["predicted_class_share"]["NEUTRAL"],
                                    -r["threshold_percentage_points"]))
    return float(best["threshold_percentage_points"]), rows


def make_pipelines() -> dict[str, Pipeline]:
    return {
        "ridge_regression_to_3class": Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "xgboost_multiclass": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=250, max_depth=3, min_child_weight=5,
                learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                objective="multi:softprob", num_class=3, eval_metric="mlogloss", tree_method="hist",
                random_state=SEED, n_jobs=1, verbosity=0))]),
        "hist_gradient_boosting_multiclass": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=SEED))]),
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    feature_spec = approved_read_json(FEATURE_SPEC)
    frozen = approved_read_json(FROZEN_SPEC)
    features = feature_spec if isinstance(feature_spec, list) else feature_spec.get("feature_columns", feature_spec.get("features", []))
    if isinstance(features, dict):
        features = features.get("feature_order", [])
    expected_features = frozen["features_exact_order"]
    if features != expected_features or len(features) != 24 or len(set(features)) != 24:
        raise ValueError("Feature schema does not match exact frozen 24-feature contract")
    train = approved_read_csv(TRAIN_CSV)
    val = approved_read_csv(VALIDATION_CSV)
    raw_meta = approved_read_json(RAW_SOURCE_META)
    if (raw_meta.get("symbol", "").upper(), raw_meta.get("feed", "").lower(), raw_meta.get("timeframe")) != ("ECL", "iex", "5Min"):
        raise ValueError("Approved raw cache provenance is not ECL Alpaca IEX 5Min")
    if raw_meta.get("permitted_new_york_date_range") != ["2025-11-13", "2026-09-24"]:
        raise ValueError("Raw source provenance extends outside the approved pre-test period")
    if sha256(RAW_SOURCE) != raw_meta.get("raw_csv_sha256"):
        raise ValueError("Permitted raw source checksum mismatch")
    raw = approved_read_csv(RAW_SOURCE)
    if not {"timestamp_utc", "close"}.issubset(raw.columns):
        raise ValueError("Raw source must contain UTC timestamps and close prices")
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True, errors="raise")
    raw["datetime"] = raw.timestamp_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    if raw.datetime.duplicated().any() or not raw.datetime.is_monotonic_increasing:
        raise ValueError("Permitted raw source timestamps are not unique and chronological")
    if raw.datetime.max() > pd.Timestamp("2026-09-24 23:59:59"):
        raise ValueError("Permitted raw source has rows after the validation cutoff")
    source_times = pd.DatetimeIndex(raw.datetime)
    close_by_time = pd.Series(pd.to_numeric(raw.close, errors="raise").to_numpy(float), index=source_times)
    # The canonical rebuild CSV stores a label-reference close and then the
    # model's `close` feature, so the raw CSV header contains `close` twice.
    # pandas names the second duplicate `close.1`; verify they agree, then make
    # the two distinct roles explicit before selecting model features.
    expected_raw = ["datetime", "future_datetime", "close", "future_return", "target", "close.1", *features[1:]]
    for split, frame in (("train", train), ("validation", val)):
        if list(frame.columns) != expected_raw:
            raise ValueError(f"{split} columns/order differ from frozen contract")
        if not np.allclose(pd.to_numeric(frame["close"], errors="coerce"),
                           pd.to_numeric(frame["close.1"], errors="coerce"), rtol=0, atol=0):
            raise ValueError(f"{split} reference close differs from the frozen model close feature")
        frame = frame.rename(columns={"close": "reference_close", "close.1": "close"})
        if split == "train":
            train = frame
        else:
            val = frame
        frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
        frame["future_datetime"] = pd.to_datetime(frame.future_datetime, errors="raise")
        if frame.datetime.duplicated().any() or not frame.datetime.is_monotonic_increasing:
            raise ValueError(f"{split} decisions must be unique and chronological")
        if not ((frame.future_datetime - frame.datetime) == pd.Timedelta(minutes=15)).all():
            raise ValueError(f"{split} has non-15-minute target endpoints")
        if (frame.datetime.dt.date != frame.future_datetime.dt.date).any():
            raise ValueError(f"{split} target crosses a New York trading date")
        if not np.array_equal(target_from_return(frame.future_return), frame.target.astype(str).to_numpy()):
            raise ValueError(f"{split} labels disagree with saved forward returns / strict threshold")
        decision_times = pd.DatetimeIndex(frame.datetime)
        future_times = pd.DatetimeIndex(frame.future_datetime)
        verify_continuous_target_paths(decision_times, future_times, source_times, split)
        raw_reference = decision_times.map(close_by_time)
        raw_endpoint = future_times.map(close_by_time)
        if not np.allclose(raw_reference.to_numpy(float), frame.reference_close.to_numpy(float), rtol=0, atol=1e-9):
            raise ValueError(f"{split} decision reference close differs from raw Alpaca cache")
        raw_return = (raw_endpoint.to_numpy(float) / raw_reference.to_numpy(float) - 1.0) * 100.0
        if not np.allclose(raw_return, frame.future_return.to_numpy(float), rtol=0, atol=1e-9):
            raise ValueError(f"{split} future return differs from raw endpoint close")
        x = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError(f"{split} feature matrix has nonfinite values")
        if frame.target.isna().any() or not frame.target.isin(CLASS_ORDER).all():
            raise ValueError(f"{split} contains invalid classes")
    if train.datetime.max() >= val.datetime.min():
        raise ValueError("Train and validation are not strictly chronological")
    if train.future_datetime.max() >= val.datetime.min():
        raise ValueError("Training target endpoint overlaps validation boundary")
    folds = approved_read_csv(FOLD_CSV)
    required_fold_cols = {"fold", "train_end", "validation_start", "validation_end"}
    if not required_fold_cols.issubset(folds.columns) or folds.fold.tolist() != [1, 2, 3, 4]:
        raise ValueError("Expected established four-fold chronological schedule")
    return train, val, folds, features, {"feature_spec": feature_spec, "frozen_spec": frozen, "raw_source_metadata": raw_meta}


def fold_masks(frame: pd.DataFrame, train_end: str, val_start: str, val_end: str) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    dates = frame.datetime.dt.strftime("%Y-%m-%d")
    eval_mask = dates.between(val_start, val_end)
    if not eval_mask.any():
        raise ValueError("Fold evaluation partition is empty")
    first_eval = frame.loc[eval_mask, "datetime"].min()
    train_mask = (dates <= train_end) & (frame.future_datetime < first_eval)
    if not train_mask.any() or frame.loc[train_mask, "future_datetime"].max() >= first_eval:
        raise ValueError("Fold purge failed")
    return train_mask.to_numpy(), eval_mask.to_numpy(), first_eval


def verify_continuous_target_paths(decision_times: pd.DatetimeIndex, future_times: pd.DatetimeIndex,
                                  source_times: pd.DatetimeIndex, split: str) -> None:
    """Verify exact +15m endpoint and complete same-session five-minute path."""
    if len(decision_times) != len(future_times):
        raise ValueError(f"{split} decision and endpoint lengths differ")
    if not ((future_times - decision_times) == pd.Timedelta(minutes=15)).all():
        raise ValueError(f"{split} endpoint is not exactly +15 minutes")
    if not np.array_equal(decision_times.date, future_times.date):
        raise ValueError(f"{split} target crosses a New York session date")
    if not decision_times.isin(source_times).all() or not future_times.isin(source_times).all():
        raise ValueError(f"{split} decision or endpoint candle is absent from raw source")
    for minutes_ahead in (5, 10, 15):
        if not (decision_times + pd.Timedelta(minutes=minutes_ahead)).isin(source_times).all():
            raise ValueError(f"{split} path crosses a missing five-minute candle at +{minutes_ahead}m")


def fit_model(name: str, x: pd.DataFrame, y: np.ndarray, y_return: np.ndarray) -> Pipeline:
    model = make_pipelines()[name]
    if name == "ridge_regression_to_3class":
        model.fit(x, y_return)
    else:
        weights = compute_sample_weight(class_weight="balanced", y=np.asarray([CLASS_TO_INT[c] for c in y]))
        model.fit(x, np.asarray([CLASS_TO_INT[c] for c in y]), model__sample_weight=weights)
    return model


def predict_model(name: str, model: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    if name == "ridge_regression_to_3class":
        return np.asarray(model.predict(x), dtype=float), None
    raw = np.asarray(model.predict_proba(x), dtype=float)
    cls = np.asarray(model.named_steps["model"].classes_)
    p = np.zeros((len(x), 3), dtype=float)
    for j, c in enumerate(cls):
        p[:, int(c)] = raw[:, j]
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError(f"{name} returned invalid probabilities")
    p /= p.sum(axis=1, keepdims=True)
    return np.asarray([CLASS_ORDER[i] for i in np.argmax(p, axis=1)]), p


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def prior_artifact_paths() -> list[Path]:
    candidates = [
        ROOT / "models/alpaca_final_freeze_v1/candidate_pipeline.joblib",
        ROOT / "models/alpaca_baselines_v1/xgboost.joblib",
        ROOT / "models/alpaca_return_regression_v1/ridge_full_train.joblib",
        ROOT / "models/alpaca_binary_research_v3/xgboost.joblib",
    ]
    candidates += sorted((ROOT / "models/alpaca_return_regression_v1").glob("ridge_fold*.joblib"))
    return [p for p in candidates if p.is_file()]


def main() -> dict[str, Any]:
    for outdir in OUTPUT_DIRS:
        outdir.mkdir(parents=True, exist_ok=True)
    # Refuse overwrites of prior outputs; reruns should be explicit and reviewed.
    marker = REPORT_DIR / "validation_metrics.json"
    if marker.exists():
        raise FileExistsError(f"Experiment outputs already exist: {marker}")
    config = json.loads((MODEL_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    train, validation, folds, features, specs = load_data()
    if xgboost.__version__ != "3.4.1":
        raise RuntimeError(f"XGBoost compatibility mismatch: expected 3.4.1, found {xgboost.__version__}")
    catboost_available = bool(importlib.util.find_spec("catboost"))
    if catboost_available:
        # User asked to implement CatBoost when installed; not silently ignored.
        raise NotImplementedError("CatBoost is installed in this environment; add explicit reviewed configuration before executing")

    before_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in prior_artifact_paths()}
    fold_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    trained_fold_models: dict[str, list[Pipeline]] = {name: [] for name in make_pipelines()}
    train_in_sample_records: list[dict[str, Any]] = []
    X_all = train[features]
    y_all = train.target.astype(str).to_numpy()
    r_all = train.future_return.to_numpy(dtype=float)
    for row in folds.itertuples(index=False):
        tr_mask, ev_mask, first_eval = fold_masks(train, row.train_end, row.validation_start, row.validation_end)
        xtr, ytr, rtr = X_all.loc[tr_mask], y_all[tr_mask], r_all[tr_mask]
        xev = X_all.loc[ev_mask]
        yev, rev = y_all[ev_mask], r_all[ev_mask]
        fold = int(row.fold)
        for name in make_pipelines():
            model = fit_model(name, xtr, ytr, rtr)
            train_pred, train_probs = predict_model(name, model, xtr)
            eval_pred, eval_probs = predict_model(name, model, xev)
            trained_fold_models[name].append(model)
            for scope, frame, actual, predicted, probs in (
                ("fit_train", train.loc[tr_mask], ytr, train_pred, train_probs),
                ("fold_oof", train.loc[ev_mask], yev, eval_pred, eval_probs),
            ):
                if name == "ridge_regression_to_3class":
                    vals = np.asarray(predicted, dtype=float)
                    for i, (_, source_row) in enumerate(frame.iterrows()):
                        prediction_records.append({"model": name, "fold": fold, "scope": scope,
                            "datetime": source_row.datetime, "actual": actual[i],
                            "future_return": float(source_row.future_return), "predicted_return": float(vals[i]), "prediction": None})
                else:
                    for i, (_, source_row) in enumerate(frame.iterrows()):
                        prediction_records.append({"model": name, "fold": fold, "scope": scope,
                            "datetime": source_row.datetime, "actual": actual[i],
                            "future_return": float(source_row.future_return), "predicted_return": None,
                            "prediction": str(predicted[i]), "p_down": float(probs[i, 0]),
                            "p_neutral": float(probs[i, 1]), "p_up": float(probs[i, 2])})
            # Training metrics for overfit diagnosis; held-out fold remains OOF.
            train_in_sample_records.append({"model": name, "fold": fold, "n_train": len(ytr),
                "n_oof": len(yev), "first_oof_timestamp": first_eval.isoformat(),
                "purged_train_rows": int(((train.datetime.dt.strftime("%Y-%m-%d") <= row.train_end) & ~tr_mask).sum()),
                "train_targets_before_eval": bool(train.loc[tr_mask, "future_datetime"].max() < first_eval),
                "fit_prediction": train_pred, "fit_probabilities": train_probs,
                "oof_prediction": eval_pred, "oof_probabilities": eval_probs,
                "y_train": ytr, "y_oof": yev})

    # Select a Ridge neutral boundary from pooled training-only OOF estimates.
    ridge_rows = [r for r in prediction_records if r["model"] == "ridge_regression_to_3class" and r["scope"] == "fold_oof"]
    ridge_threshold, threshold_rows = select_ridge_threshold(
        np.asarray([r["actual"] for r in ridge_rows]), np.asarray([r["predicted_return"] for r in ridge_rows], dtype=float))
    for rec in prediction_records:
        if rec["model"] == "ridge_regression_to_3class":
            rec["prediction"] = str(ridge_to_classes([rec["predicted_return"]], ridge_threshold)[0])
            rec["zero_boundary_binary_prediction"] = str(ridge_zero_sign([rec["predicted_return"]])[0])

    pd.DataFrame(prediction_records).to_csv(REPORT_DIR / "walk_forward_predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_json(REPORT_DIR / "ridge_threshold_sweep.json", orient="records", indent=2)

    fold_metrics: list[dict[str, Any]] = []
    for rec in train_in_sample_records:
        name, fold = rec["model"], rec["fold"]
        if name == "ridge_regression_to_3class":
            yp_train = ridge_to_classes(rec["fit_prediction"], ridge_threshold)
            yp_oof = ridge_to_classes(rec["oof_prediction"], ridge_threshold)
            zero_oof = ridge_zero_sign(rec["oof_prediction"])
            actual_dir = np.isin(rec["y_oof"], ["DOWN", "UP"])
            zero_metrics = {"zero_boundary_binary_accuracy_on_directional": float((zero_oof[actual_dir] == rec["y_oof"][actual_dir]).mean()),
                            "zero_boundary_directional_confusion_matrix_down_up": confusion_matrix(rec["y_oof"][actual_dir], zero_oof[actual_dir], labels=["DOWN", "UP"]).tolist()}
            mtrain = class_metrics(rec["y_train"], yp_train)
            moof = class_metrics(rec["y_oof"], yp_oof)
            guard = threshold_guardrail(rec["y_oof"], yp_oof)
            moof.update(zero_metrics)
        else:
            mtrain = class_metrics(rec["y_train"], rec["fit_prediction"], rec["fit_probabilities"])
            moof = class_metrics(rec["y_oof"], rec["oof_prediction"], rec["oof_probabilities"])
            guard = threshold_guardrail(rec["y_oof"], rec["oof_prediction"])
        fold_metrics.append({"model": name, "fold": fold, "n_train": rec["n_train"], "n_oof": rec["n_oof"],
            "first_oof_timestamp": rec["first_oof_timestamp"], "purged_train_rows": rec["purged_train_rows"],
            "train_metrics": mtrain, "oof_metrics": moof, "oof_guardrail": guard})
        # Keep each trained fold estimator for auditable reproducibility.
        joblib.dump(trained_fold_models[name][fold - 1], MODEL_DIR / "estimators" / f"{name}_fold{fold}.joblib")
    save_json(REPORT_DIR / "walk_forward_fold_metrics.json", fold_metrics)

    # Pooled OOF model scores; all model choices are fixed before validation.
    summary_rows = []
    pooled: dict[str, dict[str, Any]] = {}
    for name in make_pipelines():
        records = [r for r in prediction_records if r["model"] == name and r["scope"] == "fold_oof"]
        y = np.asarray([r["actual"] for r in records], dtype=str)
        if name == "ridge_regression_to_3class":
            pred = ridge_to_classes([r["predicted_return"] for r in records], ridge_threshold)
        else:
            pred = np.asarray([r["prediction"] for r in records], dtype=str)
        pooled[name] = class_metrics(y, pred)
        metrics_by_fold = [f["oof_metrics"] for f in fold_metrics if f["model"] == name]
        guard = threshold_guardrail(y, pred)
        pooled[name]["guardrail"] = guard
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "directional_macro_precision", "directional_macro_recall", "directional_macro_f1"):
            vals = [float(m[metric]) for m in metrics_by_fold]
            pooled[name][f"mean_fold_{metric}"] = float(np.mean(vals))
            pooled[name][f"std_fold_{metric}"] = float(np.std(vals, ddof=0))
        summary_rows.append({"model": name, **{k: pooled[name].get(k) for k in (
            "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "directional_macro_precision",
            "directional_macro_recall", "directional_macro_f1", "directional_prediction_coverage_all_rows",
            "mean_fold_accuracy", "std_fold_accuracy", "mean_fold_macro_f1", "std_fold_macro_f1",
            "mean_fold_balanced_accuracy", "std_fold_balanced_accuracy")}, "guardrail_passed": guard["passed"]})
    # Predeclared rank. Guardrail-valid candidates first; if none, diagnostic fallback is explicit.
    valid = [name for name, m in pooled.items() if m["guardrail"]["passed"]]
    candidate_pool = valid or list(pooled)
    selected_name = max(candidate_pool, key=lambda n: (pooled[n]["mean_fold_macro_f1"],
        pooled[n]["mean_fold_balanced_accuracy"], pooled[n]["mean_fold_directional_macro_f1"]))
    selection = {"selected_candidate": selected_name, "selection_metric": "mean chronological OOF fold macro-F1",
        "tie_breaks": ["mean fold balanced accuracy", "mean fold directional macro-F1"],
        "guardrail_passed": pooled[selected_name]["guardrail"]["passed"],
        "ridge_neutral_threshold_percentage_points": ridge_threshold,
        "ridge_zero_boundary_is_binary_only": True,
        "candidates_passing_guardrail": valid,
        "all_candidate_oof_metrics": pooled,
        "validation_seen": False}
    save_json(REPORT_DIR / "selection.json", selection)
    pd.DataFrame(summary_rows).to_csv(REPORT_DIR / "walk_forward_summary.csv", index=False)
    pd.DataFrame([{"model": k, **v["prediction_distribution"]} for k, v in pooled.items()]).to_csv(REPORT_DIR / "oof_prediction_distributions.csv", index=False)

    # Fit and save all fold-independent final candidate estimators on TRAIN only.
    # Then freeze the selected model/config/hash before any validation inference.
    X_train, y_train, r_train = train[features], train.target.astype(str).to_numpy(), train.future_return.to_numpy(float)
    final_estimators: dict[str, Pipeline] = {}
    for name in make_pipelines():
        final = fit_model(name, X_train, y_train, r_train)
        final_path = MODEL_DIR / "estimators" / f"{name}_train_only.joblib"
        joblib.dump(final, final_path)
        final_estimators[name] = final
    chosen_path = MODEL_DIR / "estimators" / f"{selected_name}_train_only.joblib"
    frozen_setup = {"selected_candidate": selected_name, "selected_artifact": str(chosen_path.relative_to(ROOT)),
        "artifact_sha256": sha256(chosen_path), "features": features, "class_order": CLASS_ORDER,
        "target": config["target"], "horizon_minutes": 15, "selection": selection,
        "model_config": config["estimators"].get("ridge_regression_to_3class" if selected_name.startswith("ridge") else selected_name),
        "threshold_policy": config["ridge_neutral_threshold_selection"] if selected_name.startswith("ridge") else None,
        "validation_evaluated": False}
    save_json(MODEL_DIR / "frozen_candidate_config.json", frozen_setup)
    frozen_setup_hash = sha256(MODEL_DIR / "frozen_candidate_config.json")

    # Single limited confirmation: chosen candidate only, using the saved frozen train-only artifact.
    selected_model = joblib.load(chosen_path)
    X_val = validation[features]
    if selected_name == "ridge_regression_to_3class":
        val_returns = np.asarray(selected_model.predict(X_val), dtype=float)
        val_pred = ridge_to_classes(val_returns, ridge_threshold)
        val_probs = None
    else:
        val_pred, val_probs = predict_model(selected_name, selected_model, X_val)
        val_returns = None
    y_val = validation.target.astype(str).to_numpy()
    val_metrics = class_metrics(y_val, val_pred, val_probs)
    val_guardrail = threshold_guardrail(y_val, val_pred)
    val_predictions = pd.DataFrame({"datetime": validation.datetime, "future_datetime": validation.future_datetime,
        "actual": y_val, "prediction": val_pred})
    if val_probs is not None:
        val_predictions["p_down"] = val_probs[:, 0]
        val_predictions["p_neutral"] = val_probs[:, 1]
        val_predictions["p_up"] = val_probs[:, 2]
    if val_returns is not None:
        val_predictions["predicted_return_pct_points"] = val_returns
        val_predictions["zero_boundary_binary_prediction"] = ridge_zero_sign(val_returns)
    val_predictions.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)
    val_metrics_payload = {"selected_candidate": selected_name, "n": len(validation),
        "date_start": validation.datetime.min().strftime("%Y-%m-%d"),
        "date_end": validation.datetime.max().strftime("%Y-%m-%d"),
        "class_distribution": validation.target.value_counts().reindex(CLASS_ORDER, fill_value=0).to_dict(),
        "metrics": val_metrics, "guardrail": val_guardrail,
        "artifact_sha256": sha256(chosen_path), "frozen_config_sha256_before_validation": frozen_setup_hash,
        "validation_used_for_selection": False}
    save_json(REPORT_DIR / "validation_metrics.json", val_metrics_payload)
    pd.DataFrame(val_metrics["confusion_matrix_actual_rows_predicted_columns"], index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(REPORT_DIR / "validation_confusion_matrix.csv", index_label="actual\\predicted")

    # Existing model comparisons use stored outputs, never refitting/inference.
    baseline_rows = evaluate_saved_controls(validation, train)
    pd.DataFrame(baseline_rows).to_csv(REPORT_DIR / "saved_baseline_comparison.csv", index=False)

    after_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in prior_artifact_paths()}
    if before_hashes != after_hashes:
        raise RuntimeError("A pre-existing model artifact hash changed during the experiment")
    save_json(MODEL_DIR / "reproducibility_manifest.json", {
        "experiment_config": config, "python": platform.python_version(), "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__, "catboost_installed": catboost_available,
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [TRAIN_CSV, VALIDATION_CSV, FEATURE_SPEC, FROZEN_SPEC, FOLD_CSV]},
        "preexisting_model_hashes_before": before_hashes, "preexisting_model_hashes_after": after_hashes,
        "preexisting_hashes_unchanged": before_hashes == after_hashes,
        "protected_test_access": False, "test_file_in_approved_input_allowlist": False,
        "validation_selection": "chosen only after training OOF model/policy selection; evaluated once",
        "source_target_rows": {"train": len(train), "validation": len(validation)},
        "chronological_boundaries": {"train": [train.datetime.min().isoformat(), train.datetime.max().isoformat()],
            "validation": [validation.datetime.min().isoformat(), validation.datetime.max().isoformat()]},
        "feature_order": features, "class_order": CLASS_ORDER, "selected_artifact_sha256": sha256(chosen_path),
        "frozen_config_sha256": sha256(MODEL_DIR / "frozen_candidate_config.json")})
    return {"selected": selected_name, "validation_metrics": val_metrics_payload, "baseline_comparison": baseline_rows}


def evaluate_saved_controls(validation: pd.DataFrame, train: pd.DataFrame) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    xgb3path = ROOT / "reports/alpaca_baselines_v1/validation_predictions.csv"
    xgb3 = approved_read_csv(xgb3path)
    if "xgboost_prediction" in xgb3.columns:
        xgb3["datetime"] = pd.to_datetime(xgb3.datetime, errors="raise")
        aligned = validation[["datetime", "target"]].merge(xgb3[["datetime", "actual", "xgboost_prediction"]], on="datetime", how="inner", validate="one_to_one")
        if len(aligned) == len(validation) and (aligned.target.astype(str) == aligned.actual.astype(str)).all():
            controls.append({"model": "existing_three_class_xgboost_baseline", "population": "all validation rows",
                **class_metrics(aligned.target.to_numpy(str), aligned.xgboost_prediction.to_numpy(str)), "source": str(xgb3path.relative_to(ROOT)), "inference_rerun": False})

    ridge_path = ROOT / "reports/alpaca_return_regression_v1/validation_regression_predictions.csv"
    ridge = approved_read_csv(ridge_path)
    ridge["datetime"] = pd.to_datetime(ridge.datetime, errors="raise")
    ridge = ridge[ridge.model == "ridge"].copy()
    if not ridge.empty:
        aligned = validation[["datetime", "target"]].merge(ridge[["datetime", "actual_target", "predicted_return"]], on="datetime", how="inner", validate="one_to_one")
        if len(aligned) == len(validation) and (aligned.target.astype(str) == aligned.actual_target.astype(str)).all():
            actual_dir = aligned.target.isin(["DOWN", "UP"]).to_numpy()
            pred = ridge_zero_sign(aligned.predicted_return.to_numpy(float))
            controls.append({"model": "existing_ridge_regression_zero_sign_directional_control", "population": "actual UP/DOWN validation rows",
                **binary_metrics(aligned.target.to_numpy(str)[actual_dir], pred[actual_dir]),
                "source": str(ridge_path.relative_to(ROOT)), "inference_rerun": False})

    binary_path = ROOT / "reports/alpaca_binary_research_v3/validation_predictions.csv"
    binary = approved_read_csv(binary_path)
    if "xgboost_prediction" in binary.columns:
        binary["datetime"] = pd.to_datetime(binary.datetime, errors="raise")
        aligned = validation[["datetime", "target"]].merge(binary[["datetime", "actual_three_class", "xgboost_prediction"]], on="datetime", how="inner", validate="one_to_one")
        if len(aligned) == len(validation) and (aligned.target.astype(str) == aligned.actual_three_class.astype(str)).all():
            actual_dir = aligned.target.isin(["DOWN", "UP"]).to_numpy()
            controls.append({"model": "existing_binary_xgboost_v3_directional_control", "population": "actual UP/DOWN validation rows",
                **binary_metrics(aligned.target.to_numpy(str)[actual_dir], aligned.xgboost_prediction.to_numpy(str)[actual_dir]),
                "source": str(binary_path.relative_to(ROOT)), "inference_rerun": False})

    # Existing training-only OOF controls are joined by timestamp to ensure the
    # same eligible observations; these are saved predictions, not new model runs.
    ridge_oof_path = ROOT / "reports/alpaca_return_regression_v1/walk_forward_regression_predictions.csv"
    ridge_oof = approved_read_csv(ridge_oof_path)
    ridge_oof["datetime"] = pd.to_datetime(ridge_oof.datetime, errors="raise")
    ridge_oof = ridge_oof[ridge_oof.model == "ridge"].copy()
    if not ridge_oof.empty:
        aligned = train[["datetime", "target"]].merge(ridge_oof[["datetime", "fold", "actual_target", "predicted_return"]], on="datetime", how="inner", validate="one_to_one")
        if (aligned.target.astype(str) == aligned.actual_target.astype(str)).all():
            actual_dir = aligned.target.isin(["DOWN", "UP"]).to_numpy()
            pred = ridge_zero_sign(aligned.predicted_return.to_numpy(float))
            controls.append({"model": "existing_ridge_regression_zero_sign_directional_control", "population": "training OOF actual UP/DOWN rows",
                "fold": "pooled", **binary_metrics(aligned.target.to_numpy(str)[actual_dir], pred[actual_dir]),
                "source": str(ridge_oof_path.relative_to(ROOT)), "inference_rerun": False})
            for fold in sorted(aligned.fold.unique()):
                part = aligned[aligned.fold == fold]
                mask = part.target.isin(["DOWN", "UP"]).to_numpy()
                fold_pred = ridge_zero_sign(part.predicted_return.to_numpy(float))
                controls.append({"model": "existing_ridge_regression_zero_sign_directional_control", "population": "training OOF actual UP/DOWN rows",
                    "fold": int(fold), **binary_metrics(part.target.to_numpy(str)[mask], fold_pred[mask]),
                    "source": str(ridge_oof_path.relative_to(ROOT)), "inference_rerun": False})

    xgb_oof_path = ROOT / "reports/alpaca_binary_research_v3/walk_forward_oof_predictions.csv"
    xgb_oof = approved_read_csv(xgb_oof_path)
    xgb_oof["datetime"] = pd.to_datetime(xgb_oof.datetime, errors="raise")
    xgb_oof = xgb_oof[(xgb_oof.family == "xgboost") & (xgb_oof.configuration == "xgb_shallow_unweighted")].copy()
    if not xgb_oof.empty:
        aligned = train[["datetime", "target"]].merge(xgb_oof[["datetime", "fold", "actual_three_class", "prediction"]], on="datetime", how="inner", validate="one_to_one")
        if (aligned.target.astype(str) == aligned.actual_three_class.astype(str)).all():
            actual_dir = aligned.target.isin(["DOWN", "UP"]).to_numpy()
            controls.append({"model": "existing_binary_xgboost_v3_directional_control", "population": "training OOF actual UP/DOWN rows",
                "fold": "pooled", **binary_metrics(aligned.target.to_numpy(str)[actual_dir], aligned.prediction.to_numpy(str)[actual_dir]),
                "source": str(xgb_oof_path.relative_to(ROOT)), "inference_rerun": False})
            for fold in sorted(aligned.fold.unique()):
                part = aligned[aligned.fold == fold]
                mask = part.target.isin(["DOWN", "UP"]).to_numpy()
                controls.append({"model": "existing_binary_xgboost_v3_directional_control", "population": "training OOF actual UP/DOWN rows",
                    "fold": int(fold), **binary_metrics(part.target.to_numpy(str)[mask], part.prediction.to_numpy(str)[mask]),
                    "source": str(xgb_oof_path.relative_to(ROOT)), "inference_rerun": False})
    return controls


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import precision_recall_fscore_support
    labels = ["DOWN", "UP"]
    pr, re, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "down_precision": float(pr[0]), "down_recall": float(re[0]), "down_f1": float(f1[0]),
        "up_precision": float(pr[1]), "up_recall": float(re[1]), "up_f1": float(f1[1]),
        "confusion_matrix_actual_rows_predicted_columns_down_up": confusion_matrix(y, pred, labels=labels).tolist(),
        "prediction_distribution": {c: int((pred == c).sum()) for c in labels}}


if __name__ == "__main__":
    result = main()
    print(json.dumps({"selected_candidate": result["selected"], "validation_metrics": result["validation_metrics"]["metrics"]}, indent=2))
