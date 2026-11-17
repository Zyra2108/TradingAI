"""Small, reproducible 10/15-minute ECL minute-horizon model comparison.

Only the two generated, pre-test train/validation CSVs in this directory are
read. Models and thresholds are fitted from training data; validation is used
once for final confirmation and is not used to revise the procedure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from pipeline import FEATURES, OUT

ROOT = OUT.resolve()
ARTIFACTS = ROOT / "artifacts"
CLASSES = ("DOWN", "NO_SIGNAL", "UP")
CLASS_TO_ID = {label: i for i, label in enumerate(CLASSES)}
MAX_ALLOWED_DATE = date(2026, 9, 24)
TRAIN_END = date(2026, 8, 5)
VALIDATION_START = date(2026, 8, 6)
SEED = 20261114
MIN_SIGNAL_COVERAGE = 0.20
MIN_SIGNAL_COUNT = 30
THRESHOLD_GRID = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MODEL_NAMES = ("logistic_regression", "lightgbm", "xgboost")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    params: dict


def model_specs() -> tuple[ModelSpec, ...]:
    """Fixed modest settings, declared once and shared by both horizons."""
    return (
        ModelSpec("logistic_regression", "sklearn", {
            "C": 0.3, "max_iter": 1500, "solver": "lbfgs", "class_weight": "balanced",
            "random_state": SEED,
        }),
        ModelSpec("lightgbm", "lightgbm", {
            "n_estimators": 120, "learning_rate": 0.05, "num_leaves": 7,
            "min_child_samples": 30, "reg_lambda": 2.0, "class_weight": "balanced",
            "random_state": SEED, "n_jobs": 1, "verbosity": -1,
            "deterministic": True, "force_col_wise": True,
        }),
        ModelSpec("xgboost", "xgboost", {
            "n_estimators": 120, "max_depth": 3, "learning_rate": 0.05,
            "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_lambda": 2.0, "objective": "multi:softprob", "num_class": 3,
            "eval_metric": "mlogloss", "tree_method": "hist", "n_jobs": 1,
            "random_state": SEED,
        }),
    )


def load_dataset(horizon: int) -> pd.DataFrame:
    """Read exactly one allowlisted dataset and enforce its date/split contract."""
    if horizon not in (10, 15):
        raise ValueError("Only the generated 10- and 15-minute datasets are allowed")
    path = (ROOT / f"dataset_{horizon}min.csv").resolve()
    if path.parent != ROOT or not path.is_file():
        raise ValueError("Expected dataset is missing or outside the research directory")
    frame = pd.read_csv(path, parse_dates=["datetime", "target_end"])
    required = {"datetime", "target_end", "session_date", "split", "feature_ready", "label", *FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError("Dataset does not match the declared feature/target schema")
    if frame.datetime.dt.date.max() > MAX_ALLOWED_DATE:
        raise ValueError("Dataset contains dates outside the permitted pre-test period")
    if set(frame.split.dropna().unique()) - {"train", "validation", "excluded", "purged"}:
        raise ValueError("Unexpected split in generated dataset")
    if frame.loc[frame.split.eq("train"), "datetime"].dt.date.max() != TRAIN_END:
        raise ValueError("Training boundary differs from the approved split")
    if frame.loc[frame.split.eq("validation"), "datetime"].dt.date.min() != VALIDATION_START:
        raise ValueError("Validation boundary differs from the approved split")
    if frame.datetime.duplicated().any() or not frame.datetime.is_monotonic_increasing:
        raise ValueError("Dataset timestamps must be unique and chronological")
    frame["feature_ready"] = frame.feature_ready.astype(str).str.lower().eq("true")
    return frame


def eligible(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    part = frame.loc[frame.split.eq(split) & frame.feature_ready & frame.label.isin(CLASSES)].copy()
    if part.empty:
        raise ValueError(f"No eligible rows for {split}")
    x = part[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Eligible feature rows must be finite")
    return part.reset_index(drop=True)


def chronological_folds(train: pd.DataFrame, horizon: int, n_folds: int = 3):
    """Three expanding date folds; remove any training labels reaching the fold."""
    dates = np.array(sorted(train.session_date.astype(str).unique()), dtype=object)
    initial = max(2, int(np.floor(len(dates) * 0.50)))
    tail = dates[initial:]
    blocks = [b for b in np.array_split(tail, n_folds) if len(b)]
    for fold_id, block in enumerate(blocks, start=1):
        valid_start_date = str(block[0])
        validation = train.loc[train.session_date.astype(str).isin(block)].copy()
        boundary = validation.datetime.min()
        candidate = train.loc[train.datetime < boundary].copy()
        # Exact endpoint labels are purged if their future close reaches the
        # first held-out decision timestamp. This remains explicit even though
        # same-session targets normally make cross-date overlap impossible.
        fitted = candidate.loc[candidate.target_end < boundary].copy()
        if fitted.empty or validation.empty:
            continue
        yield fold_id, valid_start_date, fitted.reset_index(drop=True), validation.reset_index(drop=True)


def _make_pipeline(spec: ModelSpec) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if spec.name == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
        estimator = LogisticRegression(**spec.params)
    elif spec.name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM is not installed; no package was installed") from exc
        estimator = LGBMClassifier(objective="multiclass", num_class=len(CLASSES), **spec.params)
    else:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost is not installed; no package was installed") from exc
        estimator = XGBClassifier(**spec.params)
    steps.append(("model", estimator))
    return Pipeline(steps)


def _fit(model: Pipeline, x: pd.DataFrame, y: pd.Series, spec: ModelSpec) -> Pipeline:
    y_id = y.map(CLASS_TO_ID).to_numpy(dtype=np.int64)
    fit_params = {}
    if spec.name == "xgboost":
        fit_params["model__sample_weight"] = compute_sample_weight("balanced", y_id)
    model.fit(x[FEATURES], y_id, **fit_params)
    return model


def probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    # All estimators receive stable integer classes 0=DOWN, 1=NO_SIGNAL, 2=UP.
    p = np.asarray(model.predict_proba(x[FEATURES]), dtype=np.float64)
    if p.shape != (len(x), len(CLASSES)) or not np.isfinite(p).all():
        raise ValueError("Invalid probability matrix shape or non-finite output")
    if (p < -1e-12).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Model probabilities are invalid")
    return p


def raw_predictions(p: np.ndarray) -> np.ndarray:
    return np.asarray(CLASSES, dtype=object)[np.argmax(p, axis=1)]


def apply_confidence_threshold(p: np.ndarray, threshold: float) -> np.ndarray:
    """Keep raw directional argmax only when its own class probability clears threshold."""
    pred = raw_predictions(p).copy()
    directional = (pred == "UP") | (pred == "DOWN")
    winning_prob = p[np.arange(len(p)), np.argmax(p, axis=1)]
    pred[directional & (winning_prob < threshold)] = "NO_SIGNAL"
    return pred


def select_threshold(y_true: np.ndarray, p: np.ndarray) -> dict:
    """Fixed threshold grid; optimize training OOF macro-F1 with coverage floor."""
    y_true = np.asarray(y_true, dtype=object)
    records = []
    for threshold in THRESHOLD_GRID:
        pred = apply_confidence_threshold(p, threshold)
        coverage = float(np.isin(pred, ["UP", "DOWN"]).mean())
        signals = int(np.isin(pred, ["UP", "DOWN"]).sum())
        score = float(f1_score(y_true, pred, labels=CLASSES, average="macro", zero_division=0))
        records.append({"threshold": threshold, "macro_f1": score, "signal_coverage": coverage, "signals": signals})
    feasible = [r for r in records if r["signal_coverage"] >= MIN_SIGNAL_COVERAGE and r["signals"] >= MIN_SIGNAL_COUNT]
    pool = feasible or records
    best = sorted(pool, key=lambda r: (-r["macro_f1"], -r["signal_coverage"], r["threshold"]))[0]
    return {"selected": best, "coverage_floor_met": bool(feasible), "grid_results": records,
            "selection_source": "pooled chronological training OOF only"}


def calculate_metrics(y_true, y_pred, p=None) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    pr, rc, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    by_class = {c: {"precision": float(pr[i]), "recall": float(rc[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
                for i, c in enumerate(CLASSES)}
    emitted = np.isin(y_pred, ["UP", "DOWN"])
    actual_directional = np.isin(y_true, ["UP", "DOWN"])
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "up_precision": by_class["UP"]["precision"], "up_recall": by_class["UP"]["recall"],
        "down_precision": by_class["DOWN"]["precision"], "down_recall": by_class["DOWN"]["recall"],
        "directional_macro_precision": float(np.mean([by_class["UP"]["precision"], by_class["DOWN"]["precision"]])),
        "directional_macro_recall": float(np.mean([by_class["UP"]["recall"], by_class["DOWN"]["recall"]])),
        "directional_macro_f1": float(np.mean([by_class["UP"]["f1"], by_class["DOWN"]["f1"]])),
        "signal_coverage_all_eligible": float(emitted.mean()) if len(y_true) else 0.0,
        "eligible_signal_count": int(emitted.sum()),
        "eligible_signal_accuracy": float((y_pred[emitted] == y_true[emitted]).mean()) if emitted.any() else 0.0,
        "directional_accuracy_on_actual_directional": float((y_pred[actual_directional] == y_true[actual_directional]).mean()) if actual_directional.any() else 0.0,
        "actual_directional_count": int(actual_directional.sum()),
        "no_signal_prediction_count": int((y_pred == "NO_SIGNAL").sum()),
        "actual_distribution": pd.Series(y_true).value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict(),
        "predicted_distribution": pd.Series(y_pred).value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict(),
        "per_class": by_class,
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_class_order": list(CLASSES),
        "log_loss": None,
    }
    if p is not None:
        encoded = np.asarray([CLASS_TO_ID[label] for label in y_true], dtype=np.int64)
        result["log_loss"] = float(log_loss(encoded, p, labels=list(range(len(CLASSES)))))
    return result


def _prediction_frame(h, model_name, scope, source, p, threshold=None, fold=None):
    y = source.label.to_numpy(dtype=object)
    raw = raw_predictions(p)
    selected = apply_confidence_threshold(p, threshold) if threshold is not None else raw.copy()
    result = pd.DataFrame({
        "horizon_min": h, "model": model_name, "scope": scope, "fold": fold,
        "datetime": source.datetime.to_numpy(), "target_end": source.target_end.to_numpy(),
        "actual": y, "raw_prediction": raw, "policy_prediction": selected,
        "selected_threshold": threshold,
        "p_down": p[:, 0], "p_no_signal": p[:, 1], "p_up": p[:, 2],
    })
    return result


def _add_metric_row(rows, *, horizon, model, scope, policy, y, pred, p=None, fold=None):
    metrics = calculate_metrics(y, pred, p)
    rows.append({"horizon_min": horizon, "model": model, "scope": scope, "policy": policy,
                 "fold": fold, **{k: v for k, v in metrics.items() if k not in {"per_class", "confusion_matrix", "confusion_class_order", "actual_distribution", "predicted_distribution"}}})
    return metrics


def _add_confusion_rows(rows, scope, horizon, model, policy, metrics):
    cm = metrics["confusion_matrix"]
    for i, true_class in enumerate(CLASSES):
        for j, pred_class in enumerate(CLASSES):
            rows.append({"scope": scope, "horizon_min": horizon, "model": model,
                         "policy": policy, "actual": true_class, "predicted": pred_class,
                         "count": int(cm[i][j])})


def _majority_prediction(train: pd.DataFrame, size: int) -> np.ndarray:
    winner = train.label.value_counts().reindex(CLASSES, fill_value=0).idxmax()
    return np.full(size, winner, dtype=object)


def _momentum_prediction(frame: pd.DataFrame) -> np.ndarray:
    momentum = frame["return_10_pct"].to_numpy(dtype=np.float64)
    return np.where(momentum > 0, "UP", np.where(momentum < 0, "DOWN", "NO_SIGNAL")).astype(object)


def _date_metrics(predictions: pd.DataFrame) -> list[dict]:
    rows = []
    for (h, model, policy), group in predictions.groupby(["horizon_min", "model", "policy"], sort=True):
        for day, d in group.groupby(group.datetime.dt.date, sort=True):
            m = calculate_metrics(d.actual, d.prediction)
            rows.append({"horizon_min": int(h), "model": model, "policy": policy,
                         "date": str(day), "n": m["n"], "accuracy": m["accuracy"],
                         "macro_f1": m["macro_f1"], "directional_macro_precision": m["directional_macro_precision"],
                         "directional_macro_f1": m["directional_macro_f1"],
                         "signal_coverage_all_eligible": m["signal_coverage_all_eligible"],
                         "eligible_signal_accuracy": m["eligible_signal_accuracy"]})
    return rows


def run_research() -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    specs = model_specs()
    all_oof, all_val, fold_rows, val_rows, paired_rows, confusion_rows = [], [], [], [], [], []
    summary = {"class_order": list(CLASSES), "feature_order": FEATURES, "seed": SEED,
               "threshold_grid": list(THRESHOLD_GRID), "minimum_signal_coverage": MIN_SIGNAL_COVERAGE,
               "minimum_signal_count": MIN_SIGNAL_COUNT, "horizons": {}}
    val_predictions_by_h = {}
    baseline_predictions_by_h = {}

    for horizon in (10, 15):
        full = load_dataset(horizon)
        train, validation = eligible(full, "train"), eligible(full, "validation")
        folds = list(chronological_folds(train, horizon))
        if len(folds) != 3:
            raise ValueError(f"Expected 3 chronological folds, got {len(folds)} for {horizon}m")
        horizon_summary = {
            "dataset_rows": int(len(full)), "feature_ready_rows": int(full.feature_ready.sum()),
            "warmup_or_missing_feature_rows": int((~full.feature_ready).sum()),
            "train_eligible_rows": len(train), "validation_eligible_rows": len(validation),
            "train_date_range": [str(train.datetime.dt.date.min()), str(train.datetime.dt.date.max())],
            "validation_date_range": [str(validation.datetime.dt.date.min()), str(validation.datetime.dt.date.max())],
            "train_eligible_dates": int(train.datetime.dt.date.nunique()),
            "validation_eligible_dates": int(validation.datetime.dt.date.nunique()),
            "validation_actual_distribution": validation.label.value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict(),
            "target_unavailable_among_feature_ready": int((full.split.isin(["train", "validation"])
                & full.feature_ready & ~full.label.isin(CLASSES)).sum()),
            "feature_ready_missing_endpoint": int((full.split.isin(["train", "validation"])
                & full.feature_ready & full.exclusion_reason.eq("missing_exact_endpoint")).sum()),
            "feature_ready_session_boundary": int((full.split.isin(["train", "validation"])
                & full.feature_ready & full.exclusion_reason.eq("session_or_endpoint_boundary")).sum()),
            "models": {},
        }
        y_train = train.label.to_numpy(dtype=object)
        y_valid = validation.label.to_numpy(dtype=object)

        # Two simple, deterministic comparison baselines; majority class is
        # learned from each fold/full training segment only.
        for fold_id, valid_start, fold_train, fold_val in folds:
            pred_major = _majority_prediction(fold_train, len(fold_val))
            m = calculate_metrics(fold_val.label, pred_major)
            fold_rows.append({"horizon_min": horizon, "model": "majority_baseline", "fold": fold_id,
                              "fold_start_date": valid_start, **_flat_metrics(m)})
            all_oof.append(pd.DataFrame({
                "horizon_min": horizon, "model": "majority_baseline", "scope": "training_oof",
                "fold": fold_id, "datetime": fold_val.datetime.to_numpy(),
                "target_end": fold_val.target_end.to_numpy(), "actual": fold_val.label.to_numpy(),
                "raw_prediction": pred_major, "p_down": np.nan, "p_no_signal": np.nan, "p_up": np.nan,
            }))
            pred_mom = _momentum_prediction(fold_val)
            m = calculate_metrics(fold_val.label, pred_mom)
            fold_rows.append({"horizon_min": horizon, "model": "return_10_sign_baseline", "fold": fold_id,
                              "fold_start_date": valid_start, **_flat_metrics(m)})
            all_oof.append(pd.DataFrame({
                "horizon_min": horizon, "model": "return_10_sign_baseline", "scope": "training_oof",
                "fold": fold_id, "datetime": fold_val.datetime.to_numpy(),
                "target_end": fold_val.target_end.to_numpy(), "actual": fold_val.label.to_numpy(),
                "raw_prediction": pred_mom, "p_down": np.nan, "p_no_signal": np.nan, "p_up": np.nan,
            }))

        valid_major = _majority_prediction(train, len(validation))
        valid_momentum = _momentum_prediction(validation)
        baseline_predictions_by_h[horizon] = {
            "majority_baseline": valid_major,
            "return_10_sign_baseline": valid_momentum,
        }
        for name, pred in (("majority_baseline", valid_major), ("return_10_sign_baseline", valid_momentum)):
            m = _add_metric_row(val_rows, horizon=horizon, model=name, scope="validation", policy="raw", y=y_valid, pred=pred)
            _add_confusion_rows(confusion_rows, "validation", horizon, name, "raw", m)
            all_val.append(pd.DataFrame({"horizon_min": horizon, "model": name, "policy": "raw",
                                         "datetime": validation.datetime, "actual": y_valid, "prediction": pred}))

        for spec in specs:
            fold_predictions = []
            fold_raw_metrics = []
            for fold_id, valid_start, fold_train, fold_val in folds:
                model = _fit(_make_pipeline(spec), fold_train[FEATURES], fold_train.label, spec)
                probs = probabilities(model, fold_val)
                pred = raw_predictions(probs)
                pframe = _prediction_frame(horizon, spec.name, "training_oof", fold_val, probs, fold=fold_id)
                pframe["fold_start_date"] = valid_start
                fold_predictions.append(pframe)
                m = calculate_metrics(fold_val.label, pred, probs)
                fold_raw_metrics.append(m)
                fold_rows.append({"horizon_min": horizon, "model": spec.name, "fold": fold_id,
                                  "fold_start_date": valid_start, **_flat_metrics(m)})
            oof = pd.concat(fold_predictions, ignore_index=True).sort_values("datetime", kind="stable").reset_index(drop=True)
            all_oof.append(oof)
            oof_probs = oof[["p_down", "p_no_signal", "p_up"]].to_numpy(dtype=np.float64)
            threshold_info = select_threshold(oof.actual.to_numpy(dtype=object), oof_probs)
            threshold = float(threshold_info["selected"]["threshold"])

            # Threshold and preprocessing are selected/fitted with training only.
            full_model = _fit(_make_pipeline(spec), train[FEATURES], train.label, spec)
            valid_probs = probabilities(full_model, validation)
            valid_pred_raw = raw_predictions(valid_probs)
            valid_pred_policy = apply_confidence_threshold(valid_probs, threshold)
            val_frame = _prediction_frame(horizon, spec.name, "validation", validation,
                                          valid_probs, threshold=threshold)
            val_predictions_by_h.setdefault(horizon, {})[spec.name] = val_frame
            all_val.append(val_frame.assign(policy="raw", prediction=valid_pred_raw))
            all_val.append(val_frame.assign(policy="oof_selected", prediction=valid_pred_policy))

            raw_metrics = _add_metric_row(val_rows, horizon=horizon, model=spec.name, scope="validation",
                                          policy="raw_argmax", y=y_valid, pred=valid_pred_raw, p=valid_probs)
            _add_confusion_rows(confusion_rows, "validation", horizon, spec.name, "raw_argmax", raw_metrics)
            policy_metrics = _add_metric_row(val_rows, horizon=horizon, model=spec.name, scope="validation",
                                             policy="training_oof_threshold", y=y_valid,
                                             pred=valid_pred_policy, p=valid_probs)
            _add_confusion_rows(confusion_rows, "validation", horizon, spec.name,
                                "training_oof_threshold", policy_metrics)

            pooled_oof_raw = calculate_metrics(oof.actual, oof.raw_prediction,
                                               oof[["p_down", "p_no_signal", "p_up"]].to_numpy())
            horizon_summary["models"][spec.name] = {
                "threshold_selection": threshold_info,
                "training_oof_raw_metrics": pooled_oof_raw,
                "training_oof_fold_macro_f1": [m["macro_f1"] for m in fold_raw_metrics],
                "validation_raw_metrics": raw_metrics,
                "validation_threshold_metrics": policy_metrics,
                "artifact": str(ARTIFACTS / f"{horizon}min_{spec.name}.joblib"),
                "parameters": spec.params,
            }
            joblib.dump(full_model, ARTIFACTS / f"{horizon}min_{spec.name}.joblib")

        summary["horizons"][str(horizon)] = horizon_summary

    # Fair paired comparison on timestamps having valid labels for both horizons.
    ten = load_dataset(10)
    fifteen = load_dataset(15)
    ten_valid = eligible(ten, "validation").set_index("datetime")
    fifteen_valid = eligible(fifteen, "validation").set_index("datetime")
    common = ten_valid.index.intersection(fifteen_valid.index)
    for horizon, val_df in ((10, ten_valid.loc[common].reset_index()), (15, fifteen_valid.loc[common].reset_index())):
        full_val = eligible(load_dataset(horizon), "validation").set_index("datetime")
        for name, predictions_for_horizon in baseline_predictions_by_h[horizon].items():
            baseline = pd.DataFrame({
                "actual": full_val.label,
                "prediction": predictions_for_horizon,
            }, index=full_val.index).loc[common]
            m = calculate_metrics(baseline.actual, baseline.prediction)
            paired_rows.append({"horizon_min": horizon, "model": name, "policy": "raw",
                                "n_common": len(common), **_flat_metrics(m)})
            _add_confusion_rows(confusion_rows, "paired_validation_common_timestamps", horizon, name, "raw", m)
        for name, preds in val_predictions_by_h[horizon].items():
            subset = preds.set_index("datetime").loc[common].reset_index()
            for policy, column in (("raw_argmax", "raw_prediction"),):
                m = calculate_metrics(subset.actual, subset[column], subset[["p_down", "p_no_signal", "p_up"]].to_numpy())
                paired_rows.append({"horizon_min": horizon, "model": name, "policy": policy, "n_common": len(common), **_flat_metrics(m)})
                _add_confusion_rows(confusion_rows, "paired_validation_common_timestamps", horizon, name, policy, m)
            raw_prob = subset[["p_down", "p_no_signal", "p_up"]].to_numpy()
            threshold = float(summary["horizons"][str(horizon)]["models"][name]["threshold_selection"]["selected"]["threshold"])
            pred_policy = apply_confidence_threshold(raw_prob, threshold)
            m = calculate_metrics(subset.actual, pred_policy, raw_prob)
            paired_rows.append({"horizon_min": horizon, "model": name, "policy": "training_oof_threshold", "n_common": len(common), **_flat_metrics(m)})
            _add_confusion_rows(confusion_rows, "paired_validation_common_timestamps", horizon, name, "training_oof_threshold", m)
    summary["paired_validation_common_timestamp_count"] = int(len(common))

    oof_frame = pd.concat(all_oof, ignore_index=True)
    val_frame = pd.concat(all_val, ignore_index=True)
    date_frame = pd.DataFrame(_date_metrics(val_frame))
    fold_frame = pd.DataFrame(fold_rows)
    validation_metric_frame = pd.DataFrame(val_rows)
    paired_frame = pd.DataFrame(paired_rows)
    confusion_frame = pd.DataFrame(confusion_rows)

    training_oof_rows = []
    for (h, model_name), group in oof_frame.groupby(["horizon_min", "model"], sort=True):
        pcols = group[["p_down", "p_no_signal", "p_up"]].to_numpy(dtype=np.float64)
        p = pcols if np.isfinite(pcols).all() else None
        metrics = calculate_metrics(group.actual, group.raw_prediction, p)
        training_oof_rows.append({"horizon_min": int(h), "model": model_name,
                                  "scope": "training_oof_raw", **_flat_metrics(metrics)})
    training_oof_frame = pd.DataFrame(training_oof_rows)
    date_summary = (date_frame.groupby(["horizon_min", "model", "policy"], as_index=False)
        .agg(date_count=("date", "count"), dates_with_10plus=("n", lambda x: int((x >= 10).sum())),
             min_date_n=("n", "min"), max_date_n=("n", "max"),
             median_daily_accuracy=("accuracy", "median"), min_daily_accuracy=("accuracy", "min"),
             max_daily_accuracy=("accuracy", "max"), median_daily_macro_f1=("macro_f1", "median"),
             median_daily_directional_macro_precision=("directional_macro_precision", "median"),
             median_daily_signal_coverage=("signal_coverage_all_eligible", "median")))

    # Write only below the independent research directory.
    oof_frame.to_csv(ROOT / "oof_predictions.csv", index=False)
    val_frame.to_csv(ROOT / "validation_predictions.csv", index=False)
    training_oof_frame.to_csv(ROOT / "training_oof_metrics.csv", index=False)
    fold_frame.to_csv(ROOT / "fold_metrics.csv", index=False)
    validation_metric_frame.to_csv(ROOT / "validation_metrics.csv", index=False)
    paired_frame.to_csv(ROOT / "paired_validation_metrics.csv", index=False)
    date_frame.to_csv(ROOT / "date_stability.csv", index=False)
    date_summary.to_csv(ROOT / "date_stability_summary.csv", index=False)
    confusion_frame.to_csv(ROOT / "confusion_matrices.csv", index=False)
    (ROOT / "training_config.json").write_text(json.dumps({
        "seed": SEED, "classes": CLASSES, "features": FEATURES,
        "model_specs": [asdict(s) for s in specs], "threshold_grid": THRESHOLD_GRID,
        "threshold_selection": "training-only pooled chronological OOF; maximize 3-class macro-F1 subject to >=20% all-row signal coverage and >=30 signals",
        "folds": 3, "initial_training_date_fraction": 0.50,
        "split": {"train_end": TRAIN_END.isoformat(), "validation_start": VALIDATION_START.isoformat()},
        "label": "exact endpoint close at t+H, same NYSE session, strict ±0.10%, NO_SIGNAL otherwise",
        "test_data_accessed": False, "models_trained": True,
    }, indent=2) + "\n", encoding="utf-8")
    (ROOT / "model_metrics.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_report(summary, training_oof_frame, validation_metric_frame, fold_frame, paired_frame, date_summary)
    return summary


def _flat_metrics(m: dict) -> dict:
    return {k: m[k] for k in (
        "n", "accuracy", "balanced_accuracy", "macro_f1", "up_precision", "up_recall",
        "down_precision", "down_recall", "directional_macro_precision", "directional_macro_recall",
        "directional_macro_f1", "signal_coverage_all_eligible", "eligible_signal_count",
        "eligible_signal_accuracy", "directional_accuracy_on_actual_directional",
        "actual_directional_count", "no_signal_prediction_count", "log_loss")}


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (date,)): return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_report(summary, training_metrics, validation_metrics, fold_metrics, paired, date_summary):
    def markdown_table(frame):
        columns = list(frame.columns)
        def render(value):
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{value:.4f}"
            return str(value).replace("|", "\\|").replace("\n", " ")
        lines = ["| " + " | ".join(columns) + " |",
                 "| " + " | ".join("---" for _ in columns) + " |"]
        lines.extend("| " + " | ".join(render(value) for value in row) + " |"
                     for row in frame.itertuples(index=False, name=None))
        return "\n".join(lines)

    lines = [
        "# Independent ECL one-minute horizon model research",
        "",
        "This report compares fixed, modest logistic regression, LightGBM, and XGBoost models for exact-endpoint 10- and 15-minute targets. It uses only generated training/validation datasets through 2026-09-24. The repeatedly used validation period is confirmation, not a pristine holdout. No test data, live API, or paid data were used.",
        "",
        "Signal coverage is directional predictions (UP or DOWN) divided by all eligible labeled rows. Eligible-signal accuracy counts a NO_SIGNAL actual as an error if a directional prediction was emitted. Directional accuracy on actual directional rows is also shown separately. Class order is DOWN, NO_SIGNAL, UP.",
        "",
        "Thresholds are selected once per horizon/model from pooled chronological training OOF predictions only, maximizing three-class macro-F1 on a fixed grid subject to at least 20% signal coverage and 30 signals. Validation results did not influence model or threshold selection.",
        "",
        "## Validation comparison",
        "",
        markdown_table(validation_metrics),
        "",
        "## Training OOF versus validation raw argmax",
        "",
        markdown_table(training_metrics.merge(
            validation_metrics.loc[validation_metrics.policy.eq("raw_argmax"),
                                  ["horizon_min", "model", "accuracy", "balanced_accuracy", "macro_f1",
                                   "directional_macro_precision", "signal_coverage_all_eligible",
                                   "eligible_signal_accuracy"]],
            on=["horizon_min", "model"], how="left", suffixes=("_training_oof", "_validation"),
        )),
        "",
        "Training values are pooled expanding-window OOF predictions only. Each OOF row was predicted by a model fitted on earlier training dates. Validation is the later, already-reused period.",
        "",
        "## Paired validation timestamps",
        "",
        f"Both horizons have {summary['paired_validation_common_timestamp_count']} common eligible validation timestamps.",
        "",
        markdown_table(paired),
        "",
        "## Missing-data exposure",
        "",
        markdown_table(pd.DataFrame([{
            "horizon_min": int(h), "dataset_rows": v["dataset_rows"],
            "feature_ready_rows": v["feature_ready_rows"],
            "warmup_or_missing_feature_rows": v["warmup_or_missing_feature_rows"],
            "unavailable_target_among_feature_ready": v["target_unavailable_among_feature_ready"],
            "missing_endpoint": v["feature_ready_missing_endpoint"],
            "session_or_boundary": v["feature_ready_session_boundary"],
        } for h, v in summary["horizons"].items()])),
        "",
        "## Training OOF folds",
        "",
        markdown_table(fold_metrics),
        "",
        "## Date stability",
        "",
        markdown_table(date_summary),
        "",
        "Per-date detail is saved in `date_stability.csv`. Small per-date samples are noisy and should not be interpreted as independent tests.",
        "",
        "## Limitations",
        "",
        "- Alpaca IEX minute bars are sparse; samples are available only when the exact future endpoint candle exists, and features reset at every missing-minute gap.",
        "- The eligible sample set is a small subset of all source rows after causal feature warm-up and exact-endpoint eligibility.",
        "- Validation has been used in prior research, so this is not independent final confirmation.",
        "- Classification performance is not evidence of trading profitability; no trading simulation is performed here.",
        "- No candidate is promoted by this experiment.",
    ]
    (ROOT / "model_research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        results = run_research()
    print(json.dumps({
        "outputs_dir": str(ROOT), "paired_validation_common_timestamp_count": results["paired_validation_common_timestamp_count"],
        "horizons": {h: {"train_eligible_rows": v["train_eligible_rows"],
                         "validation_eligible_rows": v["validation_eligible_rows"],
                         "models": list(v["models"])} for h, v in results["horizons"].items()},
    }, indent=2))

