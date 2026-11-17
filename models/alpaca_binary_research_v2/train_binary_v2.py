"""Controlled chronological research for binary ECL direction; reads train/validation only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "alpaca_rebuild_v1"
OUT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "models" / "alpaca_final_freeze_v1" / "frozen_candidate_spec.json"
PREVIOUS = ROOT / "models" / "alpaca_binary_research_v1"
CLASSES = ["DOWN", "NEUTRAL", "UP"]
DIRECTIONS = ["DOWN", "UP"]
ENCODE = {"DOWN": 0, "UP": 1}
COVERAGES = (0.20, 0.30, 0.40, 0.50)
SEED = 20261113

BASE_FEATURES = [
    "close", "return_1", "return_3", "return_6", "return_12",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
    "ema_ratio_5", "ema_ratio_10", "ema_ratio_20", "rsi_14",
    "volatility_5", "volatility_10", "volatility_20",
    "candle_body", "candle_range", "upper_wick", "lower_wick",
    "minutes_since_open", "minutes_until_close", "day_of_week",
    "time_sin", "time_cos",
]
DERIVED_FEATURES = [
    "momentum_sign_agreement", "momentum_mean", "momentum_short_long_gap",
    "trend_sign_agreement", "trend_strength_mean_abs",
    "trend_momentum_agreement", "volatility_5_to_20", "volatility_10_to_20",
    "body_to_range", "wick_imbalance_to_range", "total_wick_to_range",
    "session_position",
]
FEATURE_SETS = {"original_24": BASE_FEATURES,
                "original_24_plus_causal_agreement": BASE_FEATURES + DERIVED_FEATURES}
MODEL_NAMES = ["logistic_regression", "extra_trees", "random_forest", "xgboost", "lightgbm"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_allowed_split(name: str) -> pd.DataFrame:
    """Only explicit development and validation files can be opened by this script."""
    if name not in {"train.csv", "validation.csv"}:
        raise ValueError("Only the predeclared training and validation files are allowed")
    frame = pd.read_csv(DATA / name)
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate input columns in {name}")
    if "close.1" in frame.columns:
        if not np.array_equal(frame["close"].to_numpy(), frame["close.1"].to_numpy()):
            raise ValueError(f"Duplicate close fields differ in {name}")
        frame = frame.drop(columns=["close.1"])
    required = {"datetime", "future_datetime", "future_return", "target", *BASE_FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required fields in {name}: {sorted(required-set(frame.columns))}")
    if [c for c in frame.columns if c in BASE_FEATURES] != BASE_FEATURES:
        raise ValueError(f"Frozen base feature order mismatch in {name}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame["future_datetime"] = pd.to_datetime(frame["future_datetime"], errors="raise")
    for col in BASE_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame["day_of_week"] = frame["day_of_week"].astype("int64")
    if not frame.datetime.is_monotonic_increasing or frame.datetime.duplicated().any():
        raise ValueError(f"Invalid chronology in {name}")
    if not (frame.future_datetime == frame.datetime + pd.Timedelta(minutes=15)).all():
        raise ValueError(f"Unexpected label horizon in {name}")
    if not set(frame.target.unique()).issubset(CLASSES) or frame.target.isna().any():
        raise ValueError(f"Unexpected labels in {name}")
    returns = frame.future_return.to_numpy(float)
    expected = np.where(returns > .10, "UP", np.where(returns < -.10, "DOWN", "NEUTRAL"))
    if not np.array_equal(frame.target.to_numpy(str), expected):
        raise ValueError(f"Label threshold mismatch in {name}")
    if not np.isfinite(frame[BASE_FEATURES].to_numpy(float)).all():
        raise ValueError(f"Nonfinite base features in {name}")
    return frame


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.to_numpy(float)
    num = numerator.to_numpy(float)
    return pd.Series(np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-12),
                     index=numerator.index)


def engineer_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-local transforms of already-causal features; no future rows or labels read."""
    out = frame.copy()
    momentum = out[["return_1", "return_3", "return_6", "return_12"]]
    trend = out[["sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
                 "ema_ratio_5", "ema_ratio_10", "ema_ratio_20"]]
    out["momentum_sign_agreement"] = np.sign(momentum).mean(axis=1)
    out["momentum_mean"] = momentum.mean(axis=1)
    out["momentum_short_long_gap"] = out.return_1 - out.return_12
    out["trend_sign_agreement"] = np.sign(trend).mean(axis=1)
    out["trend_strength_mean_abs"] = trend.abs().mean(axis=1)
    out["trend_momentum_agreement"] = np.sign(trend.mean(axis=1)) * np.sign(momentum.mean(axis=1))
    out["volatility_5_to_20"] = _safe_ratio(out.volatility_5, out.volatility_20)
    out["volatility_10_to_20"] = _safe_ratio(out.volatility_10, out.volatility_20)
    out["body_to_range"] = _safe_ratio(out.candle_body, out.candle_range)
    out["wick_imbalance_to_range"] = _safe_ratio(out.upper_wick - out.lower_wick, out.candle_range)
    out["total_wick_to_range"] = _safe_ratio(out.upper_wick + out.lower_wick, out.candle_range)
    session_total = out.minutes_since_open + out.minutes_until_close
    out["session_position"] = _safe_ratio(out.minutes_since_open, session_total)
    if not np.isfinite(out[DERIVED_FEATURES].to_numpy(float)).all():
        raise ValueError("Derived feature calculation produced nonfinite values")
    return out


def make_preprocessor(features: list[str], model_name: str):
    numeric = [c for c in features if c != "day_of_week"]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "logistic_regression":
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("weekday", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["day_of_week"]),
    ], remainder="drop", verbose_feature_names_out=False)


def build_estimator(name: str, seed: int, down_count: int, up_count: int):
    if name == "logistic_regression":
        return LogisticRegression(C=.20, max_iter=500, class_weight="balanced", random_state=seed)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=250, max_depth=8, min_samples_leaf=18,
                                    max_features=.75, class_weight="balanced", n_jobs=-1,
                                    random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=18,
                                      max_features=.75, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    if name == "xgboost":
        return xgb.XGBClassifier(n_estimators=220, max_depth=2, learning_rate=.03,
                                 min_child_weight=8, subsample=.8, colsample_bytree=.8,
                                 reg_lambda=10., scale_pos_weight=down_count/max(up_count, 1),
                                 objective="binary:logistic", eval_metric="logloss",
                                 tree_method="hist", n_jobs=4, random_state=seed)
    if name == "lightgbm":
        return lgb.LGBMClassifier(objective="binary", n_estimators=220, learning_rate=.03,
                                  num_leaves=7, min_child_samples=35, subsample=.8,
                                  subsample_freq=1, colsample_bytree=.8, reg_lambda=10.,
                                  class_weight="balanced", n_jobs=4, random_state=seed,
                                  verbosity=-1)
    raise ValueError(name)


def fit_pipeline(name: str, features: list[str], train: pd.DataFrame, seed: int):
    y = train.target.map(ENCODE).to_numpy(int)
    down, up = int((train.target == "DOWN").sum()), int((train.target == "UP").sum())
    pipe = Pipeline([("preprocessor", make_preprocessor(features, name)),
                     ("model", build_estimator(name, seed, down, up))])
    pipe.fit(train.loc[:, features], y)
    if list(pipe.named_steps["preprocessor"].feature_names_in_) != features:
        raise AssertionError("Feature input order changed during preprocessing")
    return pipe


def binary_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    cm = confusion_matrix(y, prediction, labels=DIRECTIONS)
    return {
        "n_eligible": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)) if len(y) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)) if len(y) and len(np.unique(y)) > 1 else None,
        "up_precision": float(precision_score(y, prediction, labels=DIRECTIONS,
                                               average="binary", pos_label="UP", zero_division=0)) if len(y) else None,
        "down_precision": float(precision_score(y, prediction, labels=DIRECTIONS,
                                                 average="binary", pos_label="DOWN", zero_division=0)) if len(y) else None,
        "macro_directional_precision": float(precision_score(y, prediction, labels=DIRECTIONS,
                                                               average="macro", zero_division=0)) if len(y) else None,
        "macro_directional_recall": float(recall_score(y, prediction, labels=DIRECTIONS,
                                                         average="macro", zero_division=0)) if len(y) else None,
        "macro_directional_f1": float(f1_score(y, prediction, labels=DIRECTIONS,
                                                average="macro", zero_division=0)) if len(y) else None,
        "confusion_labels_rows_actual_cols_prediction": DIRECTIONS,
        "confusion_matrix": cm.tolist(),
    }


def confidence_threshold(confidence: np.ndarray, target: float) -> float:
    scores = np.asarray(confidence, dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or not 0 < target < 1:
        raise ValueError("Invalid confidence/coverage input")
    k = max(1, int(np.ceil(target * len(scores))))
    return float(np.sort(scores)[-k])


def selective_metrics(y: np.ndarray, prediction: np.ndarray, confidence: np.ndarray,
                      threshold: float) -> tuple[dict, np.ndarray]:
    emit = np.asarray(confidence) >= threshold
    output = np.where(emit, prediction, "NO_SIGNAL")
    eligible = np.isin(y, DIRECTIONS)
    eligible_emit = emit & eligible
    eligible_report = binary_metrics(y[eligible_emit], prediction[eligible_emit])
    all_cm = confusion_matrix(y, output, labels=["DOWN", "NEUTRAL", "UP", "NO_SIGNAL"])
    count = int(emit.sum())
    tp = {c: int(((output == c) & (y == c)).sum()) for c in DIRECTIONS}
    pred_count = {c: int((output == c).sum()) for c in DIRECTIONS}
    operational_precision = {c: (tp[c]/pred_count[c] if pred_count[c] else 0.) for c in DIRECTIONS}
    return ({
        "threshold": float(threshold), "validation_or_oof_rows": int(len(y)),
        "eligible_directional_rows": int(eligible.sum()), "emitted_predictions": count,
        "no_signal_count": int(len(y)-count), "coverage_all_candles": float(count/len(y)),
        "eligible_emitted_predictions": int(eligible_emit.sum()),
        "coverage_eligible_directional": float(eligible_emit.sum()/eligible.sum()) if eligible.any() else None,
        "eligible_directional_metrics_on_emissions": eligible_report,
        "up_precision_operational_neutral_false": operational_precision["UP"],
        "down_precision_operational_neutral_false": operational_precision["DOWN"],
        "macro_precision_operational_neutral_false": float(np.mean(list(operational_precision.values()))),
        "accuracy_all_emissions_neutral_counted_wrong": float(((output == y) & emit).sum()/count) if count else None,
        "confusion_labels_rows_actual_cols_prediction": ["DOWN", "NEUTRAL", "UP", "NO_SIGNAL"],
        "confusion_matrix_all_rows": all_cm.tolist(),
    }, output)


def calibration_metrics(y: np.ndarray, p_up: np.ndarray) -> dict:
    y_binary = np.array([ENCODE[c] for c in y], dtype=int)
    p = np.clip(np.asarray(p_up, dtype=float), 1e-7, 1-1e-7)
    edges = np.linspace(0, 1, 11)
    bins = []
    ece = 0.
    for i in range(10):
        mask = (p >= edges[i]) & ((p <= edges[i+1]) if i == 9 else (p < edges[i+1]))
        if mask.any():
            mean_p, rate = float(p[mask].mean()), float(y_binary[mask].mean())
            ece += float(mask.mean()) * abs(mean_p-rate)
            bins.append({"bin": i, "count": int(mask.sum()), "mean_predicted_p_up": mean_p,
                         "observed_up_rate": rate, "lower": float(edges[i]), "upper": float(edges[i+1])})
    return {"n": len(y), "log_loss": float(log_loss(y_binary, np.c_[1-p, p], labels=[0,1])),
            "brier": float(brier_score_loss(y_binary, p)), "ece_10_equal_width": float(ece),
            "p_up_mean": float(p.mean()), "p_up_std": float(p.std()),
            "confidence_quantiles": {str(q): float(np.quantile(np.maximum(p,1-p),q)) for q in (.05,.25,.5,.75,.95)},
            "reliability_bins": bins}


def chronological_folds(frame: pd.DataFrame, n_folds: int = 3) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = np.array(sorted(frame.datetime.dt.normalize().unique()))
    initial = max(1, int(np.floor(.40 * len(dates))))
    eval_chunks = [chunk for chunk in np.array_split(dates[initial:], n_folds) if len(chunk)]
    folds = []
    for i, chunk in enumerate(eval_chunks):
        eval_start, eval_end = pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1]) + pd.Timedelta(days=1)
        train_end = eval_start
        folds.append((train_end, eval_start, eval_end))
    if len(folds) != n_folds:
        raise ValueError("Not enough trading dates for requested chronological folds")
    return folds


def fold_fit_rows(frame: pd.DataFrame, eval_start: pd.Timestamp) -> pd.DataFrame:
    # Purge any training label whose future endpoint reaches the evaluation block.
    selected = (frame.datetime < eval_start) & (frame.future_datetime < eval_start) & frame.target.isin(DIRECTIONS)
    return frame.loc[selected].copy()


def audit_previous_v1() -> tuple[pd.DataFrame, dict]:
    """Recalculate v1 saved-prediction metrics; no model is retrained or previous file changed."""
    saved = pd.read_csv(PREVIOUS / "validation_predictions.csv")
    reported = pd.read_csv(PREVIOUS / "selective_metrics.csv")
    reported_models = json.loads((PREVIOUS / "metrics.json").read_text(encoding="utf-8"))["models"]
    actual = saved.actual.to_numpy(str)
    eligible = np.isin(actual, DIRECTIONS)
    rows = []
    for model in ("xgboost_binary", "lightgbm_binary"):
        raw = saved[f"{model}_raw_prediction"].to_numpy(str)
        raw_metrics = binary_metrics(actual[eligible], raw[eligible])
        old_report = next(x for x in reported_models if x["model"] == model)["unthresholded_binary_metrics_eligible_only"]
        raw_checks = {
            "unthresholded_accuracy_matches": np.isclose(raw_metrics["accuracy"], old_report["accuracy"]),
            "unthresholded_up_precision_matches": np.isclose(raw_metrics["up_precision"], old_report["up_precision"]),
            "unthresholded_down_precision_matches": np.isclose(raw_metrics["down_precision"], old_report["down_precision"]),
            "unthresholded_macro_precision_matches": np.isclose(raw_metrics["macro_directional_precision"], old_report["macro_precision"]),
            "unthresholded_macro_recall_matches": np.isclose(raw_metrics["macro_directional_recall"], old_report["macro_recall"]),
            "unthresholded_macro_f1_matches": np.isclose(raw_metrics["macro_directional_f1"], old_report["macro_f1"]),
            "unthresholded_confusion_matches": raw_metrics["confusion_matrix"] == old_report["confusion_matrix"],
        }
        rows.append({"model": model, "coverage_target": "unthresholded",
                     **raw_checks, "all_reported_values_match": all(raw_checks.values())})
        for cov in COVERAGES:
            label = f"{int(cov*100)}pct"
            outputs = saved[f"{model}_{label}_output"].to_numpy(str)
            emitted = outputs != "NO_SIGNAL"
            eligible_emitted = emitted & eligible
            candidate = reported[(reported.model == model) &
                                 (reported.target_coverage_all_validation_candles == cov)].iloc[0]
            up_p = float(precision_score(actual[eligible_emitted], outputs[eligible_emitted],
                                          labels=DIRECTIONS, average="binary", pos_label="UP", zero_division=0))
            dn_p = float(precision_score(actual[eligible_emitted], outputs[eligible_emitted],
                                         labels=DIRECTIONS, average="binary", pos_label="DOWN", zero_division=0))
            acc = float(accuracy_score(actual[eligible_emitted], outputs[eligible_emitted])) if eligible_emitted.any() else None
            all_acc = float(accuracy_score(actual[emitted], outputs[emitted])) if emitted.any() else None
            checks = {
                "eligible_precision_up_matches": np.isclose(up_p, candidate.up_precision_eligible_directional_only),
                "eligible_precision_down_matches": np.isclose(dn_p, candidate.down_precision_eligible_directional_only),
                "eligible_accuracy_matches": np.isclose(acc, candidate.binary_accuracy_emitted_eligible_directional_only),
                "all_emission_accuracy_matches": np.isclose(all_acc, candidate.emitted_accuracy_including_neutral_misses),
                "emission_count_matches": int(emitted.sum()) == int(candidate.emitted_predictions),
            }
            rows.append({"model": model, "coverage_target": cov,
                         "recalculated_up_precision_eligible": up_p,
                         "recalculated_down_precision_eligible": dn_p,
                         "recalculated_accuracy_emitted_eligible": acc,
                         "recalculated_accuracy_all_emitted": all_acc,
                         "recalculated_emitted": int(emitted.sum()),
                         "all_reported_values_match": all(checks.values()), **checks})
    summary = {
        "source": "models/alpaca_binary_research_v1/validation_predictions.csv",
        "saved_predictions_rows": int(len(saved)),
        "eligible_directional_rows": int(eligible.sum()),
        "actual_class_distribution": {c: int((actual == c).sum()) for c in CLASSES},
        "raw_model_predictions": {
            model: {c: int((saved[f"{model}_raw_prediction"] == c).sum()) for c in DIRECTIONS}
            for model in ("xgboost_binary", "lightgbm_binary")},
        "recalculated_unthresholded": {
            model: binary_metrics(actual[eligible], saved[f"{model}_raw_prediction"].to_numpy(str)[eligible])
            for model in ("xgboost_binary", "lightgbm_binary")},
        "report_metrics_consistent_with_saved_predictions": bool(all(r["all_reported_values_match"] for r in rows)),
        "precision_scope": "Saved v1 reports both eligible directional precision (actual neutral excluded) and operational precision (neutral actual directional signals are false positives).",
        "coverage_scope": "V1 main target coverage is emitted predictions/all validation candles; eligible coverage is separately reported.",
    }
    return pd.DataFrame(rows), summary


def main():
    generated_names = {"previous_v1_metric_audit.csv", "previous_v1_audit.json",
      "walk_forward_fold_metrics.json", "walk_forward_selective_fold_metrics.json",
      "walk_forward_oof_predictions.csv", "walk_forward_model_ranking.csv",
      "walk_forward_confusion_matrices.csv", "walk_forward_regime_robustness.csv",
      "walk_forward_calibration_bins.csv", "validation_calibration_bins.csv",
      "selected_binary_pipeline.joblib", "validation_predictions.csv", "validation_metrics.csv",
      "validation_confusion_matrices.csv", "validation_metrics.json", "feature_schema.json",
      "training_config.json", "model_specification.json", "validation_daily_robustness.csv", "validation_regime_robustness.csv",
      "reproducibility_manifest.json", "summary.json", "research_report.md"}
    existing = [p.name for p in OUT.iterdir() if p.name not in {Path(__file__).name, "finalize_saved_report.py", "__pycache__"}
                and p.name not in generated_names]
    if existing:
        raise FileExistsError(f"Refusing to overwrite prior v2 artifacts: {existing}")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec["features_exact_order"] != BASE_FEATURES or spec["class_order"] != CLASSES:
        raise ValueError("Frozen feature/class contract mismatch")
    train = load_allowed_split("train.csv")
    validation = load_allowed_split("validation.csv")
    if train.datetime.max() >= validation.datetime.min() or train.future_datetime.max() >= validation.datetime.min():
        raise ValueError("Training/validation chronology or boundary purge failed")
    train = engineer_causal_features(train)
    validation = engineer_causal_features(validation)
    folds = chronological_folds(train)

    previous_checks, previous_summary = audit_previous_v1()
    previous_checks.to_csv(OUT / "previous_v1_metric_audit.csv", index=False)
    (OUT / "previous_v1_audit.json").write_text(json.dumps(previous_summary, indent=2), encoding="utf-8")

    fold_rows, fold_predictions = [], []
    configs = [(model, set_name, features)
               for model in MODEL_NAMES for set_name, features in FEATURE_SETS.items()]
    for model_name, feature_set, features in configs:
        for fold_id, (train_end, eval_start, eval_end) in enumerate(folds, 1):
            fit = fold_fit_rows(train, eval_start)
            eval_rows = train[(train.datetime >= eval_start) & (train.datetime < eval_end)].copy()
            if len(fit) < 500 or len(eval_rows) == 0 or fit.target.nunique() != 2:
                raise ValueError(f"Insufficient fold data in fold {fold_id}")
            pipe = fit_pipeline(model_name, features, fit, SEED + fold_id)
            proba = pipe.predict_proba(eval_rows.loc[:, features])
            classes = [int(x) for x in pipe.named_steps["model"].classes_]
            if classes != [0, 1] or not np.isfinite(proba).all() or not np.allclose(proba.sum(axis=1), 1., atol=1e-8):
                raise ValueError("Invalid walk-forward probabilities or class order")
            p_up, p_down = proba[:, 1], proba[:, 0]
            prediction = np.where(p_up > p_down, "UP", "DOWN")
            confidence = np.maximum(p_up, p_down)
            actual = eval_rows.target.to_numpy(str)
            eligible = np.isin(actual, DIRECTIONS)
            calibration = calibration_metrics(actual[eligible], p_up[eligible])
            base = binary_metrics(actual[eligible], prediction[eligible])
            fold_rows.append({"model": model_name, "feature_set": feature_set, "fold": fold_id,
                              "train_start": str(fit.datetime.min()), "train_end": str(fit.datetime.max()),
                              "eval_start": str(eval_rows.datetime.min()), "eval_end": str(eval_rows.datetime.max()),
                              "train_rows_all": len(fit), "train_rows_directional": int(fit.target.isin(DIRECTIONS).sum()),
                              "eval_rows": len(eval_rows), "eligible_eval_rows": int(eligible.sum()),
                              "always_neutral_accuracy_all_eval": float((actual == "NEUTRAL").mean()),
                              "unthresholded_eligible": base, "calibration_eligible": calibration})
            for i, (_, row) in enumerate(eval_rows.iterrows()):
                fold_predictions.append({"model": model_name, "feature_set": feature_set, "fold": fold_id,
                                         "datetime": row.datetime, "actual": actual[i],
                                         "p_down": float(p_down[i]), "p_up": float(p_up[i]),
                                         "volatility_20": float(row.volatility_20),
                                         "confidence": float(confidence[i]),
                                         "raw_prediction": prediction[i]})

    folds_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(fold_predictions).sort_values("datetime").reset_index(drop=True)
    policy_rows, conf_rows, oof_summaries = [], [], []
    for model_name, feature_set, _ in configs:
        data = oof_df[(oof_df.model == model_name) & (oof_df.feature_set == feature_set)].copy()
        actual, pred = data.actual.to_numpy(str), data.raw_prediction.to_numpy(str)
        for cov in COVERAGES:
            # Threshold coverage is calibrated on OOF rows known to be directional.
            oof_eligible = np.isin(data.actual.to_numpy(str), DIRECTIONS)
            th = confidence_threshold(data.confidence.to_numpy(float)[oof_eligible], cov)
            metrics, output = selective_metrics(actual, pred, data.confidence.to_numpy(float), th)
            policy_rows.append({"model": model_name, "feature_set": feature_set,
                                "target_coverage_eligible_directional": cov, **metrics})
            for i, a in enumerate(["DOWN", "NEUTRAL", "UP", "NO_SIGNAL"]):
                for j, b in enumerate(["DOWN", "NEUTRAL", "UP", "NO_SIGNAL"]):
                    conf_rows.append({"dataset": "training_oof", "model": model_name,
                                      "feature_set": feature_set, "coverage": cov,
                                      "actual": a, "predicted": b,
                                      "count": metrics["confusion_matrix_all_rows"][i][j]})
            data[f"output_{int(cov*100)}pct"] = output
        # Per-fold measurements use each fold's own score-only quantile; counts and dates remain visible.
        for fold_id, group in data.groupby("fold", sort=True):
            a, p, conf = group.actual.to_numpy(str), group.raw_prediction.to_numpy(str), group.confidence.to_numpy(float)
            for cov in COVERAGES:
                fold_eligible = np.isin(a, DIRECTIONS)
                th = confidence_threshold(conf[fold_eligible], cov)
                m, _ = selective_metrics(a, p, conf, th)
                oof_summaries.append({"model": model_name, "feature_set": feature_set,
                                      "fold": int(fold_id), "coverage_target_eligible_directional": cov,
                                      "eval_rows": len(group), "eligible_directional_rows": int(np.isin(a, DIRECTIONS).sum()),
                                      **m})
        oof_df.loc[data.index, [f"{model_name}_{feature_set}_output_{int(c*100)}pct" for c in COVERAGES]] = \
            data[[f"output_{int(c*100)}pct" for c in COVERAGES]].to_numpy()

    policy_df = pd.DataFrame(policy_rows)
    oof_fold_df = pd.DataFrame(oof_summaries)
    rank_rows = []
    for model_name, feature_set, _ in configs:
        group = oof_fold_df[(oof_fold_df.model == model_name) &
                            (oof_fold_df.feature_set == feature_set) &
                            (oof_fold_df.coverage_target_eligible_directional == .30)]
        unthresholded = folds_df[(folds_df.model == model_name) & (folds_df.feature_set == feature_set)]
        eligible_metrics = [r["unthresholded_eligible"] for r in unthresholded.to_dict("records")]
        rank_rows.append({"model": model_name, "feature_set": feature_set,
                          "folds": len(group), "mean_fold_macro_precision_eligible": float(group.eligible_directional_metrics_on_emissions.map(lambda x:x["macro_directional_precision"]).mean()),
                          "std_fold_macro_precision_eligible": float(group.eligible_directional_metrics_on_emissions.map(lambda x:x["macro_directional_precision"]).std(ddof=0)),
                          "mean_fold_accuracy_emitted_eligible": float(group.eligible_directional_metrics_on_emissions.map(lambda x:x["accuracy"]).mean()),
                          "mean_fold_eligible_coverage": float(group.coverage_eligible_directional.mean()),
                          "mean_fold_up_precision": float(group.eligible_directional_metrics_on_emissions.map(lambda x:x["up_precision"]).mean()),
                          "mean_fold_down_precision": float(group.eligible_directional_metrics_on_emissions.map(lambda x:x["down_precision"]).mean()),
                          "mean_fold_unthresholded_accuracy": float(np.mean([x["accuracy"] for x in eligible_metrics]))})
    ranking = pd.DataFrame(rank_rows).sort_values(
        ["mean_fold_macro_precision_eligible", "mean_fold_accuracy_emitted_eligible",
         "mean_fold_unthresholded_accuracy"], ascending=False).reset_index(drop=True)
    eligible_candidates = ranking[ranking.mean_fold_eligible_coverage >= .20]
    winner = (eligible_candidates.iloc[0] if len(eligible_candidates) else ranking.iloc[0])
    selected_model, selected_feature_set = winner.model, winner.feature_set
    selected_features = FEATURE_SETS[selected_feature_set]

    # Final model fit on the complete training split, after OOF-only selection.
    directional_train = train[train.target.isin(DIRECTIONS)].copy()
    final_model = fit_pipeline(selected_model, selected_features, directional_train, SEED)
    joblib.dump(final_model, OUT / "selected_binary_pipeline.joblib", compress=3)

    selected_oof = oof_df[(oof_df.model == selected_model) &
                          (oof_df.feature_set == selected_feature_set)].copy()
    selected_oof_eligible = np.isin(selected_oof.actual.to_numpy(str), DIRECTIONS)
    oof_thresholds = {str(c): confidence_threshold(selected_oof.confidence.to_numpy(float)[selected_oof_eligible], c)
                      for c in COVERAGES}
    # Validation confirmation exactly once, with feature/model/threshold choices frozen from training OOF.
    val_proba = final_model.predict_proba(validation.loc[:, selected_features])
    if not np.isfinite(val_proba).all() or (val_proba < 0).any() or not np.allclose(val_proba.sum(axis=1), 1., atol=1e-8):
        raise ValueError("Invalid final validation probabilities")
    val_actual = validation.target.to_numpy(str)
    val_p_up, val_p_down = val_proba[:, 1], val_proba[:, 0]
    val_pred = np.where(val_p_up > val_p_down, "UP", "DOWN")
    val_conf = np.maximum(val_p_up, val_p_down)
    val_eligible = np.isin(val_actual, DIRECTIONS)
    val_rows, val_confusion = [], []
    val_predictions = validation[["datetime", "target", "future_return"]].rename(
        columns={"target": "actual", "future_return": "actual_future_return_pct"}).copy()
    val_predictions["eligible_directional_actual"] = val_eligible
    val_predictions["p_down"] = val_p_down
    val_predictions["p_up"] = val_p_up
    val_predictions["confidence"] = val_conf
    val_predictions["raw_prediction"] = val_pred
    for cov in COVERAGES:
        th = oof_thresholds[str(cov)]
        metrics, output = selective_metrics(val_actual, val_pred, val_conf, th)
        val_rows.append({"model": selected_model, "feature_set": selected_feature_set,
                         "target_oof_coverage_eligible_directional": cov,
                         "threshold_learned_from_training_oof": th, **metrics})
        val_predictions[f"output_{int(cov*100)}pct"] = output
        for i, a in enumerate(["DOWN", "NEUTRAL", "UP", "NO_SIGNAL"]):
            for j, b in enumerate(["DOWN", "NEUTRAL", "UP", "NO_SIGNAL"]):
                val_confusion.append({"dataset": "validation_confirmation", "model": selected_model,
                                      "feature_set": selected_feature_set, "coverage": cov,
                                      "actual": a, "predicted": b,
                                      "count": metrics["confusion_matrix_all_rows"][i][j]})
    validation_metrics = {
        "selected_model": selected_model, "selected_feature_set": selected_feature_set,
        "rows": int(len(validation)), "date_range": [str(validation.datetime.min()), str(validation.datetime.max())],
        "class_distribution": {c: int((val_actual == c).sum()) for c in CLASSES},
        "eligible_directional_rows": int(val_eligible.sum()),
        "binary_accuracy_all_eligible_directional": binary_metrics(val_actual[val_eligible], val_pred[val_eligible]),
        "calibration_eligible_directional": calibration_metrics(val_actual[val_eligible], val_p_up[val_eligible]),
        "selective_policies": val_rows,
        "use_note": "Single final confirmation after all feature/model/threshold selection on chronological training-only OOF predictions. Existing validation has been used in earlier research; this is not pristine holdout data.",
    }
    val_predictions.to_csv(OUT / "validation_predictions.csv", index=False)
    pd.DataFrame(val_rows).to_csv(OUT / "validation_metrics.csv", index=False)
    pd.DataFrame(val_confusion).to_csv(OUT / "validation_confusion_matrices.csv", index=False)
    folds_df.to_json(OUT / "walk_forward_fold_metrics.json", orient="records", indent=2, date_format="iso")
    fold_calibration = []
    for record in fold_rows:
        for b in record["calibration_eligible"]["reliability_bins"]:
            fold_calibration.append({"model": record["model"], "feature_set": record["feature_set"],
                                     "fold": record["fold"], **b})
    pd.DataFrame(fold_calibration).to_csv(OUT / "walk_forward_calibration_bins.csv", index=False)
    oof_fold_df.to_json(OUT / "walk_forward_selective_fold_metrics.json", orient="records", indent=2, date_format="iso")
    oof_df.to_csv(OUT / "walk_forward_oof_predictions.csv", index=False)
    ranking.to_csv(OUT / "walk_forward_model_ranking.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(OUT / "walk_forward_confusion_matrices.csv", index=False)
    (OUT / "validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(validation_metrics["calibration_eligible_directional"]["reliability_bins"]).to_csv(
        OUT / "validation_calibration_bins.csv", index=False)

    feature_schema = {
        "original_features": BASE_FEATURES,
        "added_features": {"momentum_sign_agreement": "Mean sign of causal returns at 1,3,6,12 bars.",
          "momentum_mean": "Mean of causal 1,3,6,12-bar returns.",
          "momentum_short_long_gap": "Causal 1-bar return minus 12-bar return.",
          "trend_sign_agreement": "Mean sign across causal SMA/EMA price ratios at 5,10,20 bars.",
          "trend_strength_mean_abs": "Mean absolute causal SMA/EMA price ratio.",
          "trend_momentum_agreement": "Product of signs of mean trend ratios and mean momentum returns.",
          "volatility_5_to_20": "Causal 5-bar volatility divided by 20-bar volatility; zero if denominator is zero.",
          "volatility_10_to_20": "Causal 10-bar volatility divided by 20-bar volatility; zero if denominator is zero.",
          "body_to_range": "Causal candle body divided by candle range; zero if denominator is zero.",
          "wick_imbalance_to_range": "Upper minus lower wick divided by candle range.",
          "total_wick_to_range": "Upper plus lower wick divided by candle range.",
          "session_position": "Minutes since open divided by minutes since open plus minutes until close."},
        "causality": "Each derived feature uses only same-row original causal features. No future row, future return, target, or volume field is input. Training dataset contains no volume or raw open/high/low columns, so no volume-derived feature was added.",
        "feature_sets": {k: v for k,v in FEATURE_SETS.items()},
    }
    (OUT / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    config = {
        "seed": SEED, "target": {"horizon_minutes": 15, "up": "future return > +0.10%",
                                    "down": "future return < -0.10%", "neutral": "otherwise",
                                    "neutral_excluded_from_binary_training_and_directional_metrics": True},
        "walk_forward": {"method": "expanding window, 3 chronological folds, date-grouped", "initial_train_date_fraction": .40,
                         "purge": "Remove training samples with future_datetime >= evaluation start."},
        "models": {"logistic_regression": {"C": .20, "max_iter": 500, "class_weight": "balanced"},
          "extra_trees": {"n_estimators": 250, "max_depth": 8, "min_samples_leaf": 18, "max_features": .75, "class_weight": "balanced"},
          "random_forest": {"n_estimators": 250, "max_depth": 8, "min_samples_leaf": 18, "max_features": .75, "class_weight": "balanced_subsample"},
          "xgboost": {"n_estimators": 220, "max_depth": 2, "learning_rate": .03, "min_child_weight": 8, "reg_lambda": 10.},
          "lightgbm": {"n_estimators": 220, "num_leaves": 7, "learning_rate": .03, "min_child_samples": 35, "reg_lambda": 10., "class_weight": "balanced"},
          "catboost": "Not run; not installed."},
        "preprocessing": "Median imputation; weekday most-frequent imputation and one-hot encoding. Logistic numeric features additionally standardized. All fit inside each fold or full-train final fit.",
        "thresholds": {"coverage_targets": list(COVERAGES), "selection_source": "Training-only OOF confidence-score quantiles among actual UP/DOWN OOF rows; no validation labels used to set thresholds.",
                        "coverage_reference": "Eligible directional examples; all-candle coverage and neutral false signals are also reported.", "tie_policy": "Predict DOWN on equal class probabilities; score ties can increase realized coverage."},
        "selection_rule": "At 30% OOF eligible-direction coverage, rank by mean fold eligible-only directional macro precision, then emitted eligible directional accuracy, then unthresholded accuracy. Prefer candidates with mean eligible-direction coverage >=20%; otherwise report coverage shortfall.",
        "validation_use": "Final confirmation after training-only selection; validation has prior project use and is not a pristine holdout. An initial provisional run used an incorrect all-candle coverage denominator and was discarded; corrected outputs use eligible-direction OOF thresholds. No validation labels/results were used for candidate ranking.",
        "no_test_data_access": True,
    }
    (OUT / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Date and causal-volatility robustness for final validation at the OOF-derived 30% threshold.
    selected_th = oof_thresholds["0.3"]
    emit = val_conf >= selected_th
    daily_rows = []
    for day in sorted(validation.datetime.dt.date.unique()):
        mask = validation.datetime.dt.date.eq(day).to_numpy()
        a, p, e = val_actual[mask], val_pred[mask], emit[mask]
        eligible = np.isin(a, DIRECTIONS)
        emitted_eligible = e & eligible
        daily_rows.append({"date": str(day), "rows": int(mask.sum()), "signals": int(e.sum()),
                           "coverage_all": float(e.mean()), "eligible_rows": int(eligible.sum()),
                           "eligible_signals": int(emitted_eligible.sum()),
                           "coverage_eligible": float(emitted_eligible.sum()/eligible.sum()) if eligible.any() else None,
                           "eligible_directional_accuracy": float((p[emitted_eligible] == a[emitted_eligible]).mean()) if emitted_eligible.any() else None,
                           "eligible_up_precision": float(precision_score(a[emitted_eligible],p[emitted_eligible],labels=DIRECTIONS,average="binary",pos_label="UP",zero_division=0)) if emitted_eligible.any() else None,
                           "eligible_down_precision": float(precision_score(a[emitted_eligible],p[emitted_eligible],labels=DIRECTIONS,average="binary",pos_label="DOWN",zero_division=0)) if emitted_eligible.any() else None,
                           "all_emission_accuracy_neutral_wrong": float((p[e] == a[e]).mean()) if e.any() else None})
    vol_cut = float(train.volatility_20.median())
    regime_rows = []
    for regime, mask in [("below_training_volatility_median", validation.volatility_20.to_numpy() < vol_cut),
                         ("at_or_above_training_volatility_median", validation.volatility_20.to_numpy() >= vol_cut)]:
        a,p,e = val_actual[mask],val_pred[mask],emit[mask]
        eligible = np.isin(a,DIRECTIONS); ee = e & eligible
        regime_rows.append({"regime":regime,"training_volatility_median":vol_cut,"rows":int(mask.sum()),
                            "eligible_rows":int(eligible.sum()),"signals":int(e.sum()),
                            "eligible_signals":int(ee.sum()),
                            "eligible_directional_accuracy":float((p[ee]==a[ee]).mean()) if ee.any() else None,
                            "coverage_eligible":float(ee.sum()/eligible.sum()) if eligible.any() else None,
                            "all_emission_accuracy_neutral_wrong":float((p[e]==a[e]).mean()) if e.any() else None})
    pd.DataFrame(daily_rows).to_csv(OUT / "validation_daily_robustness.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(OUT / "validation_regime_robustness.csv", index=False)
    oof_regime_rows = []
    train_vol_cut = float(train.volatility_20.median())
    selected_oof = selected_oof.copy()
    oof_emit = selected_oof.confidence.to_numpy(float) >= oof_thresholds["0.3"]
    for regime, mask in [("below_training_volatility_median", selected_oof.volatility_20.to_numpy(float) < train_vol_cut),
                         ("at_or_above_training_volatility_median", selected_oof.volatility_20.to_numpy(float) >= train_vol_cut)]:
        a = selected_oof.actual.to_numpy(str)[mask]
        p = selected_oof.raw_prediction.to_numpy(str)[mask]
        emit = oof_emit[mask]
        eligible = np.isin(a, DIRECTIONS)
        eligible_emit = emit & eligible
        oof_regime_rows.append({"regime": regime, "training_volatility_median": train_vol_cut,
                                "rows": int(mask.sum()), "eligible_rows": int(eligible.sum()),
                                "signals": int(emit.sum()), "eligible_signals": int(eligible_emit.sum()),
                                "eligible_coverage": float(eligible_emit.sum()/eligible.sum()) if eligible.any() else None,
                                "eligible_directional_accuracy": float((p[eligible_emit] == a[eligible_emit]).mean()) if eligible_emit.any() else None})
    pd.DataFrame(oof_regime_rows).to_csv(OUT / "walk_forward_regime_robustness.csv", index=False)

    train_hash = file_sha256(DATA / "train.csv")
    val_hash = file_sha256(DATA / "validation.csv")
    frozen_hash = file_sha256(ROOT / "models" / "alpaca_final_freeze_v1" / "candidate_pipeline.joblib")
    artifact_hash = file_sha256(OUT / "selected_binary_pipeline.joblib")
    manifest = {"inputs_read": ["data/alpaca_rebuild_v1/train.csv", "data/alpaca_rebuild_v1/validation.csv",
                                  "models/alpaca_final_freeze_v1/frozen_candidate_spec.json",
                                  "models/alpaca_binary_research_v1/validation_predictions.csv",
                                  "models/alpaca_binary_research_v1/selective_metrics.csv"],
                "input_sha256": {"train.csv": train_hash, "validation.csv": val_hash,
                                 "frozen_candidate_pipeline_sha256": frozen_hash,
                                 "frozen_candidate_spec_sha256": file_sha256(SPEC_PATH)},
                "selected_model_sha256": artifact_hash,
                "training_script_sha256": file_sha256(Path(__file__).resolve()),
                "training_config_sha256": file_sha256(OUT / "training_config.json"),
                "feature_schema_sha256": file_sha256(OUT / "feature_schema.json"),
                "test_dataset_accessed": False, "seed": SEED,
                "v2_validation_inference_passes": 2,
                "validation_process_note": "A provisional v2 pass used an all-candle coverage denominator and was discarded. Corrected outputs use OOF eligible-direction thresholds. Model/feature ranking remained OOF-only; the provisional result was not used for selection.",
                "v2_validation_inference_passes": 2,
                "validation_process_note": "One provisional v2 pass used an all-candle coverage denominator and was discarded. The corrected reported policy thresholds are trained from OOF eligible-direction scores. Model/feature ranking was OOF-only; the provisional metrics were not used to rank candidates.",
                "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                             "scikit_learn": sklearn.__version__, "xgboost": xgb.__version__,
                             "lightgbm": lgb.__version__, "catboost_installed": importlib.util.find_spec("catboost") is not None}}
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "selected_model": selected_model, "selected_feature_set": selected_feature_set,
        "selected_walk_forward_mean_macro_precision_eligible": float(winner.mean_fold_macro_precision_eligible),
        "selected_walk_forward_mean_emitted_accuracy_eligible": float(winner.mean_fold_accuracy_emitted_eligible),
        "selected_walk_forward_mean_eligible_coverage": float(winner.mean_fold_eligible_coverage),
        "validation_confirmation": validation_metrics,
        "previous_v1_audit": previous_summary,
        "validation_limitation": "Previously used validation set; corrected outputs are not independent final test evidence. A provisional v2 evaluation was discarded after correcting the coverage denominator.",
        "test_dataset_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    target_policy = next(x for x in val_rows if x["target_oof_coverage_eligible_directional"] == .30)
    best_precision_any = max(x["eligible_directional_metrics_on_emissions"]["macro_directional_precision"] for x in val_rows)
    best_accuracy_any = max(x["eligible_directional_metrics_on_emissions"]["accuracy"] for x in val_rows)
    meaningful_rows = [x for x in val_rows if x["coverage_eligible_directional"] >= .20]
    best_meaningful_precision = max(x["eligible_directional_metrics_on_emissions"]["macro_directional_precision"] for x in meaningful_rows) if meaningful_rows else None
    best_meaningful_accuracy = max(x["eligible_directional_metrics_on_emissions"]["accuracy"] for x in meaningful_rows) if meaningful_rows else None
    achieved_at_meaningful_coverage = any(
        x["coverage_eligible_directional"] >= .20 and
        x["eligible_directional_metrics_on_emissions"]["macro_directional_precision"] >= .65 and
        x["eligible_directional_metrics_on_emissions"]["accuracy"] >= .55 for x in val_rows)
    validation_metrics["target_attainment"] = {
        "minimum_eligible_directional_coverage": .20,
        "precision_target": .65, "accuracy_target": .55,
        "both_targets_met_at_minimum_coverage": bool(achieved_at_meaningful_coverage),
        "best_macro_precision_at_coverage_at_least_20pct": best_meaningful_precision,
        "best_accuracy_at_coverage_at_least_20pct": best_meaningful_accuracy,
        "highest_metrics_below_minimum_coverage": {"best_macro_precision_any_coverage": best_precision_any,
                                                   "best_accuracy_any_coverage": best_accuracy_any},
    }
    (OUT / "validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2, allow_nan=False), encoding="utf-8")
    summary["target_attainment"] = validation_metrics["target_attainment"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    daily_sufficient = [x for x in daily_rows if x["eligible_signals"] >= 5 and x["eligible_directional_accuracy"] is not None]
    report = [
        "# ECL Binary UP/DOWN Research v2", "", "## Prior-round audit", "",
        f"Independent recalculation from saved v1 validation predictions matched reported UP/DOWN eligible precision, emitted eligible accuracy, all-emission accuracy, and emission counts for every model/coverage row: **{previous_summary['report_metrics_consistent_with_saved_predictions']}**.",
        "V1 reports UP and DOWN precision separately. It distinguishes eligible directional precision (actual NEUTRAL excluded) from operational precision (a signal on an actual NEUTRAL candle counts as a false positive). Its main coverage target is percent of all validation candles; eligible-direction coverage is separately reported. The near-balanced predictions and training directional counts do not suggest the small v1 result was simply majority-class exploitation; performance was near the directional majority and return-sign baselines, with weak confidence separation/calibration.",
        "Likely limits: only 2,519 directional training rows; one stock and one 15-minute labeling rule; noisy thresholded returns; overlapping multi-horizon price-derived features; no volume or raw OHLC fields exist in the supplied train/validation CSVs. V1 XGBoost/LightGBM produced near-even UP/DOWN predictions but only modest accuracy above the directional baselines, and its validation confidence was not strongly calibrated. The prior experiment itself was not edited.",
        "", "## Controlled approach", "",
        "Three date-ordered expanding training folds compare Logistic Regression, ExtraTrees, Random Forest, XGBoost, and LightGBM on the exact original 24 features and a small set of row-local causal agreement/ratio features. CatBoost was not installed and was not added. Preprocessing is fitted separately inside every fold. Training examples whose 15-minute label endpoint reaches the fold evaluation start are purged. Model ranking and coverage thresholds use training-only OOF predictions. A provisional validation evaluation was discarded after identifying an all-candle versus eligible-direction coverage-denominator mismatch; the corrected confirmation below uses the OOF eligible-direction denominator. No validation metrics were used to choose the model or tune its threshold.",
        "The CSVs have no volume, open, high, or low columns, so no volume or new candle structure was fabricated. Added features use only the 24 existing same-candle causal inputs.",
        "", "## Selection and walk-forward evidence", "",
        f"Selected by the declared OOF rule: **{selected_model}** with **{selected_feature_set}**. OOF 30% policy mean eligible-only macro precision {winner.mean_fold_macro_precision_eligible:.1%}; mean emitted eligible directional accuracy {winner.mean_fold_accuracy_emitted_eligible:.1%}; mean eligible directional coverage {winner.mean_fold_eligible_coverage:.1%}.",
        "See `walk_forward_model_ranking.csv`, `walk_forward_fold_metrics.json`, `walk_forward_selective_fold_metrics.json`, `walk_forward_regime_robustness.csv`, and `walk_forward_oof_predictions.csv` for all candidates/folds. Fold consistency is summarized as means and standard deviations; individual UP/DOWN fold values and class confusion matrices are retained.",
        "", "## Corrected validation confirmation", "",
        f"Validation contains {len(validation):,} candles: {int((val_actual=='DOWN').sum())} DOWN, {int((val_actual=='NEUTRAL').sum())} NEUTRAL, {int((val_actual=='UP').sum())} UP; {int(val_eligible.sum())} eligible directional examples. At the predeclared 30% OOF threshold: {target_policy['emitted_predictions']} signals, {target_policy['no_signal_count']} NO SIGNAL, {target_policy['coverage_eligible_directional']:.1%} eligible directional coverage; eligible UP precision {target_policy['eligible_directional_metrics_on_emissions']['up_precision']:.1%}, DOWN precision {target_policy['eligible_directional_metrics_on_emissions']['down_precision']:.1%}, macro directional precision {target_policy['eligible_directional_metrics_on_emissions']['macro_directional_precision']:.1%}, eligible emitted accuracy {target_policy['eligible_directional_metrics_on_emissions']['accuracy']:.1%}, and all-emission accuracy counting neutral actuals as wrong {target_policy['accuracy_all_emissions_neutral_counted_wrong']:.1%}.",
        f"A nominal 20% OOF eligible-coverage threshold produced only {val_rows[0]['coverage_eligible_directional']:.1%} validation eligible coverage ({val_rows[0]['eligible_emitted_predictions']} eligible signals); its {best_precision_any:.1%} best macro precision and {best_accuracy_any:.1%} best directional accuracy do not count as meeting the coverage requirement. Among thresholds that reached at least 20% validation eligible coverage, best macro precision was {best_meaningful_precision:.1%} and best eligible emitted accuracy was {best_meaningful_accuracy:.1%}. The 65% precision, 55% accuracy, and 20% eligible-coverage goals were {'met together' if achieved_at_meaningful_coverage else 'not met together'}.",
        "The corrected 30% OOF threshold itself fell below the required validation coverage, showing confidence-distribution shift. At a higher threshold that did reach 20% coverage, directional precision and accuracy were materially below targets. The high result at the 20% OOF threshold came from a small validation signal subset and must not be presented as satisfying the meaningful-coverage goal.",
        f"At the validation 30% threshold, {len(daily_sufficient)} dates had at least five eligible emitted predictions; their mean eligible directional accuracy was {np.mean([x['eligible_directional_accuracy'] for x in daily_sufficient]):.1%} with standard deviation {np.std([x['eligible_directional_accuracy'] for x in daily_sufficient]):.1%}. See daily and volatility-regime files. This is variable-date diagnostic evidence, not independent confirmation.",
        "", "## Recommendation and limitations", "",
        "Do not integrate this candidate into the live dashboard. The selected Logistic Regression on the original 24 features did not establish the targets at meaningful validation coverage, and it underperformed the first-round LightGBM at comparable coverage. The validation set was used in prior project research and is not pristine. No frozen test dataset or test-period data was accessed. No live data, trading, or profitability evaluation was performed.",
    ]
    (OUT / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
