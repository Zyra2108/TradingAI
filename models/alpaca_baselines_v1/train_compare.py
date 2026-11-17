"""Fit four fixed Alpaca ECL baseline models using train/validation only.

This file intentionally has no reference to the new test CSV. Final evaluation
is performed by final_evaluate_selected.py after selection.json is written.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/alpaca_ml_pipeline_v1"
OUT = ROOT / "models/alpaca_baselines_v1"
REPORT = ROOT / "reports/alpaca_baselines_v1"
MANIFEST_PATH = DATA / "reproducibility_manifest.json"
CLASS_ORDER = ["DOWN", "NEUTRAL", "UP"]
SEED = 42

CONFIG = {
    "selection_rule": "Validation macro-F1 highest; ties broken by balanced accuracy, directional macro-F1, then lower log loss. No test data used.",
    "random_seed": SEED,
    "class_order": CLASS_ORDER,
    "models": {
        "logistic_regression": {"class_weight": "balanced", "C": 1.0, "max_iter": 2000, "solver": "lbfgs", "preprocessing": "median imputation and StandardScaler fitted within train-only sklearn pipeline"},
        "random_forest": {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1},
        "xgboost": {"n_estimators": 350, "learning_rate": 0.04, "max_depth": 3, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "objective": "multi:softprob", "eval_metric": "mlogloss", "class_imbalance": "compute_sample_weight balanced from train labels only", "n_jobs": -1},
        "hist_gradient_boosting": {"max_iter": 250, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 2.0, "class_imbalance": "compute_sample_weight balanced from train labels only"},
        "rsi_baseline": {"rule": "RSI <= 30 predicts UP; RSI >= 70 predicts DOWN; otherwise NEUTRAL. Fixed, untuned rule."},
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _load_split(name: str, features: list[str]) -> pd.DataFrame:
    path = DATA / f"{name}.csv"
    frame = pd.read_csv(path, parse_dates=["datetime", "future_datetime"])
    expected = ["datetime", "future_datetime", "future_return", "target", *features]
    if list(frame.columns) != expected:
        raise ValueError(f"{name} schema/order mismatch")
    if frame.datetime.duplicated().any() or not frame.datetime.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps must be unique and chronological")
    if not ((frame.future_datetime - frame.datetime) == pd.Timedelta(minutes=15)).all():
        raise ValueError(f"{name} contains a target endpoint with incorrect horizon")
    if not set(frame.target.unique()).issubset(CLASS_ORDER):
        raise ValueError(f"{name} has an unexpected target label")
    values = frame[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} feature matrix has missing or nonfinite values")
    return frame


def _aligned_probabilities(estimator, x: pd.DataFrame) -> np.ndarray:
    raw = estimator.predict_proba(x)
    output = np.zeros((len(x), len(CLASS_ORDER)), dtype=float)
    classes = list(estimator.classes_)
    for col, label in enumerate(classes):
        label = CLASS_ORDER[int(label)] if isinstance(label, (int, np.integer)) else str(label)
        output[:, CLASS_ORDER.index(label)] = raw[:, col]
    if not np.isfinite(output).all() or (output < 0).any() or not np.allclose(output.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Invalid model probabilities")
    # XGBoost's float32 softmax can differ from 1 by a few ulps. Normalize the
    # accepted rows before log-loss calculation and saving their probabilities.
    output /= output.sum(axis=1, keepdims=True)
    return output


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None) -> dict:
    report = classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    directional_precisions = [report[c]["precision"] for c in ["DOWN", "UP"]]
    directional_recalls = [report[c]["recall"] for c in ["DOWN", "UP"]]
    directional_f1s = [report[c]["f1-score"] for c in ["DOWN", "UP"]]
    emitted = np.isin(y_pred, ["DOWN", "UP"])
    actual_directional = np.isin(y_true, ["DOWN", "UP"])
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(report[c]["precision"]), "recall": float(report[c]["recall"]), "f1": float(report[c]["f1-score"]), "support": int(report[c]["support"])} for c in CLASS_ORDER},
        "directional_macro_precision": float(np.mean(directional_precisions)),
        "directional_macro_recall": float(np.mean(directional_recalls)),
        "directional_macro_f1": float(np.mean(directional_f1s)),
        "directional_prediction_coverage_all_rows": float(emitted.mean()),
        "directional_prediction_coverage_actual_directional_rows": float(emitted[actual_directional].mean()) if actual_directional.any() else None,
        "accuracy_among_directional_predictions": float((y_true[emitted] == y_pred[emitted]).mean()) if emitted.any() else None,
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
        "prediction_distribution": {c: int((y_pred == c).sum()) for c in CLASS_ORDER},
        "log_loss": float(log_loss(y_true, probs, labels=CLASS_ORDER)) if probs is not None else None,
    }


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    features = manifest["features"]
    train = _load_split("train", features)
    validation = _load_split("validation", features)
    x_train, y_train = train[features], train.target.to_numpy(str)
    x_val, y_val = validation[features], validation.target.to_numpy(str)
    train_weights = compute_sample_weight("balanced", y_train)
    artifacts: dict[str, object] = {}
    scores: dict[str, dict] = {}
    validation_records = pd.DataFrame({"datetime": validation.datetime, "actual": y_val})

    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)),
        ]),
        "random_forest": RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "xgboost": None,
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=2.0, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is mandatory but unavailable in the project virtual environment") from exc
    models["xgboost"] = XGBClassifier(n_estimators=350, learning_rate=0.04, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, objective="multi:softprob", num_class=3, eval_metric="mlogloss", tree_method="hist", n_jobs=-1, random_state=SEED)

    for name, estimator in models.items():
        started = time.perf_counter()
        if name == "xgboost":
            fit_y = np.asarray([CLASS_ORDER.index(v) for v in y_train], dtype=int)
            estimator.fit(x_train, fit_y, sample_weight=train_weights, verbose=False)
        elif name == "hist_gradient_boosting":
            estimator.fit(x_train, y_train, sample_weight=train_weights)
        else:
            estimator.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        probabilities = _aligned_probabilities(estimator, x_val)
        prediction_seconds = time.perf_counter() - started
        predicted = np.asarray([CLASS_ORDER[int(i)] for i in np.argmax(probabilities, axis=1)], dtype=object)
        metric = evaluate(y_val, predicted, probabilities)
        metric.update({"training_seconds": fit_seconds, "validation_prediction_seconds": prediction_seconds})
        started = time.perf_counter()
        train_probabilities = _aligned_probabilities(estimator, x_train)
        train_prediction_seconds = time.perf_counter() - started
        train_predicted = np.asarray([CLASS_ORDER[int(i)] for i in np.argmax(train_probabilities, axis=1)], dtype=object)
        metric["training_metrics_in_sample"] = evaluate(y_train, train_predicted, train_probabilities)
        metric["training_prediction_seconds"] = train_prediction_seconds
        scores[name] = metric
        artifacts[name] = estimator
        joblib.dump(estimator, OUT / f"{name}.joblib")
        validation_records[f"{name}_prediction"] = predicted
        for j, cls in enumerate(CLASS_ORDER):
            validation_records[f"{name}_p_{cls.lower()}"] = probabilities[:, j]
        pd.DataFrame(metric["confusion_matrix_rows_actual_columns_predicted"], index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(REPORT / f"confusion_{name}.csv", index=True)

    # Untuned fixed reference baselines; neither fits on validation labels.
    always_neutral = np.full(len(y_val), "NEUTRAL", dtype=object)
    scores["always_neutral"] = evaluate(y_val, always_neutral, None)
    rsi = validation.rsi_14.to_numpy(float)
    rsi_pred = np.where(rsi <= 30.0, "UP", np.where(rsi >= 70.0, "DOWN", "NEUTRAL"))
    scores["fixed_rsi_rule"] = evaluate(y_val, rsi_pred, None)
    validation_records["always_neutral_prediction"] = always_neutral
    validation_records["fixed_rsi_rule_prediction"] = rsi_pred
    for baseline in ["always_neutral", "fixed_rsi_rule"]:
        pd.DataFrame(scores[baseline]["confusion_matrix_rows_actual_columns_predicted"], index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(REPORT / f"confusion_{baseline}.csv", index=True)

    ranking = sorted(models, key=lambda name: (-scores[name]["macro_f1"], -scores[name]["balanced_accuracy"], -scores[name]["directional_macro_f1"], scores[name]["log_loss"]))
    selected = ranking[0]
    selection = {"selected_model": selected, "selection_rule": CONFIG["selection_rule"], "validation_ranking": ranking, "test_opened": False, "test_evaluated": False}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    validation_records.to_csv(REPORT / "validation_predictions.csv", index=False)
    (REPORT / "validation_metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    rows = []
    for name, metric in scores.items():
        rows.append({"model": name, **{k: metric.get(k) for k in ["accuracy", "balanced_accuracy", "macro_f1", "directional_macro_precision", "directional_macro_recall", "directional_macro_f1", "directional_prediction_coverage_all_rows", "accuracy_among_directional_predictions", "log_loss", "training_seconds", "validation_prediction_seconds"]}})
    pd.DataFrame(rows).to_csv(REPORT / "validation_comparison.csv", index=False)
    (OUT / "training_config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    schema = {"feature_count": len(features), "feature_order": features, "class_order": CLASS_ORDER, "target_horizon_minutes": 15, "target_thresholds_pct": {"down_lt": -0.10, "up_gt": 0.10}, "prediction_policy": "raw three-class argmax", "volume_as_feature": False}
    (OUT / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    selected_meta = {"selected_model": selected, "artifact": f"models/alpaca_baselines_v1/{selected}.joblib", "artifact_sha256": sha256(OUT / f"{selected}.joblib"), "selection": selection, "feature_schema": schema}
    (OUT / "selected_model_spec.json").write_text(json.dumps(selected_meta, indent=2), encoding="utf-8")
    run_manifest = {
        "source_manifest_sha256": sha256(MANIFEST_PATH),
        "train_csv_sha256": sha256(DATA / "train.csv"),
        "validation_csv_sha256": sha256(DATA / "validation.csv"),
        "test_csv_read": False,
        "model_artifacts": {name: sha256(OUT / f"{name}.joblib") for name in models},
        "feature_order": features,
        "class_order": CLASS_ORDER,
        "config": CONFIG,
        "selected_by_validation_only": selected,
    }
    (OUT / "training_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    report = ["# Alpaca IEX ECL baseline comparison", "", "Validation was used once for model comparison and selection. The validation period has previously appeared in project research, so it is not a pristine holdout.", "The new final test CSV was not opened during training or model selection.", "", f"Selected by the predeclared validation rule: **{selected}**.", "", "| Model | Accuracy | Balanced accuracy | Macro F1 | Directional macro precision | Directional macro F1 | Directional coverage | Signal accuracy | Log loss |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append("| " + " | ".join([str(row["model"])] + ["—" if row.get(k) is None else f"{row[k]:.4f}" for k in ["accuracy", "balanced_accuracy", "macro_f1", "directional_macro_precision", "directional_macro_f1", "directional_prediction_coverage_all_rows", "accuracy_among_directional_predictions", "log_loss"]]) + " |")
    (REPORT / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"selection": selection, "scores": scores, "features": features}


if __name__ == "__main__":
    result = main()
    print(json.dumps({"selection": result["selection"], "scores": result["scores"]}, indent=2))
