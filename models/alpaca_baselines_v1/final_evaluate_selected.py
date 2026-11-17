"""One-time evaluator for the already selected model.

Do not run until validation selection is frozen in selection.json. This is the
only script in the new pipeline that opens data/alpaca_ml_pipeline_v1/test.csv.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/alpaca_ml_pipeline_v1"
MODEL_DIR = ROOT / "models/alpaca_baselines_v1"
REPORT = ROOT / "reports/alpaca_baselines_v1"
CLASS_ORDER = ["DOWN", "NEUTRAL", "UP"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> dict:
    selection_path = MODEL_DIR / "selection.json"
    training_manifest_path = MODEL_DIR / "training_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_evaluated") or selection.get("test_opened"):
        raise RuntimeError("The one-time test evaluation has already been recorded")
    selected = selection["selected_model"]
    artifact = MODEL_DIR / f"{selected}.joblib"
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    expected_hash = training_manifest["model_artifacts"][selected]
    if sha256(artifact) != expected_hash:
        raise RuntimeError("Selected model artifact hash changed after validation selection")
    dataset_manifest = json.loads((DATA / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    features = dataset_manifest["features"]

    # The single intentional read of the new isolated test split occurs here,
    # after the selected candidate and artifact hash have been checked.
    test_path = DATA / "test.csv"
    test = pd.read_csv(test_path, parse_dates=["datetime", "future_datetime"])
    expected_columns = ["datetime", "future_datetime", "future_return", "target", *features]
    if list(test.columns) != expected_columns or test.datetime.duplicated().any() or not test.datetime.is_monotonic_increasing:
        raise ValueError("Final test split schema or chronological order failed integrity checks")
    if not ((test.future_datetime - test.datetime) == pd.Timedelta(minutes=15)).all():
        raise ValueError("Final test split contains incorrect target horizons")
    if not np.isfinite(test[features].to_numpy(float)).all():
        raise ValueError("Final test feature matrix contains invalid values")
    y = test.target.to_numpy(str)
    if not set(y).issubset(CLASS_ORDER):
        raise ValueError("Unexpected final test target class")

    model = joblib.load(artifact)
    started = time.perf_counter()
    raw = model.predict_proba(test[features])
    prediction_seconds = time.perf_counter() - started
    probs = np.zeros((len(test), len(CLASS_ORDER)), dtype=float)
    for j, label in enumerate(list(model.classes_)):
        label = CLASS_ORDER[int(label)] if isinstance(label, (int, np.integer)) else str(label)
        probs[:, CLASS_ORDER.index(label)] = raw[:, j]
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Final test prediction probabilities are invalid")
    preds = np.asarray([CLASS_ORDER[i] for i in np.argmax(probs, axis=1)], dtype=object)

    def metrics(y_true: np.ndarray, y_pred: np.ndarray, p: np.ndarray | None) -> dict:
        report = classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
        directional = [report[c] for c in ["DOWN", "UP"]]
        emitted = np.isin(y_pred, ["DOWN", "UP"])
        actual_directional = np.isin(y_true, ["DOWN", "UP"])
        return {
            "n": len(y_true), "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
            "per_class": {c: {"precision": float(report[c]["precision"]), "recall": float(report[c]["recall"]), "f1": float(report[c]["f1-score"]), "support": int(report[c]["support"])} for c in CLASS_ORDER},
            "directional_macro_precision": float(np.mean([x["precision"] for x in directional])),
            "directional_macro_recall": float(np.mean([x["recall"] for x in directional])),
            "directional_macro_f1": float(np.mean([x["f1-score"] for x in directional])),
            "directional_prediction_coverage_all_rows": float(emitted.mean()),
            "directional_prediction_coverage_actual_directional_rows": float(emitted[actual_directional].mean()),
            "accuracy_among_directional_predictions": float((y_true[emitted] == y_pred[emitted]).mean()) if emitted.any() else None,
            "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
            "prediction_distribution": {c: int((y_pred == c).sum()) for c in CLASS_ORDER},
            "log_loss": float(log_loss(y_true, p, labels=CLASS_ORDER)) if p is not None else None,
        }

    scores = {"selected_model": metrics(y, preds, probs)}
    neutral_pred = np.full(len(y), "NEUTRAL", dtype=object)
    scores["always_neutral_baseline"] = metrics(y, neutral_pred, None)
    train = pd.read_csv(DATA / "train.csv", usecols=["target"])
    majority = str(train.target.value_counts().idxmax())
    majority_pred = np.full(len(y), majority, dtype=object)
    scores["train_majority_baseline"] = metrics(y, majority_pred, None)
    rsi = test.rsi_14.to_numpy(float)
    rsi_pred = np.where(rsi <= 30, "UP", np.where(rsi >= 70, "DOWN", "NEUTRAL"))
    scores["fixed_rsi_baseline"] = metrics(y, rsi_pred, None)

    predictions = pd.DataFrame({"datetime": test.datetime, "future_datetime": test.future_datetime, "actual": y, "prediction": preds})
    for i, label in enumerate(CLASS_ORDER):
        predictions[f"p_{label.lower()}"] = probs[:, i]
    predictions.to_csv(REPORT / "final_test_predictions.csv", index=False)
    for name, values in {"selected_model": preds, "always_neutral_baseline": neutral_pred, "train_majority_baseline": majority_pred, "fixed_rsi_baseline": rsi_pred}.items():
        pd.DataFrame(confusion_matrix(y, values, labels=CLASS_ORDER), index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(REPORT / f"final_test_confusion_{name}.csv")
    payload = {
        "test_evaluation_count": 1,
        "selected_model": selected,
        "selected_model_sha256": expected_hash,
        "test_csv_sha256": sha256(test_path),
        "test_rows": len(test),
        "test_date_range": [str(test.datetime.min().date()), str(test.datetime.max().date())],
        "test_class_distribution": {c: int((y == c).sum()) for c in CLASS_ORDER},
        "prediction_seconds": prediction_seconds,
        "probability_rows_valid": True,
        "scores": scores,
        "test_used_for_selection_or_tuning": False,
        "no_model_changes_after_evaluation": True,
        "limitation": "The chronological dates of this rebuilt test period have appeared in the historical project's prior test research; this is a new Alpaca-only split, not a pristine project-level date holdout.",
    }
    (REPORT / "final_test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for cls in CLASS_ORDER:
        pd.DataFrame(scores["selected_model"]["per_class"][cls], index=[0]).to_csv(REPORT / f"final_test_class_{cls.lower()}.csv", index=False)
    report = ["# One-time Alpaca IEX final test evaluation", "", f"Selected model by validation only: **{selected}**; artifact SHA-256 `{expected_hash}`.", f"Test rows: {len(test):,}; dates {payload['test_date_range'][0]}–{payload['test_date_range'][1]}.", "", "This is a new Alpaca-only chronological test split. The date period has been represented in earlier project test research, so it is not a pristine project-level date holdout. No test results were used for selection or tuning, and no model changes followed this evaluation.", "", "| Evaluation | Accuracy | Balanced accuracy | Macro F1 | Directional precision | Directional recall | Directional F1 | Directional coverage | Signal accuracy | Log loss |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in scores.items():
        report.append("| " + " | ".join([name] + [f"{value[k]:.4f}" if value.get(k) is not None else "—" for k in ["accuracy", "balanced_accuracy", "macro_f1", "directional_macro_precision", "directional_macro_recall", "directional_macro_f1", "directional_prediction_coverage_all_rows", "accuracy_among_directional_predictions", "log_loss"]]) + " |")
    (REPORT / "final_test_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    selection["test_opened"] = True
    selection["test_evaluated"] = True
    selection["test_evaluation_report"] = "reports/alpaca_baselines_v1/final_test_report.md"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
