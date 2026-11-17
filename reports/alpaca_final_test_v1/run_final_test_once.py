"""One-time held-out test evaluation of the frozen Alpaca candidate.

This script deliberately performs exactly one test CSV read and one predict_proba
call. It never fits a model and refuses to overwrite any existing report output.
Do not rerun after the one-time evaluation has completed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "models/alpaca_final_freeze_v1"
TEST_CSV = ROOT / "data/alpaca_rebuild_v1/test.csv"
OUT = ROOT / "reports/alpaca_final_test_v1"
CLASSES = ["DOWN", "NEUTRAL", "UP"]
FEATURES = json.loads((FREEZE / "feature_columns.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(y, pred, probs):
    per = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=CLASSES)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1_3class": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "directional_precision_macro": float(precision_score(y, pred, labels=["DOWN", "UP"], average="macro", zero_division=0)),
        "directional_recall_macro": float(recall_score(y, pred, labels=["DOWN", "UP"], average="macro", zero_division=0)),
        "directional_f1_macro": float(f1_score(y, pred, labels=["DOWN", "UP"], average="macro", zero_division=0)),
        "directional_coverage": float(np.isin(pred, ["DOWN", "UP"]).mean()),
        "log_loss": float(log_loss(y, probs, labels=CLASSES)),
        "per_class": {c: {"precision": float(per[c]["precision"]), "recall": float(per[c]["recall"]), "f1": float(per[c]["f1-score"]), "support": int(per[c]["support"])} for c in CLASSES},
        "confusion_matrix_rows_actual_columns_predicted": cm.tolist(),
        "predicted_class_counts": {c: int((pred == c).sum()) for c in CLASSES},
    }


def main():
    existing = [p.name for p in OUT.iterdir() if p.name not in {Path(__file__).name, "__pycache__"}] if OUT.exists() else []
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing evaluation report artifacts: {existing}")
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads((FREEZE / "frozen_candidate_spec.json").read_text(encoding="utf-8"))
    expected_hashes = json.loads((FREEZE / "SHA256SUMS.json").read_text(encoding="utf-8"))
    artifact_names = ["candidate_pipeline.joblib", "lightgbm_estimator.joblib", "preprocessing_pipeline.joblib"]
    before_hashes = {name: sha256(FREEZE / name) for name in artifact_names}
    if any(before_hashes[n] != expected_hashes[n] for n in artifact_names):
        raise AssertionError("Frozen artifact hash does not match freeze SHA256SUMS.json")
    if before_hashes["candidate_pipeline.joblib"] != spec["source_candidate_sha256"]:
        raise AssertionError("Candidate pipeline hash differs from frozen specification")

    model = joblib.load(FREEZE / "candidate_pipeline.joblib")
    if list(model.feature_names_in_) != FEATURES or [str(c) for c in model.classes_] != CLASSES:
        raise AssertionError("Frozen model feature order/class order mismatch")

    # Sole test dataset read in this script/run.
    test = pd.read_csv(TEST_CSV)
    if "close.1" in test.columns:
        if not np.array_equal(test.close.to_numpy(), test["close.1"].to_numpy()):
            raise AssertionError("Repeated close values differ in test data")
        test = test.drop(columns=["close.1"])
    test["datetime"] = pd.to_datetime(test.datetime, errors="raise")
    test["future_datetime"] = pd.to_datetime(test.future_datetime, errors="raise")
    for c in FEATURES:
        test[c] = pd.to_numeric(test[c], errors="raise")
    test["day_of_week"] = test.day_of_week.astype("int64")
    if list(test.loc[:, FEATURES].columns) != FEATURES:
        raise AssertionError("Test feature order mismatch")
    if test.datetime.duplicated().any() or not test.datetime.is_monotonic_increasing:
        raise AssertionError("Test timestamps are duplicate or out of order")
    if not (test.future_datetime == test.datetime + pd.Timedelta(minutes=15)).all():
        raise AssertionError("Unexpected test label horizon")
    if test.target.isna().any() or not set(test.target.astype(str)).issubset(CLASSES):
        raise AssertionError("Invalid test labels")
    X = test[FEATURES]
    if X.isna().any().any() or not np.isfinite(X.to_numpy(float)).all():
        raise AssertionError("Test features contain missing or nonfinite values")

    # Exactly one model inference pass. Raw frozen argmax policy; no thresholds.
    probs = model.predict_proba(X)
    if probs.shape != (len(test), 3) or not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("Invalid model probabilities")
    pred = np.asarray(CLASSES)[np.argmax(probs, axis=1)]
    y = test.target.astype(str).to_numpy()
    primary = metrics(y, pred, probs)

    # Train-period class counts were stored in the rebuild manifest. Use that
    # known majority class, never the held-out labels, for the baseline choice.
    manifest = json.loads((ROOT / "data/alpaca_rebuild_v1/reproducibility_manifest.json").read_text(encoding="utf-8-sig"))
    train_counts = {c: int(manifest["split_summaries"]["train"]["class_distribution"][c]["count"]) for c in CLASSES}
    majority_class = max(CLASSES, key=lambda c: train_counts[c])
    dummy_pred = np.full(len(y), majority_class, dtype=object)
    dummy_probs = np.zeros((len(y), 3), dtype=float)
    dummy_probs[:, CLASSES.index(majority_class)] = 1.0
    baseline = metrics(y, dummy_pred, dummy_probs)

    val_metrics = spec["validation_metrics"]
    comparison = {k: {"validation": float(val_metrics[k]), "test": float(primary[k]), "test_minus_validation": float(primary[k] - val_metrics[k])} for k in val_metrics}
    after_hashes = {name: sha256(FREEZE / name) for name in artifact_names}
    frozen_unchanged = before_hashes == after_hashes and all(after_hashes[n] == expected_hashes[n] for n in artifact_names)
    if not frozen_unchanged:
        raise AssertionError("Frozen model artifacts changed during test evaluation")

    class_counts = {c: int((y == c).sum()) for c in CLASSES}
    output_metrics = {
        "evaluation": "one-time held-out test evaluation",
        "candidate": spec["candidate"], "asset": spec["asset"],
        "test_rows": int(len(test)), "test_date_range": [str(test.datetime.min().date()), str(test.datetime.max().date())],
        "target_class_order": CLASSES, "test_actual_class_counts": class_counts,
        "frozen_candidate_metrics": primary,
        "majority_baseline": {"training_majority_class": majority_class, "training_class_counts": train_counts, "test_metrics": baseline},
        "validation_comparison": comparison,
        "prediction_integrity": {"prediction_count": int(len(pred)), "probability_shape": list(probs.shape), "all_predictions_in_class_order": bool(set(pred).issubset(CLASSES)), "all_probabilities_finite_nonnegative": bool(np.isfinite(probs).all() and (probs >= 0).all()), "probability_sum_max_abs_error": float(np.max(np.abs(probs.sum(axis=1) - 1.0))), "class_probability_columns": ["p_down", "p_neutral", "p_up"], "prediction_policy": spec["prediction_policy"]},
        "frozen_hashes_before": before_hashes, "frozen_hashes_after": after_hashes, "frozen_artifacts_unchanged": frozen_unchanged,
        "test_evaluation_count": 1, "tuning_after_test": False, "model_changed_after_test": False,
        "interpretation_limit": "Classification results are not trading profitability; no backtest or live performance was measured.",
    }

    pd.DataFrame(primary["confusion_matrix_rows_actual_columns_predicted"], index=CLASSES, columns=CLASSES).rename_axis("actual\\predicted").to_csv(OUT / "confusion_matrix.csv")
    prediction_output = pd.DataFrame({
        "datetime": test.datetime.astype(str), "future_datetime": test.future_datetime.astype(str),
        "actual": y, "prediction": pred,
        "p_down": probs[:, 0], "p_neutral": probs[:, 1], "p_up": probs[:, 2],
    })
    prediction_output.to_csv(OUT / "prediction_summary.csv", index=False)
    (OUT / "final_test_metrics.json").write_text(json.dumps(output_metrics, indent=2), encoding="utf-8")

    lines = [
        "# Alpaca ECL final held-out test evaluation", "",
        "**One-time evaluation:** this report records the first and only evaluation of the held-out Alpaca test split. No tuning, feature changes, threshold changes, retraining, or model modifications followed the results.", "",
        f"Candidate: `{spec['candidate']}`. Rows: {len(test):,}. Dates: {test.datetime.min().date()} through {test.datetime.max().date()}.",
        f"Actual class counts (DOWN / NEUTRAL / UP): {class_counts['DOWN']:,} / {class_counts['NEUTRAL']:,} / {class_counts['UP']:,}.", "",
        "## Classification metrics", "",
        "| Metric | Frozen candidate | Training-majority dummy | Validation candidate | Test minus validation |", "|---|---:|---:|---:|---:|",
    ]
    fields = [
        ("Accuracy", "accuracy"), ("Balanced accuracy", "balanced_accuracy"), ("3-class macro F1", "macro_f1_3class"),
        ("Directional precision, macro UP/DOWN", "directional_precision_macro"), ("Directional recall, macro UP/DOWN", "directional_recall_macro"),
        ("Directional F1, macro UP/DOWN", "directional_f1_macro"), ("Directional prediction coverage", "directional_coverage"), ("Log loss", "log_loss"),
    ]
    for label, key in fields:
        lines.append(f"| {label} | {primary[key]:.4f} | {baseline[key]:.4f} | {val_metrics[key]:.4f} | {primary[key] - val_metrics[key]:+.4f} |")
    lines += ["", "## Per-class report", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for c in CLASSES:
        pc = primary["per_class"][c]
        lines.append(f"| {c} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']:,} |")
    lines += ["", "## Confusion matrix", "", "Rows are actual classes; columns are predicted classes (DOWN, NEUTRAL, UP).", "", "| Actual \\ Predicted | DOWN | NEUTRAL | UP |", "|---|---:|---:|---:|"]
    cm = primary["confusion_matrix_rows_actual_columns_predicted"]
    for i, c in enumerate(CLASSES): lines.append(f"| {c} | {cm[i][0]:,} | {cm[i][1]:,} | {cm[i][2]:,} |")
    lines += ["", "## Prediction checks and frozen hashes", "", f"Predictions: {len(pred):,}; UP {primary['predicted_class_counts']['UP']:,}, DOWN {primary['predicted_class_counts']['DOWN']:,}, NEUTRAL {primary['predicted_class_counts']['NEUTRAL']:,}.", f"All predictions belong to the configured class order: {output_metrics['prediction_integrity']['all_predictions_in_class_order']}. All probabilities are finite and nonnegative: {output_metrics['prediction_integrity']['all_probabilities_finite_nonnegative']}. Maximum row-sum error: {output_metrics['prediction_integrity']['probability_sum_max_abs_error']:.3g}.", "", "SHA-256 before and after evaluation (unchanged):", ""]
    for name in artifact_names: lines.append(f"- `{name}`: `{before_hashes[name]}`")
    lines += ["", "## Interpretation and limitations", "", f"The dummy baseline predicts `{majority_class}` for every test row, with its majority class determined from the training split distribution, not from test labels. The test period has only {test.datetime.dt.date.nunique()} trading dates, so results are a limited historical holdout measurement.", "", "Classification accuracy and class metrics do not measure trading profitability. This evaluation includes no backtest, transaction costs, paper/live trading, or production-readiness assessment. The test set was not used to select or alter the model.", "", "**Disclosure:** Before the evaluation, an earlier filesystem path-existence check queried metadata for `test.csv`; it did not read test contents. This report is the only test-content read and only model evaluation. No model tuning or changes followed it.", ""]
    (OUT / "final_test_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"test_rows": len(test), "date_range": [str(test.datetime.min().date()), str(test.datetime.max().date()),], "actual_counts": class_counts, "primary_metrics": primary, "majority_baseline": {"class": majority_class, **baseline}, "validation_comparison": comparison, "frozen_hashes_unchanged": frozen_unchanged, "output_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
