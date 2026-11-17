"""Audit and freeze the Alpaca balanced-LightGBM candidate if checks pass.

This script reads only train.csv and validation.csv. It contains no test-split
reader, test-path access, model fitting, live API calls, or trading logic.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    f1_score, log_loss, precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[2]
FEATURES = [
    "close", "return_1", "return_3", "return_6", "return_12",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20", "ema_ratio_5",
    "ema_ratio_10", "ema_ratio_20", "rsi_14", "volatility_5",
    "volatility_10", "volatility_20", "candle_body", "candle_range",
    "upper_wick", "lower_wick", "minutes_since_open", "minutes_until_close",
    "day_of_week", "time_sin", "time_cos",
]
CLASSES = ["DOWN", "NEUTRAL", "UP"]
DATA = ROOT / "data/alpaca_rebuild_v1"
CANDIDATES = ROOT / "models/alpaca_candidates_v1"
OUT = ROOT / "models/alpaca_final_freeze_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_allowed_split(name: str) -> pd.DataFrame:
    if name not in {"train", "validation"}:
        raise ValueError("Only train and validation are allowed")
    frame = pd.read_csv(DATA / f"{name}.csv")
    # The dataset builder has a repeated metadata/model close header. Verify
    # identity, then use the first close value as the explicitly named feature.
    if "close.1" in frame:
        if not np.array_equal(frame.close.to_numpy(), frame["close.1"].to_numpy()):
            raise AssertionError(f"close columns differ in {name}")
        frame = frame.drop(columns=["close.1"])
    frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
    frame["future_datetime"] = pd.to_datetime(frame.future_datetime, errors="raise")
    for col in FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame["day_of_week"] = frame.day_of_week.astype("int64")
    return frame


def main() -> None:
    cfg_path = CANDIDATES / "training_config.json"
    metrics_path = CANDIDATES / "validation_metrics.csv"
    pred_path = CANDIDATES / "validation_predictions.csv"
    source_model = CANDIDATES / "lightgbm_balanced_B_start.joblib"
    train_path, valid_path = DATA / "train.csv", DATA / "validation.csv"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    train, valid = read_allowed_split("train"), read_allowed_split("validation")
    checks: dict[str, dict] = {}

    def check(name, ok, evidence):
        checks[name] = {"status": "PASS" if bool(ok) else "FAIL", "evidence": str(evidence)}

    check("Only authorized splits used", cfg.get("data_inputs") == ["data/alpaca_rebuild_v1/train.csv", "data/alpaca_rebuild_v1/validation.csv"] and cfg.get("test_data_accessed") is False,
          "Training manifest names train.csv and validation.csv only; this audit has no test-split reader.")
    check("Training and validation hashes match configuration", sha256(train_path) == cfg["train_sha256"] and sha256(valid_path) == cfg["validation_sha256"], "The two allowed split hashes match training_config.json.")
    check("Chronology and horizon purge", train.datetime.max() < valid.datetime.min() and train.future_datetime.max() < valid.datetime.min(), "Every training decision and target endpoint precedes validation.")
    check("Exact 24-feature order", cfg.get("feature_columns_exact_order") == FEATURES and cfg.get("feature_count") == 24 and list(train[FEATURES].columns) == FEATURES and list(valid[FEATURES].columns) == FEATURES, "Both split inputs and configuration match frozen B feature order.")
    check("No target/time metadata as model inputs", not (set(FEATURES) & {"datetime", "future_datetime", "future_return", "target", "timestamp"}), "The model receives only the explicit feature allowlist.")
    check("Feature types", train.day_of_week.dtype == np.dtype("int64") and all(pd.api.types.is_numeric_dtype(train[c]) for c in FEATURES) and all(pd.api.types.is_numeric_dtype(valid[c]) for c in FEATURES), "day_of_week is integer/categorical; all features are numeric; validation matches.")

    # A separate process loads the complete saved pipeline and infers validation.
    model = joblib.load(source_model)
    check("Candidate loads in fresh process", True, "The audit is executed as a standalone Python process; joblib loaded the saved candidate.")
    check("Pipeline accepts 24 ordered inputs", list(model.feature_names_in_) == FEATURES and model.n_features_in_ == 24 and list(model.named_steps["preprocessor"].feature_names_in_) == FEATURES,
          "Pipeline and preprocessor feature_names_in_ equal the expected ordered feature list.")
    check("Preprocessing and estimator are saved", set(model.named_steps) == {"preprocessor", "model"}, "The artifact is a fitted sklearn Pipeline containing ColumnTransformer and LightGBM.")
    check("Class order", [str(x) for x in model.classes_] == CLASSES, f"Saved estimator classes: {[str(x) for x in model.classes_]}.")

    prep = model.named_steps["preprocessor"]
    numeric = [c for c in FEATURES if c != "day_of_week"]
    medians = prep.named_transformers_["numeric"].named_steps["imputer"].statistics_
    mode = prep.named_transformers_["day_of_week"].named_steps["imputer"].statistics_[0]
    onehot_categories = prep.named_transformers_["day_of_week"].named_steps["onehot"].categories_[0]
    prep_ok = (np.allclose(medians, train[numeric].median().to_numpy(float), rtol=1e-10, atol=1e-10)
               and int(mode) == int(train.day_of_week.mode().iloc[0])
               and np.array_equal(np.asarray(onehot_categories, dtype=int), np.sort(train.day_of_week.unique().astype(int))))
    check("Preprocessor fitted from training only", prep_ok, f"Imputer medians, weekday mode ({int(mode)}), and encoder categories match train.csv; validation values were not passed to fit.")

    p = model.predict_proba(valid[FEATURES])
    pred = np.asarray(CLASSES)[p.argmax(axis=1)]
    prob_ok = p.shape == (len(valid), 3) and np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1.0, atol=1e-10)
    check("Fresh validation probabilities", prob_ok, f"{len(valid)} rows; finite nonnegative probabilities; sums within 1e-10 of 1.")

    saved = pd.read_csv(pred_path)
    saved = saved.loc[saved.model == "lightgbm_balanced_B_start"].reset_index(drop=True)
    saved["datetime"] = pd.to_datetime(saved.datetime, errors="raise")
    same_rows = (len(saved) == len(valid) and saved.datetime.equals(valid.datetime.reset_index(drop=True))
                 and np.array_equal(saved.actual.astype(str).to_numpy(), valid.target.astype(str).to_numpy()))
    prob_columns = ["p_down", "p_neutral", "p_up"]
    same_predictions = same_rows and np.allclose(saved[prob_columns].to_numpy(float), p, rtol=1e-10, atol=1e-10) and np.array_equal(saved.prediction.astype(str).to_numpy(), pred)
    check("Saved validation outputs reproduce", same_predictions, "Fresh-process probabilities/classes/timestamps/actual labels match saved validation_predictions.csv.")

    y = valid.target.astype(str).to_numpy()
    computed = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1_3class": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
        "directional_precision_macro": precision_score(y, pred, labels=["DOWN", "UP"], average="macro", zero_division=0),
        "directional_recall_macro": recall_score(y, pred, labels=["DOWN", "UP"], average="macro", zero_division=0),
        "directional_f1_macro": f1_score(y, pred, labels=["DOWN", "UP"], average="macro", zero_division=0),
        "directional_coverage": np.isin(pred, ["DOWN", "UP"]).mean(),
        "log_loss": log_loss(y, p, labels=CLASSES),
    }
    metric_table = pd.read_csv(metrics_path)
    reported = metric_table.loc[metric_table.model == "lightgbm_balanced_B_start"].iloc[0]
    metric_ok = all(abs(float(reported[k]) - float(v)) <= 1e-12 for k, v in computed.items())
    check("Artifact reproduces reported metrics", metric_ok, f"Fresh validation recalculation matches all eight metrics to 1e-12: { {k: round(float(v), 8) for k, v in computed.items()} }")

    params = cfg["parameters"]["lightgbm_balanced"]
    estimator = model.named_steps["model"]
    keys = ["n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples", "subsample", "subsample_freq", "colsample_bytree", "reg_lambda", "class_weight", "random_state"]
    param_ok = all(estimator.get_params().get(k) == params[k] for k in keys)
    check("Model parameters match config", param_ok, "Saved LightGBM estimator parameters match training_config.json.")

    training_source = (CANDIDATES / "train_compare.py").read_text(encoding="utf-8")
    flow_ok = ('train = load_data("train.csv")' in training_source and 'valid = load_data("validation.csv")' in training_source
               and "model.fit(X_train, y_train)" in training_source and 'load_data("test.csv")' not in training_source)
    check("Training code fits on training split only", flow_ok, "Candidate script loads train/validation separately, fits on X_train/y_train, and contains no test loader.")

    failures = [name for name, item in checks.items() if item["status"] != "PASS"]
    print(json.dumps({"checks": checks, "failures": failures}, indent=2))
    if failures:
        raise SystemExit("AUDIT FAILED: no candidate freeze was created")
    if OUT.exists():
        raise SystemExit(f"Freeze output already exists; refusing overwrite: {OUT}")

    OUT.mkdir(parents=True)
    shutil.copy2(source_model, OUT / "candidate_pipeline.joblib")
    joblib.dump(estimator, OUT / "lightgbm_estimator.joblib", compress=3)
    joblib.dump(prep, OUT / "preprocessing_pipeline.joblib", compress=3)
    (OUT / "feature_columns.json").write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")
    (OUT / "class_order.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    frozen = {
        "freeze_name": "alpaca_final_freeze_v1", "candidate": "lightgbm_balanced_B_start",
        "status": "research candidate freeze; not the official live model",
        "asset": "ECL", "features_exact_order": FEATURES, "feature_count": 24,
        "class_order": CLASSES, "prediction_policy": "raw three-class argmax",
        "target": {"horizon_minutes": 15, "up": "future return > +0.10%", "down": "future return < -0.10%", "neutral": "otherwise"},
        "preprocessing": "Median imputation on numeric features, weekday most-frequent imputation and one-hot encoding; fitted only on train.csv.",
        "lightgbm_parameters": params, "train_rows": len(train), "validation_rows": len(valid),
        "train_date_range": [str(train.datetime.min().date()), str(train.datetime.max().date())],
        "validation_date_range": [str(valid.datetime.min().date()), str(valid.datetime.max().date())],
        "train_csv_sha256": sha256(train_path), "validation_csv_sha256": sha256(valid_path),
        "source_candidate_sha256": sha256(source_model), "training_config_sha256": sha256(cfg_path),
        "validation_metrics": {k: float(v) for k, v in computed.items()},
        "test_data_accessed": False,
    }
    (OUT / "frozen_candidate_spec.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    hashes = {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    (OUT / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    lines = ["# Alpaca final candidate freeze audit v1", "", "All pre-freeze checks passed. This freeze is a research candidate; it does not replace the prior official model freeze.", "", "| Check | Result | Evidence |", "|---|---|---|"]
    lines.extend(f"| {name} | {item['status']} | {item['evidence']} |" for name, item in checks.items())
    lines += ["", "No test data was opened, read, loaded, hashed, or evaluated by this audit. No live API call or trade occurred.", ""]
    (OUT / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"freeze_directory": str(OUT), "files": sorted(p.name for p in OUT.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
