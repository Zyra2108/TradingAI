"""Fixed, reproducible model comparison for the Alpaca ECL rebuild.

Reads only data/alpaca_rebuild_v1/{train,validation}.csv. It deliberately does
not reference or inspect test.csv. Candidate outputs are isolated here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/alpaca_rebuild_v1"
OUT = ROOT / "models/alpaca_candidates_v1"
FEATURES = [
    "close", "return_1", "return_3", "return_6", "return_12",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
    "ema_ratio_5", "ema_ratio_10", "ema_ratio_20", "rsi_14",
    "volatility_5", "volatility_10", "volatility_20", "candle_body",
    "candle_range", "upper_wick", "lower_wick", "minutes_since_open",
    "minutes_until_close", "day_of_week", "time_sin", "time_cos",
]
CLASSES = ["DOWN", "NEUTRAL", "UP"]
TARGET = "target"
SEED = 42

PARAMS = {
    "lightgbm_conservative": {
        "objective": "multiclass", "num_class": 3, "n_estimators": 350,
        "learning_rate": 0.02, "num_leaves": 15, "max_depth": -1,
        "min_child_samples": 60, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_lambda": 5.0, "class_weight": None,
        "random_state": SEED, "n_jobs": -1, "verbosity": -1,
    },
    "lightgbm_balanced": {
        "objective": "multiclass", "num_class": 3, "n_estimators": 500,
        "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1,
        "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_lambda": 1.0, "class_weight": "balanced",
        "random_state": SEED, "n_jobs": -1, "verbosity": -1,
    },
    "logistic_regression": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", "class_weight": None, "random_state": SEED},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data(filename: str):
    # These are the only dataset paths this script ever opens.
    frame = pd.read_csv(DATA_DIR / filename)
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate headers in {filename}: {frame.columns[frame.columns.duplicated()].tolist()}")
    # The dataset builder emits a metadata close and a feature close column.
    # pandas names the repeated header close.1; verify they are identical, then
    # use the first (same numeric values) for the sole feature named 'close'.
    if "close.1" in frame.columns:
        if not np.array_equal(frame["close"].to_numpy(), frame["close.1"].to_numpy()):
            raise ValueError(f"Metadata and model close columns differ in {filename}")
        frame.drop(columns=["close.1"], inplace=True)
    required = {"datetime", "future_datetime", "future_return", TARGET, *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} missing fields: {missing}")
    # Ensure features are exactly in the frozen order and carry expected types.
    if frame.loc[:, FEATURES].columns.tolist() != FEATURES:
        raise AssertionError("Feature order mismatch")
    frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
    frame["future_datetime"] = pd.to_datetime(frame.future_datetime, errors="raise")
    for col in FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame["day_of_week"] = frame.day_of_week.astype("int64")
    if frame[TARGET].isna().any() or not set(frame[TARGET]).issubset(CLASSES):
        raise ValueError(f"Invalid target values in {filename}")
    if not frame.datetime.is_monotonic_increasing or frame.datetime.duplicated().any():
        raise ValueError(f"Nonunique or unordered timestamps in {filename}")
    if not (frame.future_datetime == frame.datetime + pd.Timedelta(minutes=15)).all():
        raise ValueError(f"Invalid target horizon in {filename}")
    if frame[FEATURES].isna().any().any() or not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f"Missing/nonfinite features in {filename}")
    return frame


def preprocessing(scale_numeric: bool):
    numeric = [c for c in FEATURES if c != "day_of_week"]
    transformers = [
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            *(([("scale", StandardScaler())]) if scale_numeric else []),
        ]), numeric),
        ("day_of_week", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), ["day_of_week"]),
    ]
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def full_probabilities(estimator, X):
    raw = estimator.predict_proba(X)
    seen = list(estimator.classes_)
    out = np.zeros((len(X), len(CLASSES)), dtype=float)
    for j, label in enumerate(seen):
        out[:, CLASSES.index(str(label))] = raw[:, j]
    # Most-frequent dummy is represented as degenerate one-hot probabilities.
    if not np.isfinite(out).all() or not np.allclose(out.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Invalid probability matrix")
    return out


def summarize(name, y_true, pred, proba, timestamps):
    directional_labels = ["DOWN", "UP"]
    cm = confusion_matrix(y_true, pred, labels=CLASSES)
    per_class = classification_report(y_true, pred, labels=CLASSES, output_dict=True, zero_division=0)
    # Directional metrics use macro averaging over the two direction classes;
    # neutral predictions are false negatives for true direction and false
    # positives when predicted against neutral actuals.
    report = {
        "model": name,
        "n_validation": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1_3class": float(f1_score(y_true, pred, labels=CLASSES, average="macro", zero_division=0)),
        "directional_precision_macro": float(precision_score(y_true, pred, labels=directional_labels, average="macro", zero_division=0)),
        "directional_recall_macro": float(recall_score(y_true, pred, labels=directional_labels, average="macro", zero_division=0)),
        "directional_f1_macro": float(f1_score(y_true, pred, labels=directional_labels, average="macro", zero_division=0)),
        "directional_coverage": float(np.isin(pred, directional_labels).mean()),
        "log_loss": float(log_loss(y_true, proba, labels=CLASSES)),
        "predicted_distribution": {c: int((pred == c).sum()) for c in CLASSES},
        "actual_distribution": {c: int((np.asarray(y_true) == c).sum()) for c in CLASSES},
        "per_class": {c: {k: float(per_class[c][k]) for k in ("precision", "recall", "f1-score")} for c in CLASSES},
        "confusion_matrix_labels_rows_actual_cols_predicted": CLASSES,
        "confusion_matrix": cm.tolist(),
    }
    predictions = pd.DataFrame({"datetime": timestamps.astype(str), "actual": y_true, "prediction": pred})
    for i, c in enumerate(CLASSES):
        predictions[f"p_{c.lower()}"] = proba[:, i]
    return report, predictions, cm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in OUT.iterdir() if p.name not in {Path(__file__).name}]
    if existing:
        raise FileExistsError(f"Refusing to overwrite candidate artifacts: {existing}")
    train = load_data("train.csv")
    valid = load_data("validation.csv")
    # Date chronology and gap purge are checked without touching test.csv.
    if train.datetime.max() >= valid.datetime.min():
        raise ValueError("Training timestamps overlap/follow validation")
    if train.future_datetime.max() >= valid.datetime.min():
        raise ValueError("Training labels are not purged before validation")
    X_train, y_train = train[FEATURES], train[TARGET].astype(str).to_numpy()
    X_valid, y_valid = valid[FEATURES], valid[TARGET].astype(str).to_numpy()

    # Fixed model set; no search and no validation fitting.
    estimators = {
        "dummy_most_frequent": Pipeline([("preprocessor", preprocessing(False)), ("model", DummyClassifier(strategy="most_frequent"))]),
        # Current scikit-learn automatically uses multinomial loss for this
        # three-class lbfgs configuration (the removed multi_class argument is
        # no longer accepted in recent versions).
        "multinomial_logistic": Pipeline([("preprocessor", preprocessing(True)), ("model", LogisticRegression(**PARAMS["logistic_regression"]))]),
        "lightgbm_conservative": Pipeline([("preprocessor", preprocessing(False)), ("model", lgb.LGBMClassifier(**PARAMS["lightgbm_conservative"]))]),
        "lightgbm_balanced_B_start": Pipeline([("preprocessor", preprocessing(False)), ("model", lgb.LGBMClassifier(**PARAMS["lightgbm_balanced"]))]),
    }
    metrics, all_predictions, matrices = [], [], []
    for name, model in estimators.items():
        model.fit(X_train, y_train)
        proba = full_probabilities(model, X_valid)
        pred = np.asarray(CLASSES)[np.argmax(proba, axis=1)]
        report, predictions, cm = summarize(name, y_valid, pred, proba, valid.datetime)
        metrics.append(report)
        predictions.insert(0, "model", name)
        all_predictions.append(predictions)
        matrices.extend({"model": name, "actual": CLASSES[i], "predicted": CLASSES[j], "count": int(cm[i, j])} for i in range(3) for j in range(3))
        joblib.dump(model, OUT / f"{name}.joblib", compress=3)

    pd.DataFrame([{k: v for k, v in m.items() if k not in {"per_class", "confusion_matrix", "confusion_matrix_labels_rows_actual_cols_predicted", "predicted_distribution", "actual_distribution"}} for m in metrics]).to_csv(OUT / "validation_metrics.csv", index=False)
    (OUT / "validation_metrics_full.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT / "validation_predictions.csv", index=False)
    pd.DataFrame(matrices).to_csv(OUT / "validation_confusion_matrices.csv", index=False)
    pd.DataFrame([{"model": m["model"], "class": c, "predicted_count": m["predicted_distribution"][c], "predicted_fraction": m["predicted_distribution"][c] / m["n_validation"], "actual_count": m["actual_distribution"][c]} for m in metrics for c in CLASSES]).to_csv(OUT / "validation_class_distributions.csv", index=False)
    pd.DataFrame([{"model": m["model"], "class": c, **m["per_class"][c]} for m in metrics for c in CLASSES]).to_csv(OUT / "validation_per_class_metrics.csv", index=False)

    config = {
        "data_inputs": ["data/alpaca_rebuild_v1/train.csv", "data/alpaca_rebuild_v1/validation.csv"],
        "test_data_accessed": False,
        "train_sha256": sha256(DATA_DIR / "train.csv"), "validation_sha256": sha256(DATA_DIR / "validation.csv"),
        "train_rows": int(len(train)), "validation_rows": int(len(valid)),
        "train_date_range": [str(train.datetime.min().date()), str(train.datetime.max().date())],
        "validation_date_range": [str(valid.datetime.min().date()), str(valid.datetime.max().date())],
        "class_order": CLASSES, "feature_columns_exact_order": FEATURES, "feature_count": len(FEATURES),
        "categorical_feature": {"day_of_week": "int64, imputed most-frequent and one-hot encoded"},
        "numeric_preprocessing": "Median imputation fit inside train pipeline; StandardScaler for logistic regression only. Tree models use unscaled numeric features.",
        "target": "Saved target labels; project definition +15m, UP > +0.10%, DOWN < -0.10%, otherwise NEUTRAL.",
        "split": "Chronological train 2025-11-13–2026-08-05; validation 2026-08-06–2026-09-24. Validation is never used in fit.",
        "purge_check": "max training future_datetime < first validation decision timestamp",
        "seed": SEED, "parameters": PARAMS,
        "class_probability_order": CLASSES,
        "software_versions": {"python": __import__("sys").version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__},
        "selection_guidance": "Compare validation macro-F1 and directional F1/precision/recall with coverage; also consider accuracy and log loss. No single-metric automatic selection.",
    }
    (OUT / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Alpaca ECL candidates v1\n\n"
        "Four fixed candidates were trained on `train.csv`; evaluation uses `validation.csv` only. `test.csv` is not referenced by this script.\n\n"
        "The exact 24 B inputs are listed in `training_config.json`. The builder CSV has a repeated `close` header (metadata close plus feature close); the loader verifies values match and uses the named `close` feature without changing the source dataset. Metadata fields, including timestamps, future return and target, are excluded from X.\n\n"
        "Class order for probabilities and confusion matrices is DOWN, NEUTRAL, UP. Directional macro metrics average the DOWN and UP one-vs-rest metrics; coverage is the fraction predicted DOWN or UP.\n\n"
        "Model configurations, preprocessing, split dates, hashes and package versions are in `training_config.json`. Re-run from the project root with `python models/alpaca_candidates_v1/train_compare.py`; the script refuses to overwrite existing artifacts.\n",
        encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
