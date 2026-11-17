"""One-time, read-only diagnostic evaluation of saved ECL Alpaca IEX models.

The script reads the single canonical test split only for inference/scoring. It
never fits models, changes thresholds, or writes outside this new report folder.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "alpaca_all_models_test_diagnostic"
PRED_DIR = OUT / "predictions"
CM_DIR = OUT / "confusion_matrices"
TEST = ROOT / "data" / "alpaca_ml_pipeline_v1" / "test.csv"
TRAIN = ROOT / "data" / "alpaca_ml_pipeline_v1" / "train.csv"
DATA_MANIFEST = ROOT / "data" / "alpaca_ml_pipeline_v1" / "reproducibility_manifest.json"
CANONICAL = ROOT / "data" / "alpaca_ml_pipeline_v1" / "canonical_ecl_iex_5min.csv"
FEATURES = [
    "close", "return_1", "return_3", "return_6", "return_12",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
    "ema_ratio_5", "ema_ratio_10", "ema_ratio_20", "rsi_14",
    "volatility_5", "volatility_10", "volatility_20",
    "candle_body", "candle_range", "upper_wick", "lower_wick",
    "minutes_since_open", "minutes_until_close", "day_of_week", "time_sin", "time_cos",
]
CLASSES = ["DOWN", "NEUTRAL", "UP"]
DIRECTIONAL = ["DOWN", "UP"]

# The named models plus the frozen Balanced LightGBM B reference and the
# separate v1/v3 binary RF/XGBoost artifacts requested by the user.
MODEL_FILES: dict[str, list[str]] = {
    "logistic_regression_3class": ["models/alpaca_baselines_v1/logistic_regression.joblib"],
    "random_forest_classifier_3class": ["models/alpaca_baselines_v1/random_forest.joblib"],
    "xgboost_classifier_3class": ["models/alpaca_baselines_v1/xgboost.joblib"],
    "hist_gradient_boosting_classifier_3class": ["models/alpaca_baselines_v1/hist_gradient_boosting.joblib"],
    "frozen_balanced_lightgbm_B_reference": ["models/alpaca_final_freeze_v1/candidate_pipeline.joblib"],
    "binary_random_forest_v3": ["models/alpaca_binary_research_v3/random_forest.joblib"],
    "binary_xgboost_v3": ["models/alpaca_binary_research_v3/xgboost.joblib"],
    "binary_xgboost_v1_additional": ["models/alpaca_binary_research_v1/xgboost_binary.joblib"],
    "ridge_regression": ["models/alpaca_return_regression_v1/ridge_full_train.joblib"],
    "random_forest_regression": ["models/alpaca_return_regression_v1/random_forest_regressor_full_train.joblib"],
    "xgboost_regression": ["models/alpaca_return_regression_v1/xgboost_regressor_full_train.joblib"],
    "hist_gradient_boosting_regression": ["models/alpaca_return_regression_v1/hist_gradient_boosting_regressor_full_train.joblib"],
}
SEQUENCE_FILES: dict[int, list[str]] = {
    length: [
        f"models/alpaca_sequence_research_v1/causal_1d_cnn_len{length}.keras",
        f"models/alpaca_sequence_research_v1/channel_scaler_len{length}.joblib",
    ]
    for length in (6, 12, 24, 36)
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aligned_probabilities(model: Any, x: pd.DataFrame, expected: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return predict_proba columns in the saved spec's semantic class order."""
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(list(model.named_steps.values())[-1], "classes_", None)
    if classes is None:
        raise ValueError("saved estimator exposes no classes_ metadata")
    classes = list(classes)
    raw = np.asarray(model.predict_proba(x), dtype=float)
    if raw.ndim != 2 or raw.shape != (len(x), len(classes)):
        raise ValueError(f"predict_proba shape {raw.shape} does not match classes {len(classes)}")
    mapped: list[str] = []
    for value in classes:
        if isinstance(value, (int, np.integer)) or (isinstance(value, (float, np.floating)) and float(value).is_integer()):
            idx = int(value)
            if idx < 0 or idx >= len(expected):
                raise ValueError(f"encoded class index {idx} is outside saved class contract")
            mapped.append(expected[idx])
        else:
            mapped.append(str(value))
    if len(set(mapped)) != len(mapped) or not set(mapped).issubset(expected):
        raise ValueError(f"saved classes {mapped} do not match expected {expected}")
    output = np.zeros((len(x), len(expected)), dtype=float)
    for col, name in enumerate(mapped):
        output[:, expected.index(name)] = raw[:, col]
    if not np.isfinite(output).all() or (output < 0).any():
        raise ValueError("model returned invalid probabilities")
    if not np.allclose(output.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("model probability rows do not sum to 1")
    return output, mapped


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "n_evaluated": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in labels
        },
        "confusion_matrix_actual_rows_predicted_columns": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "prediction_distribution": {name: int(np.sum(y_pred == name)) for name in labels},
    }


def directional_metrics(y_all: np.ndarray, prediction_all: np.ndarray, eligible_mask: np.ndarray) -> dict[str, Any]:
    actual = y_all[eligible_mask]
    predicted = prediction_all[eligible_mask]
    out = metrics(actual, predicted, DIRECTIONAL)
    out["test_directional_rows_available"] = int(np.sum(np.isin(y_all, DIRECTIONAL)))
    out["eligible_directional_coverage"] = float(len(actual) / max(1, out["test_directional_rows_available"]))
    out["emitted_directional_signal_coverage_of_evaluable_rows"] = float(np.isin(prediction_all[eligible_mask], DIRECTIONAL).mean()) if len(actual) else 0.0
    # All actual NEUTRAL rows count as misses for these binary/sequence/regression models.
    out["all_candle_accuracy_neutral_wrong"] = float(np.sum((y_all == prediction_all) & np.isin(y_all, DIRECTIONAL)) / len(y_all))
    return out


def class_name(value: Any, class_order: list[str]) -> str:
    if isinstance(value, (int, np.integer)):
        return class_order[int(value)]
    text = str(value)
    if text in class_order:
        return text
    try:
        index = int(float(text))
        return class_order[index]
    except (ValueError, IndexError):
        raise ValueError(f"prediction label {value!r} does not match {class_order}")


def load_test_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("symbol"), manifest.get("feed"), manifest.get("timeframe")) != ("ECL", "IEX", "5Min"):
        raise ValueError("canonical ML manifest is not ECL Alpaca IEX 5Min")
    if manifest["split"]["date_ranges"]["test"] != ["2026-09-25", "2026-11-12"]:
        raise ValueError("canonical test split date range differs from the established test period")
    if not TEST.is_file() or not TRAIN.is_file():
        raise FileNotFoundError("canonical test or train split is unavailable")
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    expected = ["datetime", "future_datetime", "future_return", "target", *FEATURES]
    if list(train.columns) != expected or list(test.columns) != expected:
        raise ValueError("train/test columns or exact 24-feature order do not match the frozen contract")
    for split_name, frame in (("train", train), ("test", test)):
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        frame["future_datetime"] = pd.to_datetime(frame["future_datetime"], errors="raise")
        if frame["datetime"].duplicated().any() or not frame["datetime"].is_monotonic_increasing:
            raise ValueError(f"{split_name} rows are duplicated or not chronological")
        if not (frame.future_datetime - frame.datetime).eq(pd.Timedelta(minutes=15)).all():
            raise ValueError(f"{split_name} target endpoints are not exactly 15 minutes")
        expected_target = np.where(frame.future_return.to_numpy(float) > 0.10, "UP", np.where(frame.future_return.to_numpy(float) < -0.10, "DOWN", "NEUTRAL"))
        if not np.array_equal(expected_target, frame.target.astype(str).to_numpy()):
            raise ValueError(f"{split_name} target labels do not match saved +/-0.10% return labels")
        matrix = frame[FEATURES].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{split_name} feature matrix contains missing/nonfinite values")
    if test.datetime.dt.strftime("%Y-%m-%d").min() != "2026-09-25" or test.datetime.dt.strftime("%Y-%m-%d").max() != "2026-11-12":
        raise ValueError("test rows fall outside the established chronological test range")
    if train.datetime.max() >= test.datetime.min():
        raise ValueError("training/test chronological boundary is invalid")
    return train, test, manifest


def load_sequence_raw(manifest: dict[str, Any]) -> pd.DataFrame:
    if sha256(CANONICAL) != manifest["canonical_sha256"]:
        raise ValueError("canonical ECL IEX source hash differs from the dataset manifest")
    raw = pd.read_csv(CANONICAL)
    time_col = "timestamp_utc" if "timestamp_utc" in raw.columns else "timestamp"
    if time_col not in raw.columns:
        raise ValueError("canonical OHLCV source has no timestamp column")
    raw["timestamp_utc"] = pd.to_datetime(raw[time_col], utc=True, errors="raise")
    raw["datetime"] = raw.timestamp_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    raw = raw.sort_values("datetime", kind="stable").reset_index(drop=True)
    if raw.datetime.duplicated().any():
        raise ValueError("canonical OHLCV source has duplicate local candle timestamps")
    if raw.datetime.dt.strftime("%Y-%m-%d").max() != "2026-11-12":
        raise ValueError("canonical raw source extends beyond the authorized test period")
    cols = ["open", "high", "low", "close", "volume"]
    raw[cols] = raw[cols].apply(pd.to_numeric, errors="raise")
    vals = raw[cols].to_numpy(float)
    if not np.isfinite(vals).all() or (raw["volume"] < 0).any() or (raw[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("canonical raw sequence source contains invalid OHLCV")
    if (raw.low > raw[["open", "close"]].min(axis=1)).any() or (raw.high < raw[["open", "close"]].max(axis=1)).any() or (raw.low > raw.high).any():
        raise ValueError("canonical raw source contains structurally invalid OHLC")
    return raw


def make_sequences(raw: pd.DataFrame, test: pd.DataFrame, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the stored CNN input transform and strict session/gap rules."""
    raw_times = pd.DatetimeIndex(raw.datetime)
    raw_values = raw[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    positions = {timestamp: index for index, timestamp in enumerate(raw_times)}
    arrays: list[np.ndarray] = []
    eligible = np.zeros(len(test), dtype=bool)
    for row, stamp in enumerate(pd.DatetimeIndex(test.datetime)):
        end = positions.get(stamp)
        if end is None:
            continue
        start = end - length + 1
        if start < 0:
            continue
        times = raw_times[start:end + 1]
        if len(times) != length or times[-1] != stamp or times[0].date() != stamp.date():
            continue
        if any(times[i] - times[i - 1] != pd.Timedelta(minutes=5) or times[i].date() != stamp.date() for i in range(1, length)):
            continue
        values = raw_values[start:end + 1].copy()
        last_close = values[-1, 3]
        if not np.isfinite(last_close) or last_close <= 0 or (values[:, 4] < 0).any() or not np.isfinite(values).all():
            continue
        values[:, :4] = 100.0 * (values[:, :4] / last_close - 1.0)
        values[:, 4] = np.log1p(values[:, 4])
        arrays.append(values.astype(np.float32))
        eligible[row] = True
    result = np.stack(arrays) if arrays else np.empty((0, length, 5), dtype=np.float32)
    return result, eligible


def save_prediction_file(name: str, test: pd.DataFrame, prediction: np.ndarray, *, eligible: np.ndarray | None = None, probs: np.ndarray | None = None, class_order: list[str] | None = None, predicted_return: np.ndarray | None = None) -> None:
    frame = test[["datetime", "future_datetime", "target", "future_return"]].copy()
    frame["prediction"] = prediction
    if eligible is not None:
        frame["sequence_eligible"] = eligible
        frame.loc[~eligible, "prediction"] = "NOT_EVALUATED_INSUFFICIENT_CONTIGUOUS_SEQUENCE"
    if predicted_return is not None:
        frame["predicted_return_pct_points"] = predicted_return
    if probs is not None and class_order is not None:
        for index, cls in enumerate(class_order):
            frame[f"p_{cls.lower()}"] = probs[:, index]
    frame.to_csv(PRED_DIR / f"{name}.csv", index=False)


def evaluate() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "evaluation_metrics.json").exists():
        raise FileExistsError("diagnostic outputs already exist; refusing to overwrite one-time results")
    PRED_DIR.mkdir(exist_ok=False)
    CM_DIR.mkdir(exist_ok=False)
    artifact_paths = [ROOT / item for values in MODEL_FILES.values() for item in values]
    artifact_paths += [ROOT / item for values in SEQUENCE_FILES.values() for item in values]
    missing = [str(p.relative_to(ROOT)) for p in artifact_paths if not p.is_file()]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in artifact_paths if p.is_file()}
    (OUT / "artifact_hashes_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")

    train, test, manifest = load_test_frame()
    y_all = test.target.astype(str).to_numpy()
    directional_mask = np.isin(y_all, DIRECTIONAL)
    x = test[FEATURES]
    train_y = train.target.astype(str).to_numpy()
    train_class_counts = {cls: int(np.sum(train_y == cls)) for cls in CLASSES}
    train_directional_counts = {cls: int(np.sum(train_y == cls)) for cls in DIRECTIONAL}
    train_majority_3class = max(CLASSES, key=lambda cls: train_class_counts[cls])
    train_majority_direction = max(DIRECTIONAL, key=lambda cls: train_directional_counts[cls])
    y = y_all
    model_records: dict[str, dict[str, Any]] = {}
    prediction_records: list[dict[str, Any]] = []

    def add_3class(name: str, artifact: str, model: Any) -> None:
        p, _ = aligned_probabilities(model, x, CLASSES)
        pred = np.asarray(CLASSES, dtype=object)[np.argmax(p, axis=1)]
        result = metrics(y_all, pred, CLASSES)
        result.update({"model_type": "three_class_classifier", "artifact": artifact, "n_test_rows": len(test), "all_candle_accuracy": result["accuracy"], "directional_accuracy_on_actual_up_down": float(np.mean(pred[directional_mask] == y_all[directional_mask])), "directional_prediction_coverage_all_rows": float(np.isin(pred, DIRECTIONAL).mean()), "class_probability_order": CLASSES})
        model_records[name] = result
        save_prediction_file(name, test, pred, probs=p, class_order=CLASSES)
        pd.DataFrame(result["confusion_matrix_actual_rows_predicted_columns"], index=CLASSES, columns=CLASSES).to_csv(CM_DIR / f"{name}.csv", index=True)

    def add_binary(name: str, artifact: str, model: Any, expected: list[str] = DIRECTIONAL) -> None:
        p, _ = aligned_probabilities(model, x, expected)
        pred = np.asarray(expected, dtype=object)[np.argmax(p, axis=1)]
        result = directional_metrics(y_all, pred, directional_mask)
        result.update({"model_type": "binary_classifier", "artifact": artifact, "n_test_rows": len(test), "prediction_count": len(test), "class_probability_order": expected})
        model_records[name] = result
        save_prediction_file(name, test, pred, probs=p, class_order=expected)
        pd.DataFrame(result["confusion_matrix_actual_rows_predicted_columns"], index=DIRECTIONAL, columns=DIRECTIONAL).to_csv(CM_DIR / f"{name}.csv", index=True)

    for name, paths in MODEL_FILES.items():
        if name.startswith("ridge_regression") or name.endswith("_regression"):
            continue
        if name in {"binary_random_forest_v3", "binary_xgboost_v3", "binary_xgboost_v1_additional"}:
            continue
        try:
            artifact = paths[0]
            model = joblib.load(ROOT / artifact)
            if name == "frozen_balanced_lightgbm_B_reference":
                spec = json.loads((ROOT / "models/alpaca_final_freeze_v1/frozen_candidate_spec.json").read_text(encoding="utf-8"))
                if spec["features_exact_order"] != FEATURES or spec["class_order"] != CLASSES or spec["target"]["horizon_minutes"] != 15:
                    raise ValueError("frozen B feature/class/horizon specification mismatch")
            if name.endswith("_3class") or name == "frozen_balanced_lightgbm_B_reference":
                add_3class(name, artifact, model)
            else:
                add_binary(name, artifact, model)
        except Exception as exc:  # Report each model failure without omitting it.
            model_records[name] = {"status": "FAILED_OR_INCOMPATIBLE", "model_type": "three_class_classifier" if name.endswith("_3class") or name.startswith("frozen_") else "binary_classifier", "artifact": paths[0], "error_type": type(exc).__name__, "safe_error": str(exc)}

    for name in ("binary_random_forest_v3", "binary_xgboost_v3", "binary_xgboost_v1_additional"):
        artifact = MODEL_FILES[name][0]
        try:
            model = joblib.load(ROOT / artifact)
            add_binary(name, artifact, model, DIRECTIONAL)
        except Exception as exc:
            model_records[name] = {"status": "FAILED_OR_INCOMPATIBLE", "model_type": "binary_classifier", "artifact": artifact, "error_type": type(exc).__name__, "safe_error": str(exc)}

    # Regression models use their saved fitted pipelines and the fixed zero sign boundary.
    for name in ("ridge_regression", "random_forest_regression", "xgboost_regression", "hist_gradient_boosting_regression"):
        artifact = MODEL_FILES[name][0]
        try:
            model = joblib.load(ROOT / artifact)
            pred_return = np.asarray(model.predict(x), dtype=float)
            if not np.isfinite(pred_return).all():
                raise ValueError("predicted returns include nonfinite values")
            pred = np.where(pred_return > 0.0, "UP", "DOWN")
            result = directional_metrics(y_all, pred, directional_mask)
            actual_return = test.future_return.to_numpy(float)
            result.update({
                "model_type": "return_regressor_zero_sign",
                "artifact": artifact,
                "n_test_rows": len(test),
                "prediction_count": len(test),
                "decision_boundary_pct_points": 0.0,
                "mae_pct_points": float(np.mean(np.abs(actual_return - pred_return))),
                "rmse_pct_points": float(np.sqrt(np.mean((actual_return - pred_return) ** 2))),
                "return_correlation": float(np.corrcoef(actual_return, pred_return)[0, 1]) if np.std(pred_return) > 0 else None,
            })
            model_records[name] = result
            save_prediction_file(name, test, pred, predicted_return=pred_return)
            pd.DataFrame(result["confusion_matrix_actual_rows_predicted_columns"], index=DIRECTIONAL, columns=DIRECTIONAL).to_csv(CM_DIR / f"{name}.csv", index=True)
        except Exception as exc:
            model_records[name] = {"status": "FAILED_OR_INCOMPATIBLE", "model_type": "return_regressor_zero_sign", "artifact": artifact, "error_type": type(exc).__name__, "safe_error": str(exc)}

    # CNNs use the saved channel scaler, model, exact OHLCV transform, and 0.5 threshold.
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        raw = load_sequence_raw(manifest)
        for length, paths in SEQUENCE_FILES.items():
            model_path, scaler_path = paths
            try:
                sequences, eligible = make_sequences(raw, test, length)
                scaler = joblib.load(ROOT / scaler_path)
                model = tf.keras.models.load_model(ROOT / model_path, compile=False)
                if len(sequences):
                    shape = sequences.shape
                    transformed = scaler.transform(sequences.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
                    p_up = np.asarray(model.predict(transformed, verbose=0), dtype=float).reshape(-1)
                    if not np.isfinite(p_up).all() or ((p_up < 0) | (p_up > 1)).any():
                        raise ValueError("CNN returned invalid UP probabilities")
                    pred = np.where(p_up >= 0.5, "UP", "DOWN")
                    p_all = np.full(len(test), np.nan, dtype=float)
                    p_all[eligible] = p_up
                    pred_all = np.full(len(test), "", dtype=object)
                    pred_all[eligible] = pred
                    result = directional_metrics(y_all, pred_all, directional_mask & eligible)
                    result.update({"model_type": "sequence_cnn_binary", "artifact": model_path, "scaler_artifact": scaler_path, "sequence_length_candles": length, "n_test_rows": len(test), "n_sequence_eligible_rows": int(eligible.sum()), "sequence_directional_eligibility_coverage": float(np.sum(directional_mask & eligible) / max(1, directional_mask.sum())), "prediction_count": int(eligible.sum()), "decision_threshold_p_up": 0.5, "prediction_distribution": {"DOWN": int(np.sum(pred == "DOWN")), "UP": int(np.sum(pred == "UP"))}})
                    model_records[f"cnn_{length}_candles"] = result
                    frame = test[["datetime", "future_datetime", "target", "future_return"]].copy()
                    frame["sequence_eligible"] = eligible
                    frame["prediction"] = np.where(eligible, pred_all, "NOT_EVALUATED_INSUFFICIENT_CONTIGUOUS_SEQUENCE")
                    frame["p_up"] = p_all
                    frame["p_down"] = np.where(np.isfinite(p_all), 1.0 - p_all, np.nan)
                    frame.to_csv(PRED_DIR / f"cnn_{length}_candles.csv", index=False)
                    pd.DataFrame(result["confusion_matrix_actual_rows_predicted_columns"], index=DIRECTIONAL, columns=DIRECTIONAL).to_csv(CM_DIR / f"cnn_{length}_candles.csv", index=True)
                else:
                    raise ValueError("no test sequences met saved continuity rules")
            except Exception as exc:
                model_records[f"cnn_{length}_candles"] = {"status": "FAILED_OR_INCOMPATIBLE", "model_type": "sequence_cnn_binary", "artifact": model_path, "scaler_artifact": scaler_path, "error_type": type(exc).__name__, "safe_error": str(exc)}
    except Exception as exc:
        for length, paths in SEQUENCE_FILES.items():
            model_records.setdefault(f"cnn_{length}_candles", {"status": "FAILED_OR_INCOMPATIBLE", "model_type": "sequence_cnn_binary", "artifact": paths[0], "scaler_artifact": paths[1], "error_type": type(exc).__name__, "safe_error": str(exc)})

    # Train-derived baselines: always-neutral and training majority; include
    # explicit UP/DOWN reference rows without deriving their choice from test labels.
    baseline_records: dict[str, Any] = {}
    for name, pred in (
        ("always_neutral", np.repeat("NEUTRAL", len(test))),
        ("training_majority_3class", np.repeat(train_majority_3class, len(test))),
        ("training_majority_direction", np.repeat(train_majority_direction, len(test))),
        ("always_up", np.repeat("UP", len(test))),
        ("always_down", np.repeat("DOWN", len(test))),
    ):
        full = metrics(y_all, pred, CLASSES)
        full["model_type"] = "baseline"
        full["training_majority_3class"] = train_majority_3class
        full["training_majority_direction"] = train_majority_direction
        full["all_candle_accuracy"] = full["accuracy"]
        full["directional"] = directional_metrics(y_all, pred, directional_mask)
        baseline_records[name] = full
        save_prediction_file(f"baseline_{name}", test, pred)
        pd.DataFrame(full["confusion_matrix_actual_rows_predicted_columns"], index=CLASSES, columns=CLASSES).to_csv(CM_DIR / f"baseline_{name}.csv", index=True)

    after = {str(p.relative_to(ROOT)): sha256(p) for p in artifact_paths if p.is_file()}
    (OUT / "artifact_hashes_after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    hashes_unchanged = before == after and not missing
    class_distribution = {cls: int(np.sum(y_all == cls)) for cls in CLASSES}
    records = {name: value for name, value in model_records.items()}
    for name, value in records.items():
        if value.get("status") != "FAILED_OR_INCOMPATIBLE":
            value["status"] = "PASS"
    metrics_payload = {
        "evaluation": {
            "dataset": str(TEST.relative_to(ROOT)),
            "symbol": "ECL", "provider": "Alpaca", "feed": "IEX", "timeframe": "5Min",
            "horizon_minutes": 15, "label_threshold_pct": {"down_lt": -0.10, "up_gt": 0.10},
            "test_rows": int(len(test)), "date_start": str(test.datetime.min().date()), "date_end": str(test.datetime.max().date()),
            "first_decision_timestamp": str(test.datetime.iloc[0]), "last_decision_timestamp": str(test.datetime.iloc[-1]),
            "last_future_endpoint": str(test.future_datetime.max()),
            "class_order_3class": CLASSES, "class_distribution": class_distribution,
            "directional_rows": int(directional_mask.sum()),
            "training_class_distribution_for_baseline": train_class_counts,
            "training_directional_distribution_for_baseline": train_directional_counts,
            "training_majority_3class": train_majority_3class,
            "training_majority_direction": train_majority_direction,
            "test_target_policy": "Existing target values used unchanged; labels verified against saved future_return; exact +15-minute endpoint timestamp verified.",
            "post_test_tuning_or_model_changes": False,
            "preexisting_validation_or_walk_forward_metrics_read_only_for_comparison": True,
            "artifact_hashes_unchanged": hashes_unchanged,
            "missing_required_artifacts": missing,
        },
        "models": records,
        "baselines": baseline_records,
    }
    (OUT / "evaluation_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    rows = []
    for name, item in records.items():
        row = {"model": name, "status": item.get("status"), "model_type": item.get("model_type"), "n_evaluated": item.get("n_evaluated"), "accuracy": item.get("accuracy"), "balanced_accuracy": item.get("balanced_accuracy"), "macro_f1": item.get("macro_f1"), "directional_accuracy_on_actual_up_down": item.get("directional_accuracy_on_actual_up_down"), "up_precision": item.get("per_class", {}).get("UP", {}).get("precision"), "up_recall": item.get("per_class", {}).get("UP", {}).get("recall"), "down_precision": item.get("per_class", {}).get("DOWN", {}).get("precision"), "down_recall": item.get("per_class", {}).get("DOWN", {}).get("recall"), "neutral_precision": item.get("per_class", {}).get("NEUTRAL", {}).get("precision"), "neutral_recall": item.get("per_class", {}).get("NEUTRAL", {}).get("recall"), "all_candle_accuracy": item.get("all_candle_accuracy_neutral_wrong", item.get("all_candle_accuracy", item.get("accuracy"))), "eligible_directional_coverage": item.get("eligible_directional_coverage"), "sequence_directional_eligibility_coverage": item.get("sequence_directional_eligibility_coverage"), "prediction_count": item.get("prediction_count", item.get("n_test_rows")), "mae_pct_points": item.get("mae_pct_points"), "rmse_pct_points": item.get("rmse_pct_points"), "return_correlation": item.get("return_correlation"), "confusion_matrix": json.dumps(item.get("confusion_matrix_actual_rows_predicted_columns")) if item.get("confusion_matrix_actual_rows_predicted_columns") is not None else "", "artifact": item.get("artifact"), "error": item.get("safe_error", "")}
        rows.append(row)
    flat = pd.DataFrame(rows)
    flat.to_csv(OUT / "per_model_metrics.csv", index=False)
    baseline_rows = []
    for name, item in baseline_records.items():
        dm = item["directional"]
        baseline_rows.append({"baseline": name, "three_class_accuracy": item["accuracy"], "three_class_balanced_accuracy": item["balanced_accuracy"], "three_class_macro_f1": item["macro_f1"], "directional_accuracy": dm["accuracy"], "directional_balanced_accuracy": dm["balanced_accuracy"], "directional_macro_f1": dm["macro_f1"], "all_candle_accuracy": item["accuracy"], "confusion_matrix_3class": json.dumps(item["confusion_matrix_actual_rows_predicted_columns"]), "confusion_matrix_directional": json.dumps(dm["confusion_matrix_actual_rows_predicted_columns"])})
    pd.DataFrame(baseline_rows).to_csv(OUT / "baseline_metrics.csv", index=False)

    multiclass = flat[(flat.status == "PASS") & (flat.model_type == "three_class_classifier")].sort_values(["accuracy", "balanced_accuracy"], ascending=False)
    directional = flat[(flat.status == "PASS") & (flat.model_type.isin(["binary_classifier", "sequence_cnn_binary", "return_regressor_zero_sign"]))].sort_values(["accuracy", "balanced_accuracy"], ascending=False)
    multiclass.to_csv(OUT / "ranked_three_class_models.csv", index=False)
    directional.to_csv(OUT / "ranked_directional_models.csv", index=False)
    all_candle = flat[flat.status == "PASS"].sort_values("all_candle_accuracy", ascending=False)
    all_candle.to_csv(OUT / "ranked_all_candle_accuracy.csv", index=False)
    inventory_rows = []
    for name, paths in list(MODEL_FILES.items()) + [(f"cnn_{n}_candles", files) for n, files in SEQUENCE_FILES.items()]:
        inventory_rows.append({"model": name, "artifacts": ";".join(paths), "status": records.get(name, {}).get("status", "NOT_EVALUATED"), "reason": records.get(name, {}).get("safe_error", "Included in requested evaluation.")})
    inventory_rows.extend([
        {"model": "alpaca_binary_research_v1_lightgbm_binary", "artifacts": "models/alpaca_binary_research_v1/lightgbm_binary.joblib", "status": "NOT_IN_REQUESTED_MODEL_SET", "reason": "The request adds separate binary Random Forest and XGBoost baselines; this separate LightGBM binary candidate is outside that specified set."},
        {"model": "alpaca_binary_research_v3_logistic_and_hist_gradient_boosting_binary", "artifacts": "models/alpaca_binary_research_v3/logistic_regression.joblib; models/alpaca_binary_research_v3/hist_gradient_boosting.joblib", "status": "NOT_IN_REQUESTED_MODEL_SET", "reason": "The requested additional binary artifacts are Random Forest and XGBoost; the named four-model classification group is the three-class baseline suite."},
        {"model": "one_minute_and_alternative_horizon_research_artifacts", "artifacts": "models/alpaca_1min_research_v1/; models/alpaca_1min_horizon_research_v1/; models/alpaca_target_research_v1/", "status": "INCOMPATIBLE_TASK_CONTRACT", "reason": "Different candle frequency, horizon, or target/policy; not comparable to ECL five-minute, 15-minute +/-0.10% evaluation."},
        {"model": "alpaca_candidates_v1_additional_ensembles", "artifacts": "models/alpaca_candidates_v1/ensemble/", "status": "NOT_IN_REQUESTED_MODEL_SET", "reason": "Ensemble candidates are outside the explicit model list; the frozen Balanced LightGBM B is included as an additional three-class reference."},
    ])
    pd.DataFrame(inventory_rows).to_csv(OUT / "model_inventory.csv", index=False)
    return metrics_payload


if __name__ == "__main__":
    result = evaluate()
    ev = result["evaluation"]
    print(json.dumps({"test_rows": ev["test_rows"], "date_start": ev["date_start"], "date_end": ev["date_end"], "class_distribution": ev["class_distribution"], "models": {k: v.get("status") for k, v in result["models"].items()}, "artifact_hashes_unchanged": ev["artifact_hashes_unchanged"]}, indent=2))
