"""Read-only inference and evaluation metadata for existing research models.

These models are deliberately separate from the frozen production prediction
engine. Their latest-candle output is an unpersisted research preview and is
never inserted into either prediction ledger.
"""
from __future__ import annotations

import csv
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from live_app.calendar import NY
from live_app.features import calculate_b_features, ready_feature_row

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "close", "return_1", "return_3", "return_6", "return_12",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
    "ema_ratio_5", "ema_ratio_10", "ema_ratio_20", "rsi_14",
    "volatility_5", "volatility_10", "volatility_20", "candle_body",
    "candle_range", "upper_wick", "lower_wick", "minutes_since_open",
    "minutes_until_close", "day_of_week", "time_sin", "time_cos",
]
HORIZON_MINUTES = 15
CLASS_ORDER = ["DOWN", "NEUTRAL", "UP"]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "xgboost_multiclass": {
        "name": "XGBoost Multiclass", "kind": "multiclass",
        "path": ROOT / "models/alpaca_three_class_v1/estimators/xgboost_multiclass_train_only.joblib",
        "sha256": "08aea3085131c3e20625a7b9c56fd161741f58408c02a7d30e807bbc4bea228b",
        "class_labels": ["DOWN", "NEUTRAL", "UP"],
        "summary": "Three-class research model; experimental, not a production replacement.",
    },
    "ridge_return": {
        "name": "Ridge Regression Directional", "kind": "regression",
        "path": ROOT / "models/alpaca_return_regression_v1/ridge_full_train.joblib",
        "sha256": "dc01a3cca5aa519a3a953aede4e6ce4fc6053d8d0b2d3e257561c8535959f3b7",
        "class_labels": None,
        "summary": "Continuous return estimate in percentage points; sign maps to UP or DOWN.",
    },
    "xgboost_binary_v3": {
        "name": "XGBoost Binary Classifier v3", "kind": "binary",
        "path": ROOT / "models/alpaca_binary_research_v3/xgboost.joblib",
        "sha256": "b9fc849ec1c867c91128ce6ae62b6d39d73d64226160380988822e7ee5055daa",
        "class_labels": ["DOWN", "UP"],
        "summary": "Binary directional research model; it does not emit HOLD/NEUTRAL.",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_metric(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _json_confusion(value: Any) -> list[list[int]] | None:
    if isinstance(value, list):
        return value
    if value is None:
        return None
    try:
        result = json.loads(str(value))
        return result if isinstance(result, list) else None
    except (ValueError, TypeError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def build_evaluation_catalog() -> dict[str, Any]:
    """Read only the saved OOF and validation summaries for these artifacts."""
    three_root = ROOT / "reports/alpaca_three_class_v1"
    binary_root = ROOT / "reports/alpaca_binary_research_v3"
    return_root = ROOT / "reports/alpaca_return_regression_v1"
    three_val = json.loads((three_root / "validation_metrics.json").read_text(encoding="utf-8"))
    three_wf = _read_csv(three_root / "walk_forward_summary.csv")
    three_folds = _read_csv(three_root / "walk_forward_fold_table.csv")
    three_wf_row = next(row for row in three_wf if row["model"] == "xgboost_multiclass")
    three_fold_rows = [row for row in three_folds if row["model"] == "xgboost_multiclass"]
    three_metrics = three_val["metrics"]

    binary_val = _read_csv(binary_root / "validation_model_comparison.csv")
    binary_row = next(row for row in binary_val if row["family"] == "xgboost")
    binary_wf = _read_csv(binary_root / "walk_forward_config_comparison.csv")
    binary_wf_row = next(row for row in binary_wf if row["family"] == "xgboost" and row["configuration"] == "xgb_shallow_unweighted")
    binary_confusions = _read_csv(binary_root / "validation_confusion_matrices.csv")
    binary_cm = [[0, 0], [0, 0]]
    for row in binary_confusions:
        if row["family"] == "xgboost":
            i = {"DOWN": 0, "UP": 1}[row["actual"]]
            j = {"DOWN": 0, "UP": 1}[row["predicted"]]
            binary_cm[i][j] = int(row["count"])

    ridge_val_rows = _read_csv(return_root / "validation_metrics.csv")
    ridge_val = next(row for row in ridge_val_rows if row["model"] == "ridge")
    ridge_wf_rows = _read_csv(return_root / "walk_forward_overall_metrics.csv")
    ridge_wf = next(row for row in ridge_wf_rows if row["model"] == "ridge")
    ridge_fold_rows = [row for row in _read_csv(return_root / "walk_forward_fold_metrics.csv") if row["model"] == "ridge"]

    def directional_validation(row: dict[str, Any], *, regression: bool) -> dict[str, Any]:
        if regression:
            return {
                "n": int(float(row["n_directional"])),
                "accuracy": _finite_metric(row["directional_accuracy"]),
                "balanced_accuracy": _finite_metric(row["balanced_accuracy"]),
                "macro_f1": _finite_metric(row["directional_macro_f1"]),
                "up_precision": _finite_metric(row["up_precision"]), "up_recall": _finite_metric(row["up_recall"]),
                "down_precision": _finite_metric(row["down_precision"]), "down_recall": _finite_metric(row["down_recall"]),
                "all_candle_accuracy": _finite_metric(row["all_candle_directional_accuracy_neutral_wrong"]),
                "coverage": _finite_metric(row["directional_signal_coverage"]),
                "mae": _finite_metric(row["mae_all_rows_pct_points"]),
                "rmse": _finite_metric(row["rmse_all_rows_pct_points"]),
                "confusion_matrix": _json_confusion(row["directional_confusion_matrix_actual_rows_predicted_columns_down_up"]),
            }
        down_tp, down_fn = binary_cm[0][0], binary_cm[0][1]
        up_fn, up_tp = binary_cm[1][0], binary_cm[1][1]
        down_precision = down_tp / (down_tp + up_fn) if down_tp + up_fn else 0.0
        down_recall = down_tp / (down_tp + down_fn) if down_tp + down_fn else 0.0
        up_precision = up_tp / (up_tp + down_fn) if up_tp + down_fn else 0.0
        up_recall = up_tp / (up_tp + up_fn) if up_tp + up_fn else 0.0
        return {
            "n": int(float(row["n_directional"])),
            "accuracy": _finite_metric(row["accuracy"]),
            "balanced_accuracy": _finite_metric(row["balanced_accuracy"]),
            "macro_f1": _finite_metric(row["macro_f1"]),
            "all_candle_accuracy": _finite_metric(row["all_candle_exact_directional_accuracy_neutral_counts_wrong"]),
            "coverage": 1.0,
            "up_precision": up_precision, "up_recall": up_recall,
            "down_precision": down_precision, "down_recall": down_recall,
            "confusion_matrix": binary_cm,
        }

    catalog = {
        "baselines": {
            "validation_rows": int(three_val["n"]),
            "always_neutral_accuracy_all_rows": float(three_val["class_distribution"]["NEUTRAL"]) / float(three_val["n"]),
            "directional_majority_accuracy_eligible_rows": max(
                int(three_val["class_distribution"]["DOWN"]), int(three_val["class_distribution"]["UP"])
            ) / (int(three_val["class_distribution"]["DOWN"]) + int(three_val["class_distribution"]["UP"])),
            "directional_rows": int(three_val["class_distribution"]["DOWN"]) + int(three_val["class_distribution"]["UP"]),
        },
        "xgboost_multiclass": {
            "validation": {
                "population": "all three-class rows", "date_start": three_val["date_start"], "date_end": three_val["date_end"],
                "n": int(three_val["n"]), "accuracy": three_metrics["accuracy"],
                "balanced_accuracy": three_metrics["balanced_accuracy"], "macro_f1": three_metrics["macro_f1"],
                "weighted_f1": three_metrics["weighted_f1"], "per_class": three_metrics["per_class"],
                "directional_macro_f1": three_metrics["directional_macro_f1"],
                "coverage": three_metrics["directional_prediction_coverage_all_rows"],
                "prediction_distribution": three_metrics["prediction_distribution"],
                "confusion_matrix": three_metrics["confusion_matrix_actual_rows_predicted_columns"],
            },
            "walk_forward": {
                "population": "chronological training OOF", "n": int(three_wf_row["n"]),
                "mean_fold_accuracy": float(three_wf_row["mean_fold_accuracy"]), "std_fold_accuracy": float(three_wf_row["std_fold_accuracy"]),
                "mean_fold_macro_f1": float(three_wf_row["mean_fold_macro_f1"]), "std_fold_macro_f1": float(three_wf_row["std_fold_macro_f1"]),
                "mean_fold_balanced_accuracy": float(three_wf_row["mean_fold_balanced_accuracy"]),
                "folds": [{"fold": int(row["fold"]), "accuracy": float(row["oof_accuracy"]), "macro_f1": float(row["oof_macro_f1"]), "balanced_accuracy": float(row["oof_balanced_accuracy"])} for row in three_fold_rows],
            },
            "training": {
                "population": "in-sample fit scores within each walk-forward training fold",
                "mean_fold_accuracy": float(np.mean([float(row["train_accuracy"]) for row in three_fold_rows])),
                "mean_fold_macro_f1": float(np.mean([float(row["train_macro_f1"]) for row in three_fold_rows])),
            },
        },
        "ridge_return": {
            "validation": {"population": "actual UP/DOWN rows; NEUTRAL counted wrong only in all-candle accuracy", "date_start": "2026-08-06", "date_end": "2026-09-24", **directional_validation(ridge_val, regression=True)},
            "walk_forward": {"population": "chronological training OOF", "n": int(ridge_wf["n_directional"]),
                "mean_fold_accuracy": _finite_metric(ridge_wf["directional_accuracy"]),
                "mean_fold_balanced_accuracy": _finite_metric(ridge_wf["balanced_accuracy"]),
                "mean_fold_macro_f1": _finite_metric(ridge_wf["directional_macro_f1"]),
                "folds": [{"fold": int(row["fold"]), "accuracy": float(row["directional_accuracy"]), "macro_f1": float(row["directional_macro_f1"]), "balanced_accuracy": float(row["balanced_accuracy"])} for row in ridge_fold_rows],
            },
            "training": {"metrics_available": False, "note": "Saved report does not include in-sample class metrics; chronological OOF metrics are shown."},
        },
        "xgboost_binary_v3": {
            "validation": {"population": "actual UP/DOWN rows; NEUTRAL counted wrong only in all-candle accuracy", "date_start": "2026-08-06", "date_end": "2026-09-24", **directional_validation(binary_row, regression=False)},
            "walk_forward": {"population": "chronological training OOF", "n": None,
                "mean_fold_accuracy": float(binary_wf_row["mean_accuracy"]), "std_fold_accuracy": float(binary_wf_row["std_accuracy"]),
                "mean_fold_balanced_accuracy": float(binary_wf_row["mean_balanced_accuracy"]), "mean_fold_macro_f1": float(binary_wf_row["mean_macro_f1"]),
                "folds": []},
            "training": {"metrics_available": False, "note": "Saved report does not include in-sample class metrics; chronological OOF metrics are shown."},
        },
    }
    # Include binary XGBoost's individual folds, not only the four-fold mean.
    for row in _read_csv(binary_root / "walk_forward_fold_metrics.csv"):
        if row["family"] == "xgboost" and row["configuration"] == "xgb_shallow_unweighted":
            catalog["xgboost_binary_v3"]["walk_forward"]["folds"].append({
                "fold": int(row["fold"]), "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]), "balanced_accuracy": float(row["balanced_accuracy"]),
            })
    return catalog


class ResearchModelService:
    """Lazy, integrity-checked research model previews over completed candles."""

    def __init__(self, *, feature_order: list[str] | None = None,
                 model_loader: Callable[[Path], Any] = joblib.load,
                 feature_builder: Callable[..., pd.DataFrame] = calculate_b_features,
                 clock: Callable[[], datetime] | None = None):
        self.feature_order = list(feature_order or FEATURES)
        if self.feature_order != FEATURES:
            raise ValueError("Research models require the exact 24-feature B schema")
        self.model_loader = model_loader
        self.feature_builder = feature_builder
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._models: dict[str, Any] = {}
        self._preview_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._catalog: dict[str, Any] | None = None
        self._feature_lock = threading.RLock()
        self._latest_feature_key: tuple[Any, ...] | None = None
        self._latest_feature_row: pd.Series | None = None
        self._latest_feature_missing: list[str] = []

    def _features_for_latest(self, ordered: list[dict[str, Any]]) -> tuple[pd.Series | None, list[str]]:
        latest = ordered[-1]
        key = (int(latest["time"]), len(ordered), int(ordered[0]["time"]),
               *(float(latest[name]) for name in ("open", "high", "low", "close")))
        with self._feature_lock:
            if key == self._latest_feature_key:
                return (self._latest_feature_row.copy() if self._latest_feature_row is not None else None,
                        list(self._latest_feature_missing))
            frame = pd.DataFrame([{
                "datetime": datetime.fromtimestamp(int(row["time"]), timezone.utc).astimezone(NY).replace(tzinfo=None),
                "open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"],
            } for row in ordered])
            features = self.feature_builder(frame, self.feature_order)
            feature_row, missing = ready_feature_row(features, features.iloc[-1]["datetime"])
            self._latest_feature_key = key
            self._latest_feature_row = feature_row.copy() if feature_row is not None else None
            self._latest_feature_missing = list(missing)
            return (feature_row.copy() if feature_row is not None else None, list(missing))

    def catalog(self) -> dict[str, Any]:
        if self._catalog is None:
            evaluations = build_evaluation_catalog()
            self._catalog = {
                "baselines": evaluations["baselines"],
                "models": [{
                    "id": model_id, "name": spec["name"], "kind": spec["kind"],
                    "experimental": True, "horizon_minutes": HORIZON_MINUTES,
                    "features": len(self.feature_order), "probabilities_available": spec["kind"] != "regression",
                    "summary": spec["summary"], "evaluation": evaluations[model_id],
                } for model_id, spec in MODEL_SPECS.items()],
                "evaluation_disclosure": {
                    "validation": "2026-08-06 through 2026-09-24; reused in prior research, not a pristine holdout.",
                    "walk_forward": "Training-period chronological out-of-fold predictions.",
                    "test_diagnostics": "Previously exposed test diagnostics are not fresh out-of-sample evidence and are not attached to these exact candidate cards.",
                },
            }
        return self._catalog

    def _model(self, model_id: str) -> Any:
        if model_id not in MODEL_SPECS:
            raise KeyError("Unknown research model")
        if model_id not in self._models:
            spec = MODEL_SPECS[model_id]
            path: Path = spec["path"]
            if not path.is_file() or _sha256(path) != spec["sha256"]:
                raise RuntimeError("Research model artifact is missing or failed integrity validation")
            model = self.model_loader(path)
            if list(getattr(model, "feature_names_in_", [])) != self.feature_order:
                raise RuntimeError("Research model feature order does not match the frozen B schema")
            if spec["kind"] == "multiclass" and [int(value) for value in model.classes_] != [0, 1, 2]:
                raise RuntimeError("Multiclass research model class mapping is invalid")
            if spec["kind"] == "binary" and [int(value) for value in model.classes_] != [0, 1]:
                raise RuntimeError("Binary research model class mapping is invalid")
            self._models[model_id] = model
        return self._models[model_id]

    def predict_latest(self, candles: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
        if model_id not in MODEL_SPECS:
            raise KeyError("Unknown research model")
        ordered = sorted(candles, key=lambda row: int(row["time"]))
        if not ordered:
            return {"status": "no_candles", "model_id": model_id, "message": "No completed candle is available."}
        latest = ordered[-1]
        stamp = int(latest["time"])
        cached = self._preview_cache.get((model_id, stamp))
        if cached is not None:
            return dict(cached)
        feature_row, missing = self._features_for_latest(ordered)
        if feature_row is None and not missing:
            result = {"status": "features_unavailable", "model_id": model_id,
                      "message": "The latest completed candle does not have a unique feature row.", "missing_features": missing}
            return result
        if missing:
            return {"status": "insufficient_features", "model_id": model_id,
                    "message": "Feature warmup is incomplete for this candle.", "missing_features": missing,
                    "candle_time_utc": latest.get("candle_time_utc"), "reference_price": float(latest["close"])}
        model = self._model(model_id)
        x = pd.DataFrame([{name: float(feature_row[name]) for name in self.feature_order}],
                         columns=self.feature_order, dtype="float64")
        if list(x.columns) != self.feature_order:
            raise RuntimeError("Model preview feature order changed before inference")
        spec = MODEL_SPECS[model_id]
        probabilities: dict[str, float] | None = None
        predicted_return: float | None = None
        if spec["kind"] == "multiclass":
            raw = np.asarray(model.predict_proba(x), dtype=float)
            if raw.shape != (1, 3) or not np.isfinite(raw).all() or (raw < 0).any() or not np.isclose(raw.sum(), 1, atol=1e-6):
                raise RuntimeError("Multiclass model returned invalid probabilities")
            probabilities = dict(zip(CLASS_ORDER, raw[0].tolist()))
            prediction = CLASS_ORDER[int(np.argmax(raw[0]))]
        elif spec["kind"] == "regression":
            raw = np.asarray(model.predict(x), dtype=float).reshape(-1)
            if raw.size != 1 or not np.isfinite(raw).all():
                raise RuntimeError("Ridge model returned an invalid return estimate")
            predicted_return = float(raw[0])
            prediction = "UP" if predicted_return > 0 else "DOWN"
        else:
            raw = np.asarray(model.predict_proba(x), dtype=float)
            if raw.shape != (1, 2) or not np.isfinite(raw).all() or (raw < 0).any() or not np.isclose(raw.sum(), 1, atol=1e-6):
                raise RuntimeError("Binary model returned invalid probabilities")
            probabilities = {"DOWN": float(raw[0, 0]), "UP": float(raw[0, 1])}
            prediction = "UP" if raw[0, 1] > raw[0, 0] else "DOWN"
        generated = self.clock()
        if generated.tzinfo is None or generated.utcoffset() is None:
            raise RuntimeError("Preview clock must be timezone-aware")
        result = {
            "status": "preview", "model_id": model_id, "model_name": spec["name"],
            "kind": spec["kind"], "experimental": True,
            "prediction": prediction, "signal": {"UP": "BUY", "DOWN": "SELL", "NEUTRAL": "HOLD"}[prediction],
            "probabilities": probabilities, "predicted_return_percentage_points": predicted_return,
            "candle_time_utc": latest.get("candle_time_utc") or datetime.fromtimestamp(stamp, timezone.utc).isoformat().replace("+00:00", "Z"),
            "candle_time_epoch": stamp, "generated_at_utc": generated.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "reference_price": float(latest["close"]), "horizon_minutes": HORIZON_MINUTES,
            "persisted": False, "source": "unpersisted_research_model_preview",
            "model_sha256": spec["sha256"],
        }
        self._preview_cache[(model_id, stamp)] = result
        # Keep only the current candle's three small result objects in memory.
        self._preview_cache = {key: value for key, value in self._preview_cache.items() if key[1] == stamp}
        return dict(result)
