"""Isolated binary UP/DOWN ECL research: chronological OOF then one validation pass.

Only the approved train.csv and validation.csv are read. This module has no
reference to either test CSV or to any test predictions/metrics.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data/alpaca_ml_pipeline_v1"
MODEL_DIR = ROOT / "models/alpaca_binary_research_v3"
REPORT_DIR = ROOT / "reports/alpaca_binary_research_v3"
CONFIG_PATH = MODEL_DIR / "configuration.json"
CLASS_ORDER = ["DOWN", "UP"]
LABEL_TO_INT = {"DOWN": 0, "UP": 1}
NEUTRAL = "NEUTRAL"
SEED = 42
MAX_DATE = pd.Timestamp("2026-09-24")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binary_label_from_return(future_return: np.ndarray | pd.Series) -> np.ndarray:
    """Map strict existing thresholds; values on either boundary are neutral."""
    values = np.asarray(future_return, dtype=float)
    return np.where(values > 0.10, "UP", np.where(values < -0.10, "DOWN", NEUTRAL))


def load_dataset(split: str, feature_order: list[str]) -> pd.DataFrame:
    if split not in {"train", "validation"}:
        raise ValueError("Only train and validation splits are permitted")
    path = DATA_DIR / f"{split}.csv"
    frame = pd.read_csv(path, parse_dates=["datetime", "future_datetime"])
    expected = ["datetime", "future_datetime", "future_return", "target", *feature_order]
    if list(frame.columns) != expected:
        raise ValueError(f"{split} feature columns/order differ from the frozen contract")
    if frame.empty or frame.datetime.isna().any() or frame.future_datetime.isna().any():
        raise ValueError(f"{split} has missing decision/endpoint timestamps")
    if frame.datetime.duplicated().any() or not frame.datetime.is_monotonic_increasing:
        raise ValueError(f"{split} timestamps must be unique and increasing")
    if frame.datetime.max().normalize() > MAX_DATE or frame.future_datetime.max().normalize() > MAX_DATE:
        raise ValueError(f"{split} exceeds the permitted 2026-09-24 cutoff")
    if not ((frame.future_datetime - frame.datetime) == pd.Timedelta(minutes=15)).all():
        raise ValueError(f"{split} contains an endpoint other than exactly 15 minutes")
    expected_target = binary_label_from_return(frame.future_return)
    if not np.array_equal(expected_target, frame.target.to_numpy(str)):
        raise ValueError(f"{split} target labels do not match strict ±0.10% thresholds")
    if not set(frame.target.unique()).issubset({*CLASS_ORDER, NEUTRAL}):
        raise ValueError(f"{split} contains unexpected class labels")
    x = frame[feature_order].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"{split} features contain missing or non-finite values")
    if frame.target.isna().any():
        raise ValueError(f"{split} has an unlabeled row")
    return frame


def make_folds(train: pd.DataFrame, initial_fraction: float = 0.60, n_folds: int = 4) -> list[dict]:
    dates = np.array(sorted(train.datetime.dt.date.unique()), dtype=object)
    first_eval = int(np.floor(len(dates) * initial_fraction))
    remaining = dates[first_eval:]
    blocks = [b for b in np.array_split(remaining, n_folds) if len(b)]
    folds = []
    for fold_id, block in enumerate(blocks, start=1):
        first_date = block[0]
        validation_start = pd.Timestamp(first_date)
        val_mask = train.datetime.dt.date.isin(set(block))
        train_mask = train.datetime.dt.date < first_date
        # Purge any training example whose forward endpoint reaches the next block.
        train_mask &= train.future_datetime < validation_start
        tr = train.loc[train_mask]
        va = train.loc[val_mask]
        if tr.empty or va.empty:
            raise ValueError(f"Empty chronological fold {fold_id}")
        train_dates = sorted(tr.datetime.dt.date.unique())
        fold_dates = sorted(va.datetime.dt.date.unique())
        folds.append({
            "fold": fold_id,
            "train_index": tr.index.to_numpy(),
            "validation_index": va.index.to_numpy(),
            "train_start": str(train_dates[0]),
            "train_end": str(train_dates[-1]),
            "validation_start": str(fold_dates[0]),
            "validation_end": str(fold_dates[-1]),
            "purged_rows": int((train.datetime.dt.date < first_date).sum() - len(tr)),
        })
    if len(folds) != n_folds:
        raise ValueError("Not enough chronological dates to form all requested folds")
    return folds


def configurations() -> dict[str, list[dict]]:
    return {
        "logistic_regression": [
            {"name": "lr_c01_unweighted", "C": 0.1, "class_weight": None},
            {"name": "lr_c1_balanced", "C": 1.0, "class_weight": "balanced"},
        ],
        "random_forest": [
            {"name": "rf_compact_unweighted", "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 10, "class_weight": None},
            {"name": "rf_balanced", "n_estimators": 400, "max_depth": 12, "min_samples_leaf": 5, "class_weight": "balanced_subsample"},
        ],
        "xgboost": [
            {"name": "xgb_shallow_unweighted", "n_estimators": 250, "max_depth": 2, "learning_rate": 0.04, "weighted": False},
            {"name": "xgb_shallow_weighted", "n_estimators": 250, "max_depth": 2, "learning_rate": 0.04, "weighted": True},
        ],
        "hist_gradient_boosting": [
            {"name": "hgb_regular_unweighted", "max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 2.0, "weighted": False},
            {"name": "hgb_regular_weighted", "max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 2.0, "weighted": True},
        ],
    }


def make_estimator(family: str, config: dict):
    if family == "logistic_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=config["C"], class_weight=config["class_weight"], max_iter=2000, solver="lbfgs", random_state=SEED)),
        ])
    if family == "random_forest":
        return RandomForestClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"], min_samples_leaf=config["min_samples_leaf"], max_features="sqrt", class_weight=config["class_weight"], n_jobs=-1, random_state=SEED)
    if family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"], learning_rate=config["learning_rate"], min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=SEED)
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=config["max_iter"], learning_rate=config["learning_rate"], max_leaf_nodes=config["max_leaf_nodes"], min_samples_leaf=config["min_samples_leaf"], l2_regularization=config["l2_regularization"], random_state=SEED)
    raise ValueError(f"Unknown family: {family}")


def fit_estimator(family: str, config: dict, x: pd.DataFrame, y: np.ndarray):
    model = make_estimator(family, config)
    sample_weight = None
    if family == "xgboost":
        y_fit = np.asarray([LABEL_TO_INT[v] for v in y], dtype=int)
        if config.get("weighted"):
            sample_weight = compute_sample_weight("balanced", y_fit)
        model.fit(x, y_fit, sample_weight=sample_weight, verbose=False)
    elif family == "hist_gradient_boosting":
        if config.get("weighted"):
            sample_weight = compute_sample_weight("balanced", y)
        model.fit(x, y, sample_weight=sample_weight)
    else:
        model.fit(x, y)
    return model


def probabilities(model, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    out = np.zeros((len(x), 2), dtype=float)
    for j, cls in enumerate(model.classes_):
        label = CLASS_ORDER[int(cls)] if isinstance(cls, (int, np.integer)) else str(cls)
        out[:, CLASS_ORDER.index(label)] = raw[:, j]
    sums = out.sum(axis=1, keepdims=True)
    if not np.isfinite(out).all() or (out < 0).any() or (sums <= 0).any() or not np.allclose(sums, 1, atol=1e-6):
        raise ValueError("Model returned invalid binary probabilities")
    out /= sums
    return out


def binary_prediction_output(prob: np.ndarray, confidence_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Always emit UP or DOWN; warning is separate and never abstains."""
    p = np.asarray(prob, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("Expected a finite two-column DOWN/UP probability matrix")
    pred = np.asarray([CLASS_ORDER[i] for i in np.argmax(p, axis=1)], dtype=object)
    confidence = p.max(axis=1)
    low = confidence < float(confidence_threshold)
    if not set(pred).issubset(CLASS_ORDER):
        raise ValueError("Binary classifier emitted an invalid class")
    return pred, confidence, low


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray | None = None) -> dict:
    pp, rr, ff, support = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_ORDER, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    return {
        "n_directional": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "class_metrics": {c: {"precision": float(pp[i]), "recall": float(rr[i]), "f1": float(ff[i]), "support": int(support[i])} for i, c in enumerate(CLASS_ORDER)},
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "prediction_distribution": {c: int((y_pred == c).sum()) for c in CLASS_ORDER},
        "log_loss": float(log_loss(y_true, prob, labels=CLASS_ORDER)) if prob is not None else None,
    }


def baseline_predictions(name: str, frame: pd.DataFrame, majority: str) -> np.ndarray:
    if name == "always_up":
        return np.full(len(frame), "UP", dtype=object)
    if name == "always_down":
        return np.full(len(frame), "DOWN", dtype=object)
    if name == "training_majority":
        return np.full(len(frame), majority, dtype=object)
    if name == "rsi_50":
        return np.where(frame.rsi_14.to_numpy(float) >= 50.0, "UP", "DOWN")
    raise ValueError(name)


def evaluate_predictions(frame: pd.DataFrame, yhat_all: np.ndarray, p_all: np.ndarray, low_confidence_threshold: float) -> dict:
    actual3 = frame.target.to_numpy(str)
    directional = np.isin(actual3, CLASS_ORDER)
    y = actual3[directional]
    pred = yhat_all[directional]
    p = p_all[directional]
    result = binary_metrics(y, pred, p)
    result["all_eligible_rows"] = int(len(frame))
    result["neutral_rows_excluded_from_binary_metrics"] = int((actual3 == NEUTRAL).sum())
    result["directional_sample_coverage_of_all_eligible_rows"] = float(directional.mean())
    result["binary_prediction_coverage_of_all_rows"] = 1.0
    result["all_candle_exact_directional_accuracy_neutral_counts_wrong"] = float((yhat_all == actual3).mean())
    result["confidence_mean_all_rows"] = float(p_all.max(axis=1).mean())
    result["confidence_median_all_rows"] = float(np.median(p_all.max(axis=1)))
    result["low_confidence_threshold_from_training_oof"] = float(low_confidence_threshold)
    result["low_confidence_rows"] = int((p_all.max(axis=1) < low_confidence_threshold).sum())
    result["low_confidence_fraction_all_rows"] = float((p_all.max(axis=1) < low_confidence_threshold).mean())
    result["low_confidence_flag_does_not_change_binary_prediction"] = True
    return result


def _fold_config_metric(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    return binary_metrics(y, pred, prob)


def _select_family_config(summary: pd.DataFrame, family: str) -> str:
    candidates = summary.loc[summary.family.eq(family)].copy()
    guarded = candidates.loc[(candidates.mean_balanced_accuracy >= 0.50) & (candidates.mean_up_recall >= 0.40) & (candidates.mean_down_recall >= 0.40)]
    pool = guarded if not guarded.empty else candidates
    if not guarded.empty:
        pool = pool.sort_values(["mean_accuracy", "mean_balanced_accuracy", "mean_macro_f1", "mean_log_loss"], ascending=[False, False, False, True])
    else:
        pool = pool.sort_values(["mean_balanced_accuracy", "mean_accuracy", "mean_macro_f1", "mean_log_loss"], ascending=[False, False, False, True])
    return str(pool.iloc[0].configuration)


def _oof_confidence_threshold(oof: pd.DataFrame, family: str, config_name: str) -> float:
    q = oof.loc[oof.family.eq(family) & oof.configuration.eq(config_name), "confidence"].to_numpy(float)
    if not len(q):
        raise ValueError("No training-only OOF confidence values for selected configuration")
    return float(np.quantile(q, 0.25))


def main() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # XGBoost is mandatory; fail clearly before fitting any candidate.
    import xgboost

    dataset_manifest_path = DATA_DIR / "reproducibility_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    feature_order = list(dataset_manifest["features"])
    if len(feature_order) != 24:
        raise ValueError("This experiment requires the exact 24-feature frozen B schema")
    train = load_dataset("train", feature_order)
    validation = load_dataset("validation", feature_order)
    if validation.datetime.min() <= train.datetime.max():
        raise ValueError("Train and validation decision timestamps overlap or are out of order")
    y_train_all = train.target.to_numpy(str)
    directional_train = np.isin(y_train_all, CLASS_ORDER)
    train_directional = train.loc[directional_train].copy()
    train_y = train_directional.target.to_numpy(str)
    if set(np.unique(train_y)) != set(CLASS_ORDER):
        raise ValueError("Both UP and DOWN must appear in binary training data")

    config_map = configurations()
    folds = make_folds(train)
    fold_records, oof_records, baseline_records = [], [], []
    config_fold_metrics: dict[tuple[str, str], list[dict]] = {}

    for fold in folds:
        fold_train_all = train.loc[fold["train_index"]]
        fold_val_all = train.loc[fold["validation_index"]]
        fold_train = fold_train_all.loc[fold_train_all.target.isin(CLASS_ORDER)]
        y_fit = fold_train.target.to_numpy(str)
        y_val_all = fold_val_all.target.to_numpy(str)
        val_directional = np.isin(y_val_all, CLASS_ORDER)
        y_val = y_val_all[val_directional]
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_val)) < 2:
            raise ValueError(f"Fold {fold['fold']} lacks both binary classes")
        majority = pd.Series(y_fit).value_counts().idxmax()
        for base in ["always_up", "always_down", "training_majority", "rsi_50"]:
            bp = baseline_predictions(base, fold_val_all, majority)
            bm = binary_metrics(y_val, bp[val_directional])
            baseline_records.append({"fold": fold["fold"], "baseline": base, **bm})
        fold_records.append({k: fold[k] for k in ["fold", "train_start", "train_end", "validation_start", "validation_end", "purged_rows"]} | {
            "train_rows_all_classes": int(len(fold_train_all)), "train_rows_directional": int(len(fold_train)),
            "train_down": int((fold_train.target == "DOWN").sum()), "train_up": int((fold_train.target == "UP").sum()),
            "validation_rows_all_classes": int(len(fold_val_all)), "validation_rows_directional": int(val_directional.sum()),
            "validation_neutral": int((y_val_all == NEUTRAL).sum()), "validation_down": int((y_val == "DOWN").sum()), "validation_up": int((y_val == "UP").sum()),
        })
        for family, family_configs in config_map.items():
            for cfg in family_configs:
                started = time.perf_counter()
                model = fit_estimator(family, cfg, fold_train[feature_order], y_fit)
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                p_all = probabilities(model, fold_val_all[feature_order])
                predict_seconds = time.perf_counter() - started
                pred_all, _, _ = binary_prediction_output(p_all, 0.0)
                metrics = _fold_config_metric(y_val, pred_all[val_directional], p_all[val_directional])
                config_fold_metrics.setdefault((family, cfg["name"]), []).append(metrics | {"fold": fold["fold"], "family": family, "configuration": cfg["name"], "training_seconds": fit_seconds, "prediction_seconds": predict_seconds})
                for i, (_, row) in enumerate(fold_val_all.iterrows()):
                    oof_records.append({
                        "fold": fold["fold"], "family": family, "configuration": cfg["name"],
                        "datetime": row.datetime, "actual_three_class": row.target,
                        "actual_binary": row.target if row.target in CLASS_ORDER else "",
                        "prediction": pred_all[i], "p_down": p_all[i, 0], "p_up": p_all[i, 1],
                        "confidence": float(p_all[i].max()), "low_confidence_oof_q25": False,
                    })

    oof = pd.DataFrame(oof_records)
    config_rows = []
    for (family, cfg_name), fold_metrics in config_fold_metrics.items():
        config_rows.append({
            "family": family, "configuration": cfg_name, "folds": len(fold_metrics),
            "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])), "std_accuracy": float(np.std([m["accuracy"] for m in fold_metrics], ddof=0)),
            "folds_accuracy_gt_55pct": int(sum(m["accuracy"] > 0.55 for m in fold_metrics)),
            "mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in fold_metrics])),
            "mean_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
            "mean_up_precision": float(np.mean([m["class_metrics"]["UP"]["precision"] for m in fold_metrics])),
            "mean_up_recall": float(np.mean([m["class_metrics"]["UP"]["recall"] for m in fold_metrics])),
            "mean_down_precision": float(np.mean([m["class_metrics"]["DOWN"]["precision"] for m in fold_metrics])),
            "mean_down_recall": float(np.mean([m["class_metrics"]["DOWN"]["recall"] for m in fold_metrics])),
            "mean_log_loss": float(np.mean([m["log_loss"] for m in fold_metrics])),
            "mean_training_seconds": float(np.mean([m["training_seconds"] for m in fold_metrics])),
            "mean_prediction_seconds": float(np.mean([m["prediction_seconds"] for m in fold_metrics])),
        })
    config_summary = pd.DataFrame(config_rows)
    family_config_selection = {family: _select_family_config(config_summary, family) for family in config_map}

    # Fit one walk-forward-selected configuration per family on all training dates.
    train_majority = str(pd.Series(train_y).value_counts().idxmax())
    val_y_all = validation.target.to_numpy(str)
    val_directional = np.isin(val_y_all, CLASS_ORDER)
    val_y = val_y_all[val_directional]
    validation_predictions = pd.DataFrame({"datetime": validation.datetime, "actual_three_class": val_y_all, "actual_binary": np.where(val_y_all == NEUTRAL, "", val_y_all), "future_return": validation.future_return})
    validation_scores: dict[str, dict] = {}
    validation_matrices = []
    validation_timing = {}
    trained_models = {}
    threshold_by_family = {}
    selected_config_by_family = {}
    for family, cfg_name in family_config_selection.items():
        cfg = next(c for c in config_map[family] if c["name"] == cfg_name)
        fit_started = time.perf_counter()
        model = fit_estimator(family, cfg, train_directional[feature_order], train_y)
        final_fit_seconds = time.perf_counter() - fit_started
        trained_models[family] = model
        predict_started = time.perf_counter()
        p_all = probabilities(model, validation[feature_order])
        validation_predict_seconds = time.perf_counter() - predict_started
        validation_timing[family] = {"final_train_fit_seconds": final_fit_seconds, "validation_inference_seconds_all_rows": validation_predict_seconds}
        threshold = _oof_confidence_threshold(oof, family, cfg_name)
        threshold_by_family[family] = threshold
        pred_all, confidence_all, low_confidence_all = binary_prediction_output(p_all, threshold)
        selected_config_by_family[family] = cfg
        score = evaluate_predictions(validation, pred_all, p_all, threshold)
        score["selected_training_oof_configuration"] = cfg
        score.update(validation_timing[family])
        validation_scores[family] = score
        validation_predictions[f"{family}_prediction"] = pred_all
        validation_predictions[f"{family}_p_down"] = p_all[:, 0]
        validation_predictions[f"{family}_p_up"] = p_all[:, 1]
        validation_predictions[f"{family}_confidence"] = confidence_all
        validation_predictions[f"{family}_low_confidence"] = low_confidence_all
        yhat_directional = pred_all[val_directional]
        cm = confusion_matrix(val_y, yhat_directional, labels=CLASS_ORDER)
        for r, actual in enumerate(CLASS_ORDER):
            for c, predicted in enumerate(CLASS_ORDER):
                validation_matrices.append({"family": family, "actual": actual, "predicted": predicted, "count": int(cm[r, c])})

    # Fixed, non-fitted validation baselines. Majority direction is from training only.
    val_baselines = {}
    for base in ["always_up", "always_down", "training_majority", "rsi_50"]:
        pred = baseline_predictions(base, validation, train_majority)
        val_baselines[base] = binary_metrics(val_y, pred[val_directional])
        validation_predictions[f"baseline_{base}_prediction"] = pred
        cm = confusion_matrix(val_y, pred[val_directional], labels=CLASS_ORDER)
        for r, actual in enumerate(CLASS_ORDER):
            for c, predicted in enumerate(CLASS_ORDER):
                validation_matrices.append({"family": f"baseline_{base}", "actual": actual, "predicted": predicted, "count": int(cm[r, c])})

    # A final candidate must clear the predeclared validation guardrails; never
    # pick solely by accuracy. If none pass, the report records no qualified model.
    qualified = []
    for family, score in validation_scores.items():
        if score["accuracy"] >= 0.55 and score["balanced_accuracy"] >= 0.50 and score["class_metrics"]["UP"]["recall"] >= 0.40 and score["class_metrics"]["DOWN"]["recall"] >= 0.40:
            qualified.append((family, score))
    if qualified:
        qualified.sort(key=lambda it: (-it[1]["macro_f1"], -it[1]["balanced_accuracy"], -it[1]["accuracy"], it[1]["log_loss"]))
        selected_family = qualified[0][0]
        selection_status = "qualified_under_predeclared_validation_guardrails"
    else:
        selected_family = None
        selection_status = "no_candidate_met_validation_accuracy_balance_and_class_recall_guardrails"

    # OOF confidence quantile defines the low-confidence warning only; low
    # confidence does not abstain or alter the binary prediction.
    selected_threshold = threshold_by_family[selected_family] if selected_family else None
    if selected_family:
        low = oof.family.eq(selected_family) & oof.configuration.eq(family_config_selection[selected_family])
        oof.loc[low, "low_confidence_oof_q25"] = oof.loc[low, "confidence"] < selected_threshold

    # Save research artifacts only within this new, isolated v3 directory.
    for family, model in trained_models.items():
        joblib.dump(model, MODEL_DIR / f"{family}.joblib")
    if selected_family:
        joblib.dump(trained_models[selected_family], MODEL_DIR / "selected_binary_model.joblib")
    oof.to_csv(REPORT_DIR / "walk_forward_oof_predictions.csv", index=False)
    config_summary.to_csv(REPORT_DIR / "walk_forward_config_comparison.csv", index=False)
    pd.DataFrame(fold_records).to_csv(REPORT_DIR / "walk_forward_folds.csv", index=False)
    pd.DataFrame(baseline_records).to_csv(REPORT_DIR / "walk_forward_baselines.csv", index=False)
    walk_forward_cm_rows = []
    for (family, cfg_name), fold_metrics in config_fold_metrics.items():
        for m in fold_metrics:
            for r, actual in enumerate(CLASS_ORDER):
                for c, predicted in enumerate(CLASS_ORDER):
                    walk_forward_cm_rows.append({"family": family, "configuration": cfg_name, "fold": m["fold"], "actual": actual, "predicted": predicted, "count": m["confusion_matrix_rows_actual_columns_predicted"][r][c]})
    pd.DataFrame(walk_forward_cm_rows).to_csv(REPORT_DIR / "walk_forward_confusion_matrices.csv", index=False)
    validation_predictions.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)
    pd.DataFrame(validation_matrices).to_csv(REPORT_DIR / "validation_confusion_matrices.csv", index=False)

    # Validation metrics and baseline summary. Directional metrics are strictly
    # on UP/DOWN actual labels; all-row binary exact accuracy is separately shown.
    validation_payload = {
        "selected_family": selected_family,
        "selection_status": selection_status,
        "selected_family_config": selected_config_by_family.get(selected_family) if selected_family else None,
        "training_oof_confidence_warning_threshold": selected_threshold,
        "models": validation_scores,
        "baselines": val_baselines,
        "validation_rows_all_classes": int(len(validation)),
        "validation_rows_directional": int(val_directional.sum()),
        "validation_rows_neutral": int((val_y_all == NEUTRAL).sum()),
        "validation_directional_sample_coverage": float(val_directional.mean()),
        "validation_date_range": [str(validation.datetime.min().date()), str(validation.datetime.max().date())],
        "validation_used_once_after_training_oof_selection": True,
    }
    (MODEL_DIR / "validation_metrics.json").write_text(json.dumps(validation_payload, indent=2), encoding="utf-8")
    pd.DataFrame([{"family": k, **{kk: vv for kk, vv in v.items() if kk in ["n_directional", "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "directional_sample_coverage_of_all_eligible_rows", "all_candle_exact_directional_accuracy_neutral_counts_wrong", "low_confidence_threshold_from_training_oof", "low_confidence_fraction_all_rows", "final_train_fit_seconds", "validation_inference_seconds_all_rows"]}} for k, v in validation_scores.items()]).to_csv(REPORT_DIR / "validation_model_comparison.csv", index=False)
    pd.DataFrame([{"baseline": k, **v} for k, v in val_baselines.items()]).to_csv(REPORT_DIR / "validation_baseline_comparison.csv", index=False)

    # Confidence group analysis uses OOF-only cut points, then reports the
    # already-planned single validation confirmation by those fixed groups.
    confidence_analysis = {}
    if selected_family:
        conf_oof = oof.loc[oof.family.eq(selected_family) & oof.configuration.eq(family_config_selection[selected_family]), "confidence"].to_numpy(float)
        edges = np.unique(np.quantile(conf_oof, [0, .25, .5, .75, 1]))
        if len(edges) >= 3:
            vp = validation_predictions
            c = vp[f"{selected_family}_confidence"].to_numpy(float)
            groups = np.searchsorted(edges[1:-1], c, side="right")
            vp_dir = np.isin(vp.actual_three_class.to_numpy(str), CLASS_ORDER)
            for g in range(len(edges) - 1):
                rows = groups == g
                keep = rows & vp_dir
                confidence_analysis[f"OOF_confidence_group_{g+1}"] = {
                    "validation_all_rows": int(rows.sum()),
                    "validation_directional_rows": int(keep.sum()),
                    "directional_accuracy": float((vp.loc[keep, "actual_binary"].to_numpy(str) == vp.loc[keep, f"{selected_family}_prediction"].to_numpy(str)).mean()) if keep.any() else None,
                    "confidence_lower_bound_oof": float(edges[g]),
                    "confidence_upper_bound_oof": float(edges[g + 1]),
                }
    validation_payload["confidence_group_analysis_oof_cutpoints"] = confidence_analysis
    (MODEL_DIR / "validation_metrics.json").write_text(json.dumps(validation_payload, indent=2), encoding="utf-8")
    pd.DataFrame([{"confidence_group": name, **values} for name, values in confidence_analysis.items()]).to_csv(REPORT_DIR / "validation_confidence_groups.csv", index=False)

    # Data audit, manifests, schema, and result text.
    audit = {
        "train_rows_all": len(train), "train_rows_directional": int(directional_train.sum()), "train_neutral_excluded": int((train.target == NEUTRAL).sum()),
        "train_class_counts_directional": train_directional.target.value_counts().reindex(CLASS_ORDER, fill_value=0).to_dict(),
        "train_directional_coverage": float(directional_train.mean()),
        "validation_rows_all": len(validation), "validation_rows_directional": int(val_directional.sum()), "validation_neutral_excluded_from_binary_metrics": int((validation.target == NEUTRAL).sum()),
        "validation_class_counts_directional": validation.loc[val_directional, "target"].value_counts().reindex(CLASS_ORDER, fill_value=0).to_dict(),
        "validation_directional_coverage": float(val_directional.mean()),
        "feature_count": len(feature_order), "feature_order": feature_order,
        "train_date_range": [str(train.datetime.min().date()), str(train.datetime.max().date())],
        "validation_date_range": [str(validation.datetime.min().date()), str(validation.datetime.max().date())],
        "max_date_checked": str(max(train.datetime.max(), validation.datetime.max()).date()),
        "nonfinite_features": 0, "duplicate_decision_timestamps": 0, "invalid_horizon_rows": 0,
        "target_definition_verified_against_future_return": True,
        "protected_test_files_opened": False,
    }
    (MODEL_DIR / "dataset_label_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    schema = {"features": feature_order, "feature_count": len(feature_order), "class_order": CLASS_ORDER, "horizon_minutes": 15, "threshold_pct": 0.10, "neutral_training_policy": "exclude", "inference_policy": "always predict UP or DOWN; attach low-confidence warning"}
    (MODEL_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = {
        "experiment": "alpaca_binary_research_v3", "source_manifest_sha256": sha256(dataset_manifest_path),
        "train_csv_sha256": sha256(DATA_DIR / "train.csv"), "validation_csv_sha256": sha256(DATA_DIR / "validation.csv"),
        "test_csv_accessed": False, "post_2026_09_24_data_accessed": False,
        "feature_order": feature_order, "class_order": CLASS_ORDER, "seed": SEED,
        "xgboost_version": xgboost.__version__, "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        "fold_boundaries": fold_records, "family_selected_configs_by_training_oof": family_config_selection,
        "candidate_model_hashes": {family: sha256(MODEL_DIR / f"{family}.joblib") for family in trained_models},
        "selected_family": selected_family, "selection_status": selection_status,
        "validation_evaluation_count": 1, "validation_used_for_training_or_tuning": False,
        "configuration": config,
    }
    if selected_family:
        manifest["selected_model_sha256"] = sha256(MODEL_DIR / "selected_binary_model.joblib")
    (MODEL_DIR / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_lines = [
        "# ECL Alpaca IEX binary UP/DOWN research v3", "",
        "This isolated experiment uses only the rebuilt Alpaca IEX train and validation splits through 2026-09-24. NEUTRAL rows are excluded from binary fitting and directional metrics; all-candle exact accuracy is reported separately with actual neutral rows counted as errors. No test CSV or test prediction/report was opened.",
        "The existing `models/alpaca_binary_research_v1/` already contained a prior study, so this run is saved under v3 to preserve it.", "",
        f"Training directional rows: {audit['train_rows_directional']:,} of {audit['train_rows_all']:,}; NEUTRAL excluded {audit['train_neutral_excluded']:,}; coverage {audit['train_directional_coverage']:.1%}; DOWN/UP counts {audit['train_class_counts_directional']['DOWN']}/{audit['train_class_counts_directional']['UP']}.",
        f"Validation directional rows: {audit['validation_rows_directional']:,} of {audit['validation_rows_all']:,}; NEUTRAL excluded {audit['validation_neutral_excluded_from_binary_metrics']:,}; coverage {audit['validation_directional_coverage']:.1%}; DOWN/UP counts {audit['validation_class_counts_directional']['DOWN']}/{audit['validation_class_counts_directional']['UP']}.",
        "", "## Walk-forward results", "", "Per-configuration fold metrics are in `reports/alpaca_binary_research_v3/walk_forward_config_comparison.csv`; individual folds and their dates/counts are in `walk_forward_folds.csv`. Settings were selected without validation labels.", "",
        "| Family | Config selected by training OOF | Mean accuracy | Fold SD | Folds >55% | Mean balanced accuracy | Mean macro-F1 | Mean UP recall | Mean DOWN recall | Mean log loss |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, cfg_name in family_config_selection.items():
        r = config_summary.loc[(config_summary.family == family) & (config_summary.configuration == cfg_name)].iloc[0]
        report_lines.append(f"| {family} | {cfg_name} | {r.mean_accuracy:.3f} | {r.std_accuracy:.3f} | {int(r.folds_accuracy_gt_55pct)}/4 | {r.mean_balanced_accuracy:.3f} | {r.mean_macro_f1:.3f} | {r.mean_up_recall:.3f} | {r.mean_down_recall:.3f} | {r.mean_log_loss:.3f} |")
    report_lines += ["", "## One-time validation confirmation", "", "The validation period has been reused in prior project research and is not pristine. Each family’s OOF-selected configuration was evaluated on it once; no tuning followed.", "", "| Family | Directional accuracy | Balanced accuracy | Macro-F1 | UP P/R/F1 | DOWN P/R/F1 | Directional coverage of all rows | All-candle exact accuracy (neutral wrong) | Log loss | Low-confidence fraction |", "|---|---:|---:|---:|---|---|---:|---:|---:|---:|"]
    for family, score in validation_scores.items():
        u, d = score["class_metrics"]["UP"], score["class_metrics"]["DOWN"]
        report_lines.append(f"| {family} | {score['accuracy']:.3f} | {score['balanced_accuracy']:.3f} | {score['macro_f1']:.3f} | {u['precision']:.3f}/{u['recall']:.3f}/{u['f1']:.3f} | {d['precision']:.3f}/{d['recall']:.3f}/{d['f1']:.3f} | {score['directional_sample_coverage_of_all_eligible_rows']:.3f} | {score['all_candle_exact_directional_accuracy_neutral_counts_wrong']:.3f} | {score['log_loss']:.3f} | {score['low_confidence_fraction_all_rows']:.3f} |")
    report_lines += ["", "### Validation baselines (directional rows only)", "", "| Baseline | Accuracy | Balanced accuracy | Macro-F1 | UP recall | DOWN recall |", "|---|---:|---:|---:|---:|---:|"]
    for name, score in val_baselines.items():
        report_lines.append(f"| {name} | {score['accuracy']:.3f} | {score['balanced_accuracy']:.3f} | {score['macro_f1']:.3f} | {score['class_metrics']['UP']['recall']:.3f} | {score['class_metrics']['DOWN']['recall']:.3f} |")
    report_lines += ["", f"**Validation decision:** {selection_status}. Selected model: {selected_family or 'none'}. The validation guardrails require at least 55% directional accuracy, 50% balanced accuracy, and at least 40% recall for each direction. The result is not a pristine final test and does not support a live promotion.", "", "## Confidence and limitations", "", f"Low-confidence threshold is the selected model's training-only OOF 25th percentile of max(P(UP), P(DOWN)): {selected_threshold if selected_threshold is not None else 'not applicable; no qualifying model'}. It only sets a warning flag; output remains UP or DOWN. OOF-derived confidence boundaries and validation accuracy-by-group are recorded in `validation_metrics.json`.", "", "The prior processed test period was not opened or used. Validation dates have been repeatedly used in earlier work. This experiment does not establish genuine independent future performance, trading profitability, or suitability for production. A later post-freeze Alpaca IEX period is required for final confirmation. No live integration or model promotion occurred.", ""]
    (REPORT_DIR / "research_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {"folds": fold_records, "config_summary": config_summary.to_dict(orient="records"), "selected_configs": family_config_selection, "validation": validation_payload, "audit": audit}


if __name__ == "__main__":
    # Check mandatory XGBoost availability before any model fitting.
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise SystemExit("XGBoost is required for this experiment and is not installed; no substitute was used.") from exc
    results = main()
    print(json.dumps({"selected_configs": results["selected_configs"], "validation_selection": results["validation"]["selection_status"], "selected_family": results["validation"]["selected_family"], "dataset_audit": results["audit"]}, indent=2))
