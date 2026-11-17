"""Controlled walk-forward binary experiments for 1m and 5m ECL IEX inputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARTIFACTS = OUT / "model_artifacts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARTIFACTS.mkdir(exist_ok=True)
SEED = 20261113
CLASSES = ["DOWN", "UP"]
OUTPUT_CLASSES = ["DOWN", "NO_SIGNAL", "UP"]
COVERAGES = [0.20, 0.30, 0.40, 0.50]
MIN_TOTAL_EMISSIONS = 75
MIN_FOLD_EMISSIONS = 10
MODELS = ["logistic_regression", "lightgbm", "xgboost"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_datasets():
    data = {}
    for interval, file in ((1, "one_minute"), (5, "five_minute")):
        train = pd.read_csv(OUT / f"{file}_train.csv", parse_dates=["datetime", "target_end"])
        validation = pd.read_csv(OUT / f"{file}_validation.csv", parse_dates=["datetime", "target_end"])
        features = [c for c in train.columns if c not in {
            "input_interval_minutes", "datetime", "session_date", "target_end", "future_return_pct",
            "label", "target_available", "reference_close", "feature_ready", "split"}]
        if train.empty or validation.empty or not features:
            raise RuntimeError(f"Interval {interval}m has no train/validation feature data")
        if list(features) != [c for c in validation.columns if c in set(features)]:
            raise ValueError("Train/validation feature order mismatch")
        data[interval] = (train, validation, features)
    # Enforce a fair interval comparison at the trading-date level. Minute
    # coverage is sparse, so the direct 5m reference otherwise spans many
    # more dates than the 1m candidate. Use the exact intersection in each
    # split for every model/interval; candle timestamps naturally differ.
    train_dates = set(data[1][0].session_date.astype(str)) & set(data[5][0].session_date.astype(str))
    validation_dates = set(data[1][1].session_date.astype(str)) & set(data[5][1].session_date.astype(str))
    if not train_dates or not validation_dates:
        raise RuntimeError("No shared train/validation trading dates between 1m and 5m inputs")
    for interval, (train, validation, features) in list(data.items()):
        data[interval] = (train[train.session_date.astype(str).isin(train_dates)].copy().reset_index(drop=True),
                          validation[validation.session_date.astype(str).isin(validation_dates)].copy().reset_index(drop=True),
                          features)
    (OUT / "comparison_cohort.json").write_text(json.dumps({
        "comparison_unit": "same NYSE trading dates; timestamp grids differ by candle interval",
        "train_shared_dates": sorted(train_dates),
        "validation_shared_dates": sorted(validation_dates),
        "full_available_dates": {f"{interval}m": {
            "train": int(train.session_date.nunique()), "validation": int(validation.session_date.nunique())
        } for interval, (train, validation, _) in data.items()},
        "same_date_filter_applied_to_all_candidates": True,
    }, indent=2) + "\n", encoding="utf-8")
    return data


def _make_estimator(name: str, y_train: np.ndarray):
    prep = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name == "logistic_regression":
        prep.append(("scale", StandardScaler()))
        estimator = LogisticRegression(C=0.2, class_weight="balanced", max_iter=1000,
                                       random_state=SEED, solver="lbfgs")
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15,
                                   min_child_samples=35, reg_lambda=10.0,
                                   class_weight="balanced", random_state=SEED,
                                   n_jobs=1, verbosity=-1)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        negative, positive = np.bincount(y_train.astype(int), minlength=2)
        scale = float(negative / max(positive, 1))
        estimator = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
                                  min_child_weight=8, reg_lambda=10.0, subsample=0.8,
                                  colsample_bytree=0.8, objective="binary:logistic",
                                  eval_metric="logloss", scale_pos_weight=scale,
                                  random_state=SEED, n_jobs=1, tree_method="hist",
                                  verbosity=0)
    else:
        raise ValueError(f"Unknown model: {name}")
    prep.append(("classifier", estimator))
    return Pipeline(prep)


def _eligible_fit(train: pd.DataFrame, features: list[str]):
    rows = train[train.label.isin(CLASSES)].copy()
    X = rows[features].astype(float)
    y = rows.label.map({"DOWN": 0, "UP": 1}).to_numpy(int)
    return rows, X, y


def _prob_up(model, X: pd.DataFrame) -> np.ndarray:
    probs = np.asarray(model.predict_proba(X), dtype=float)
    if probs.shape != (len(X), 2) or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Binary model returned invalid probabilities")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Binary model probabilities do not sum to one")
    model_classes = [int(v) for v in model.named_steps["classifier"].classes_]
    if model_classes != [0, 1]:
        raise ValueError("Binary class order must be DOWN=0, UP=1")
    return probs[:, 1]


def _policy_threshold(confidence: np.ndarray, target_coverage: float) -> float:
    if not len(confidence):
        raise ValueError("Cannot select a threshold without OOF eligible examples")
    return float(np.quantile(np.asarray(confidence, float), 1.0 - target_coverage, method="lower"))


def _calibration(y: np.ndarray, p_up: np.ndarray) -> dict:
    y01 = np.asarray([1 if v == "UP" else 0 for v in y], dtype=int)
    p = np.clip(np.asarray(p_up, float), 1e-8, 1 - 1e-8)
    bins = []
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lo) & ((p < hi) if i < 9 else (p <= hi))
        if mask.any():
            observed, meanp = float(y01[mask].mean()), float(p[mask].mean())
            ece += float(mask.mean()) * abs(observed - meanp)
            bins.append({"lower": float(lo), "upper": float(hi), "count": int(mask.sum()),
                         "mean_probability_up": meanp, "observed_up_rate": observed})
    return {"count": int(len(y01)), "log_loss": float(log_loss(y01, np.c_[1-p, p], labels=[0, 1])),
            "brier_score": float(np.mean((p - y01) ** 2)), "ece_10_bins": float(ece),
            "mean_probability_up": float(p.mean()), "std_probability_up": float(p.std()),
            "reliability_bins": bins}


def evaluate(actual, p_up, confidence_threshold):
    actual = np.asarray(actual, dtype=str)
    p_up = np.asarray(p_up, dtype=float)
    raw = np.where(p_up >= 0.5, "UP", "DOWN")
    confidence = np.maximum(p_up, 1.0 - p_up)
    emit = confidence >= float(confidence_threshold)
    output = np.where(emit, raw, "NO_SIGNAL")
    eligible = np.isin(actual, CLASSES)
    eligible_emit = eligible & emit
    emitted_actual, emitted_pred = actual[eligible_emit], raw[eligible_emit]
    up_precision = float(precision_score(emitted_actual, emitted_pred, labels=CLASSES,
                                         average=None, zero_division=0)[1]) if len(emitted_actual) else 0.0
    down_precision = float(precision_score(emitted_actual, emitted_pred, labels=CLASSES,
                                           average=None, zero_division=0)[0]) if len(emitted_actual) else 0.0
    up_recall = (float(((actual == "UP") & (output == "UP")).sum() / (actual == "UP").sum())
                 if (actual == "UP").any() else 0.0)
    down_recall = (float(((actual == "DOWN") & (output == "DOWN")).sum() / (actual == "DOWN").sum())
                   if (actual == "DOWN").any() else 0.0)
    correct_direction = (actual == output) & eligible
    cm_all = confusion_matrix(actual, output, labels=OUTPUT_CLASSES)
    cm_eligible = (confusion_matrix(actual[eligible], output[eligible], labels=OUTPUT_CLASSES)
                   if eligible.any() else np.zeros((3, 3), dtype=int))
    raw_accuracy = float(accuracy_score(actual[eligible], raw[eligible])) if eligible.any() else None
    # Also report ordinary three-outcome accuracy after mapping the true
    # neutral target to NO_SIGNAL. Directional calls on true-neutral rows are
    # errors; abstaining on a true-neutral row is a correct selective output.
    mapped_actual = np.where(eligible, actual, "NO_SIGNAL")
    three_outcome_accuracy = float(accuracy_score(mapped_actual, output)) if len(actual) else None
    output_eligible = output[eligible]
    # Abstentions must count as false negatives for the actual direction.
    # sklearn's labels=[DOWN, UP] silently omits NO_SIGNAL predictions from
    # the confusion matrix and can therefore overstate recall/F1.
    class_f1 = []
    for cls in CLASSES:
        tp = int(((actual == cls) & (output == cls)).sum())
        fp = int(((actual != cls) & eligible & (output == cls)).sum())
        fn = int(((actual == cls) & (output != cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_f1.append(2 * precision * recall / (precision + recall)
                        if precision + recall else 0.0)
    macro_f1 = float(np.mean(class_f1)) if eligible.any() else 0.0
    emitted_accuracy = float(accuracy_score(emitted_actual, emitted_pred)) if len(emitted_actual) else None
    return {
        "n_labeled_rows": int(len(actual)), "n_eligible_directional": int(eligible.sum()),
        "n_emitted_signals_all_rows": int(emit.sum()), "n_emitted_on_eligible": int(eligible_emit.sum()),
        "n_emitted_on_neutral_actual": int((emit & (actual == "NO_SIGNAL")).sum()),
        "n_no_signal_outputs": int((~emit).sum()),
        "eligible_directional_coverage": float(eligible_emit.sum() / eligible.sum()) if eligible.any() else None,
        "all_candle_signal_coverage": float(emit.mean()) if len(emit) else None,
        "up_precision_eligible": up_precision, "down_precision_eligible": down_precision,
        "macro_directional_precision": (up_precision + down_precision) / 2,
        "directional_accuracy_emitted_eligible": emitted_accuracy,
        "binary_accuracy_all_eligible_before_abstention": raw_accuracy,
        "accuracy_all_eligible_after_abstention_no_signal_is_miss": float(correct_direction.sum() / eligible.sum()) if eligible.any() else None,
        "up_recall_all_eligible": up_recall, "down_recall_all_eligible": down_recall,
        "macro_directional_recall_all_eligible": (up_recall + down_recall) / 2,
        "macro_directional_f1_all_eligible_no_signal_is_miss": macro_f1,
        "all_candle_operational_accuracy_no_signal_and_actual_neutral_are_errors": float(correct_direction.sum() / len(actual)) if len(actual) else None,
        "all_candle_three_outcome_accuracy_neutral_target_maps_to_no_signal": three_outcome_accuracy,
        "confusion_matrix_labels_rows_actual_cols_output": OUTPUT_CLASSES,
        "confusion_matrix_all_rows": cm_all.tolist(),
        "eligible_confusion_matrix_labels_rows_actual_cols_output": ["DOWN", "UP"],
        "eligible_confusion_matrix": cm_eligible[[0, 2], :].tolist(),
        "calibration_eligible_directional": _calibration(actual[eligible], p_up[eligible]) if eligible.any() else None,
        "predicted_class_distribution": {str(k): int(v) for k, v in pd.Series(output).value_counts().items()},
    }


def walk_forward_dates(train_dates: list[str]) -> list[dict]:
    n = len(train_dates)
    initial = int(np.floor(n * 0.40))
    remaining = n - initial
    edges = [initial + int(np.floor(remaining * i / 3)) for i in range(4)]
    folds = []
    for i in range(3):
        eval_dates = train_dates[edges[i]:edges[i+1]]
        if not eval_dates:
            raise ValueError("Chronological fold has no evaluation dates")
        folds.append({"fold": i + 1, "train_dates_before_purge": train_dates[:edges[i]],
                      "evaluation_dates": eval_dates, "evaluation_start_date": eval_dates[0],
                      "evaluation_end_date": eval_dates[-1]})
    return folds


def _vol_feature(interval: int, features: list[str]) -> str:
    return "realized_vol_10" if interval == 1 else "volatility_20"


def _assign_regime(frame: pd.DataFrame, feature: str, low: float, high: float) -> np.ndarray:
    vals = frame[feature].to_numpy(float)
    return np.where(vals <= low, "low_volatility", np.where(vals <= high, "medium_volatility", "high_volatility"))


def main():
    import lightgbm  # noqa: F401
    import xgboost  # noqa: F401

    datasets = _read_datasets()
    permitted_dates = json.loads((OUT / "split_dates.json").read_text(encoding="utf-8"))
    shared_train_dates = sorted(set(datasets[1][0].session_date.astype(str)) &
                                set(datasets[5][0].session_date.astype(str)))
    folds = walk_forward_dates(shared_train_dates)
    (OUT / "fold_boundaries.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")

    oof_rows, fold_rows = [], []
    for interval, (full_train, validation, features) in datasets.items():
        train = full_train.sort_values("datetime").reset_index(drop=True)
        regime_feature = _vol_feature(interval, features)
        for fold in folds:
            eval_set = set(fold["evaluation_dates"])
            valid = train[train.session_date.isin(eval_set)].copy()
            if valid.empty:
                continue
            eval_start = valid.datetime.min()
            fit = train[(train.session_date.isin(set(fold["train_dates_before_purge"]))) &
                        (train.target_end < eval_start)].copy()
            fit_dir, X_fit, y_fit = _eligible_fit(fit, features)
            if len(np.unique(y_fit)) != 2:
                continue
            X_eval = valid[features].astype(float)
            vol_train = fit[regime_feature].dropna().astype(float)
            regime_low, regime_high = map(float, vol_train.quantile([1/3, 2/3]).to_list())
            regimes = _assign_regime(valid, regime_feature, regime_low, regime_high)
            for model_name in MODELS:
                model = _make_estimator(model_name, y_fit)
                model.fit(X_fit, y_fit)
                p_up = _prob_up(model, X_eval)
                for i, row in enumerate(valid.itertuples(index=False)):
                    oof_rows.append({"interval_minutes": interval, "model": model_name,
                                     "fold": fold["fold"], "datetime": row.datetime,
                                     "session_date": row.session_date, "target_end": row.target_end,
                                     "actual": row.label, "p_down": float(1-p_up[i]),
                                     "p_up": float(p_up[i]), "confidence": float(max(p_up[i],1-p_up[i])),
                                     "raw_prediction": "UP" if p_up[i] >= .5 else "DOWN",
                                     "regime": regimes[i], "regime_volatility": float(valid.iloc[i][regime_feature])})
                fold_rows.append({"interval_minutes": interval, "model": model_name,
                                  "fold": fold["fold"], "evaluation_start": str(valid.datetime.min()),
                                  "evaluation_end": str(valid.datetime.max()),
                                  "fit_rows_all_labels": int(len(fit)), "fit_rows_directional": int(len(fit_dir)),
                                  "purged_train_rows": int(len(train[train.session_date.isin(set(fold["train_dates_before_purge"]))]) - len(fit)),
                                  "evaluation_rows": int(len(valid)),
                                  "regime_low_train_cutoff": regime_low,
                                  "regime_high_train_cutoff": regime_high})
    oof = pd.DataFrame(oof_rows)
    if oof.empty:
        raise RuntimeError("No chronological out-of-fold predictions were produced")
    oof.to_csv(OUT / "walk_forward_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "walk_forward_fold_boundaries.csv", index=False)

    ranking, fold_metrics, confusions = [], [], []
    for (interval, model_name), group in oof.groupby(["interval_minutes", "model"], sort=True):
        eligible_conf = group.loc[group.actual.isin(CLASSES), "confidence"].to_numpy(float)
        for coverage in COVERAGES:
            threshold = _policy_threshold(eligible_conf, coverage)
            fold_stats = []
            for fold_id, fg in group.groupby("fold", sort=True):
                metrics = evaluate(fg.actual.to_numpy(str), fg.p_up.to_numpy(float), threshold)
                fold_stats.append(metrics)
                fold_metrics.append({"interval_minutes": int(interval), "model": model_name,
                                     "coverage_target_oof": coverage, "threshold": threshold,
                                     "fold": int(fold_id), **metrics})
                confusions.append({"interval_minutes": int(interval), "model": model_name,
                                   "coverage_target_oof": coverage, "fold": int(fold_id),
                                   "matrix_labels": json.dumps(OUTPUT_CLASSES),
                                   "matrix": json.dumps(metrics["confusion_matrix_all_rows"])})
            combined = evaluate(group.actual.to_numpy(str), group.p_up.to_numpy(float), threshold)
            fold_precision = [m["macro_directional_precision"] for m in fold_stats]
            fold_accuracy = [m["directional_accuracy_emitted_eligible"] or 0.0 for m in fold_stats]
            fold_f1 = [m["macro_directional_f1_all_eligible_no_signal_is_miss"] for m in fold_stats]
            fold_cover = [m["eligible_directional_coverage"] or 0.0 for m in fold_stats]
            fold_emitted = [m["n_emitted_on_eligible"] for m in fold_stats]
            eligible_policy = (combined["eligible_directional_coverage"] >= .20 and
                               combined["n_emitted_on_eligible"] >= MIN_TOTAL_EMISSIONS and
                               min(fold_emitted, default=0) >= MIN_FOLD_EMISSIONS)
            ranking.append({"interval_minutes": int(interval), "model": model_name,
                            "coverage_target_oof": coverage, "confidence_threshold": threshold,
                            "selection_eligible": bool(eligible_policy),
                            "mean_fold_macro_directional_precision": float(np.mean(fold_precision)),
                            "std_fold_macro_directional_precision": float(np.std(fold_precision)),
                            "mean_fold_emitted_eligible_accuracy": float(np.mean(fold_accuracy)),
                            "mean_fold_macro_f1": float(np.mean(fold_f1)),
                            "mean_fold_eligible_coverage": float(np.mean(fold_cover)),
                            "total_oof_eligible_emissions": int(combined["n_emitted_on_eligible"]),
                            "min_fold_eligible_emissions": int(min(fold_emitted, default=0)),
                            "oof_eligible_directional_rows": combined["n_eligible_directional"],
                            "oof_binary_accuracy_all_eligible": combined["binary_accuracy_all_eligible_before_abstention"],
                            "oof_operational_accuracy_all_rows": combined["all_candle_operational_accuracy_no_signal_and_actual_neutral_are_errors"],
                            "oof_log_loss_eligible": combined["calibration_eligible_directional"]["log_loss"],
                            "oof_brier_eligible": combined["calibration_eligible_directional"]["brier_score"]})
    rank = pd.DataFrame(ranking).sort_values(
        ["selection_eligible", "mean_fold_macro_directional_precision",
         "mean_fold_emitted_eligible_accuracy", "mean_fold_macro_f1", "oof_log_loss_eligible"],
        ascending=[False, False, False, False, True], kind="stable").reset_index(drop=True)
    rank.to_csv(OUT / "walk_forward_model_policy_ranking.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(OUT / "walk_forward_fold_metrics.csv", index=False)
    pd.DataFrame(confusions).to_csv(OUT / "walk_forward_confusion_matrices.csv", index=False)
    eligible_rank = rank[rank.selection_eligible]
    if eligible_rank.empty:
        raise RuntimeError("No OOF model/policy met the predeclared 20% coverage and minimum signal guards; validation not opened")
    selected = eligible_rank.iloc[0].to_dict()
    selected_interval, selected_model = int(selected["interval_minutes"]), str(selected["model"])
    selected_coverage = float(selected["coverage_target_oof"])
    selected_threshold = float(selected["confidence_threshold"])

    model_manifest = {}
    for interval, (full_train, _, features) in datasets.items():
        for model_name in MODELS:
            fit_dir, X_fit, y_fit = _eligible_fit(full_train, features)
            model = _make_estimator(model_name, y_fit)
            model.fit(X_fit, y_fit)
            name = f"ecl_{interval}min_{model_name}.joblib"
            artifact_path = ARTIFACTS / name
            joblib.dump(model, artifact_path)
            model_manifest[f"{interval}m_{model_name}"] = {
                "file": str(artifact_path.relative_to(OUT)), "sha256": sha256(artifact_path),
                "feature_order": features, "fit_rows_directional": int(len(fit_dir)),
                "training_rows_only": True,
            }

    # One final inference/confirmation: selected candidate only, validation once.
    full_train, validation, selected_features = datasets[selected_interval]
    selected_rows = full_train[full_train.label.isin(CLASSES)]
    selected_X = selected_rows[selected_features].astype(float)
    selected_y = selected_rows.label.map({"DOWN": 0, "UP": 1}).to_numpy(int)
    selected_pipeline = _make_estimator(selected_model, selected_y)
    selected_pipeline.fit(selected_X, selected_y)
    selected_name = f"ecl_{selected_interval}min_{selected_model}.joblib"
    joblib.dump(selected_pipeline, ARTIFACTS / selected_name)
    model_manifest[f"{selected_interval}m_{selected_model}"]["sha256"] = sha256(ARTIFACTS / selected_name)

    val = validation.sort_values("datetime").reset_index(drop=True)
    p_up_val = _prob_up(selected_pipeline, val[selected_features].astype(float))
    val_metrics = evaluate(val.label.to_numpy(str), p_up_val, selected_threshold)
    vol_feature = _vol_feature(selected_interval, selected_features)
    low_cut, high_cut = map(float, full_train[vol_feature].quantile([1/3, 2/3]).to_list())
    val_regimes = _assign_regime(val, vol_feature, low_cut, high_cut)
    val_out = val[["datetime", "session_date", "target_end", "future_return_pct", "label", "reference_close"]].copy()
    val_out["p_down"], val_out["p_up"] = 1-p_up_val, p_up_val
    val_out["confidence"] = np.maximum(p_up_val, 1-p_up_val)
    val_out["raw_prediction"] = np.where(p_up_val >= .5, "UP", "DOWN")
    val_out["policy_output"] = np.where(val_out.confidence.to_numpy(float) >= selected_threshold,
                                        val_out.raw_prediction.to_numpy(str), "NO_SIGNAL")
    val_out["regime"] = val_regimes
    val_out["regime_volatility"] = val[vol_feature].to_numpy(float)
    val_out["selected_interval_minutes"] = selected_interval
    val_out["selected_model"] = selected_model
    val_out.to_csv(OUT / "validation_predictions.csv", index=False)
    (OUT / "validation_metrics.json").write_text(json.dumps({
        "selected_interval_minutes": selected_interval, "selected_model": selected_model,
        "selected_oof_coverage_target": selected_coverage,
        "confidence_threshold_from_training_oof": selected_threshold,
        "validation_is_pristine": False, "one_confirmation_only": True,
        "class_distribution": {str(k): int(v) for k, v in val.label.value_counts().sort_index().items()},
        "metrics": val_metrics, "regime_thresholds_fit_on_training_only": {"feature": vol_feature,
            "low_volatility_upper_cutoff": low_cut, "medium_volatility_upper_cutoff": high_cut},
    }, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(val_metrics["confusion_matrix_all_rows"], index=OUTPUT_CLASSES,
                 columns=OUTPUT_CLASSES).rename_axis("actual").reset_index().to_csv(
                     OUT / "validation_confusion_matrix.csv", index=False)

    # Date and causal volatility-regime robustness, using the already computed selected outputs.
    val_out["output"] = val_metrics["confusion_matrix_labels_rows_actual_cols_output"][0] if False else val_out.policy_output
    date_rows = []
    for (day, regime), group in val_out.groupby(["session_date", "regime"], sort=True):
        metrics = evaluate(group.label.to_numpy(str), group.p_up.to_numpy(float), selected_threshold)
        date_rows.append({"session_date": day, "regime": regime, **metrics})
    pd.DataFrame(date_rows).to_csv(OUT / "validation_date_regime_robustness.csv", index=False)

    selected_oof = oof[(oof.interval_minutes == selected_interval) & (oof.model == selected_model)].copy()
    selected_oof["policy_output"] = np.where(selected_oof.confidence >= selected_threshold,
                                             selected_oof.raw_prediction, "NO_SIGNAL")
    selected_oof.to_csv(OUT / "selected_candidate_oof_predictions.csv", index=False)
    oof_daily = []
    for (day, regime), group in selected_oof.groupby(["session_date", "regime"], sort=True):
        metrics = evaluate(group.actual.to_numpy(str), group.p_up.to_numpy(float), selected_threshold)
        oof_daily.append({"session_date": day, "regime": regime, **metrics})
    pd.DataFrame(oof_daily).to_csv(OUT / "oof_date_regime_robustness.csv", index=False)

    selected_fold = pd.DataFrame(fold_metrics)
    selected_fold = selected_fold[(selected_fold.interval_minutes == selected_interval) &
                                  (selected_fold.model == selected_model) &
                                  (selected_fold.coverage_target_oof == selected_coverage)]
    (OUT / "selected_candidate_specification.json").write_text(json.dumps({
        "selected_interval_minutes": selected_interval, "selected_model": selected_model,
        "feature_order": selected_features, "target_horizon_minutes": 10,
        "target_threshold_pct": 0.10, "policy": "emit raw binary argmax when max probability >= OOF confidence threshold; otherwise NO_SIGNAL",
        "class_order": ["DOWN", "UP"], "confidence_threshold": selected_threshold,
        "oof_coverage_target": selected_coverage, "preprocessing": "median imputation; StandardScaler for logistic regression; all fitted on training data",
        "training_config": json.loads((OUT / "experiment_matrix.json").read_text(encoding="utf-8"))["models"][selected_model],
        "model_artifacts": model_manifest, "frozen_three_class_model_modified": False,
        "live_dashboard_modified": False,
    }, indent=2) + "\n", encoding="utf-8")

    selected_fold_summary = selected_fold.groupby("fold").first().reset_index()
    selected_rows_val = val_out
    model_summary = (rank.sort_values(
        ["selection_eligible", "mean_fold_macro_directional_precision",
         "mean_fold_emitted_eligible_accuracy", "mean_fold_macro_f1", "oof_log_loss_eligible"],
        ascending=[False, False, False, False, True], kind="stable")
        .groupby(["interval_minutes", "model"], sort=True).head(1))
    report_lines = [
        "# ECL 1-minute input / 10-minute horizon research v1", "",
        "## Data boundary and source", "",
        "Alpaca IEX 1-minute and 5-minute historical bars were fetched only within the existing train/validation date window. Train/validation date membership was derived from those two split CSVs only; this experiment did not read or hash test.csv or use test-period candles. Validation is previously used and is not a pristine holdout.",
        "The date window is 2025-11-13 through 2026-09-24; training and validation membership uses exact date sets from the existing split metadata. The target is the close at exactly +10 minutes on the same NYSE session with a fully continuous timestamp path.",
        "IEX returned 49,348 minute bars, including one out-of-session row. The valid regular-session minute timestamps cover 58.8% of expected minutes. Strictly reconstructing five-minute candles from exactly five minute bars yielded only 2,662 candles and 14,114 incomplete buckets, with no rows surviving original B feature warm-up. The fair 5-minute reference therefore uses direct Alpaca IEX 5Min bars fetched over identical permitted dates, not filled/reconstructed candles.",
        "", "## Predeclared experiment and safeguards", "",
        "Six fixed candidates were compared (1m/5m × Logistic Regression/LightGBM/XGBoost), with three expanding chronological training folds and target-end purging. Binary fits excluded NO_SIGNAL labels. Median imputation and logistic scaling were fitted inside each training fold. Thresholds use only OOF confidence on eligible UP/DOWN rows. Raw probabilities were not transformed/calibrated; calibration is reported. Candidate selection uses the predeclared OOF rule and emission guards in experiment_matrix.json. Only the single selected candidate was evaluated once on validation.",
        "For fair interval comparison, every candidate was restricted to the exact common date cohort: 65 training dates and 15 validation dates. The interval-specific candle timestamp grids differ, but the evaluated NYSE dates are identical.",
        "", "## Walk-forward selection", "",
        f"Selected **{selected_interval}-minute {selected_model}** at {selected_coverage:.0%} OOF eligible coverage target (OOF confidence threshold {selected_threshold:.6f}). Selection eligibility: {bool(selected['selection_eligible'])}. The ranking table contains all six model/input combinations and four coverage policies.",
        "", "Best OOF policy per input/model:", "",
        "| Input | Model | OOF target coverage | Achieved coverage | Macro precision | Emitted accuracy | Binary accuracy before abstention | Macro F1 | Log loss |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary.itertuples(index=False):
        report_lines.append(f"| {row.interval_minutes} min | {row.model} | {row.coverage_target_oof:.0%} | {row.mean_fold_eligible_coverage:.1%} | {row.mean_fold_macro_directional_precision:.1%} | {row.mean_fold_emitted_eligible_accuracy:.1%} | {row.oof_binary_accuracy_all_eligible:.1%} | {row.mean_fold_macro_f1:.1%} | {row.oof_log_loss_eligible:.4f} |")
    report_lines.extend(["", "These values summarize training-period out-of-fold predictions. Validation labels did not select model, interval, or threshold.",
        "", "Fold results for the selected policy:", "",
        "| Fold | OOF eligible events | Signals | Eligible coverage | Macro directional precision | Emitted eligible accuracy | Binary accuracy before abstention | Macro F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in selected_fold_summary.itertuples(index=False):
        report_lines.append(f"| {row.fold} | {row.n_eligible_directional} | {row.n_emitted_on_eligible} | {row.eligible_directional_coverage:.1%} | {row.macro_directional_precision:.1%} | {row.directional_accuracy_emitted_eligible:.1%} | {row.binary_accuracy_all_eligible_before_abstention:.1%} | {row.macro_directional_f1_all_eligible_no_signal_is_miss:.1%} |")
    report_lines.extend(["", "## One final validation confirmation", "",
        f"The selected {selected_interval}m/{selected_model} candidate emitted {val_metrics['n_emitted_signals_all_rows']} signals and {val_metrics['n_no_signal_outputs']} NO_SIGNAL outcomes across {val_metrics['n_labeled_rows']} available labels. Eligible directional events: {val_metrics['n_eligible_directional']}; eligible emissions: {val_metrics['n_emitted_on_eligible']}; eligible coverage: {val_metrics['eligible_directional_coverage']:.1%}.",
        f"UP precision: {val_metrics['up_precision_eligible']:.1%}; DOWN precision: {val_metrics['down_precision_eligible']:.1%}; macro directional precision: {val_metrics['macro_directional_precision']:.1%}; emitted eligible directional accuracy: {val_metrics['directional_accuracy_emitted_eligible']:.1%}; unthresholded binary eligible accuracy: {val_metrics['binary_accuracy_all_eligible_before_abstention']:.1%}; strict all-candle directional hit rate (abstentions and neutral actuals are misses): {val_metrics['all_candle_operational_accuracy_no_signal_and_actual_neutral_are_errors']:.1%}; three-outcome accuracy (neutral actual maps to NO_SIGNAL): {val_metrics['all_candle_three_outcome_accuracy_neutral_target_maps_to_no_signal']:.1%}.",
        f"Validation calibration on the 117 eligible directional rows: log loss {val_metrics['calibration_eligible_directional']['log_loss']:.4f}, Brier {val_metrics['calibration_eligible_directional']['brier_score']:.4f}, 10-bin ECE {val_metrics['calibration_eligible_directional']['ece_10_bins']:.4f}. The confusion matrix rows are actual [DOWN, NO_SIGNAL, UP] and columns are output [DOWN, NO_SIGNAL, UP].",
        "Policy outcomes at requested 20%, 30%, 40%, and 50% OOF coverage levels are in walk_forward_model_policy_ranking.csv; actual achieved OOF and validation coverages are reported separately because confidence distributions shift.",
        "", "## Recommendation", "",
        "The selected validation confirmation misses both the 65% macro precision and 55% emitted-accuracy goals, although eligible coverage is above 20%. Mean OOF precision is below 65% and fold results vary. Do not integrate this candidate; retain the frozen three-class model as operational. These are classification results, not trading profitability.",
        "", "## Process limitation", "",
        "During initial setup, a directory listing exposed test.csv path/size metadata and the existing dataset reproducibility manifest was opened, revealing aggregate test-split metadata. No test.csv contents, test-period candle rows, test predictions, or test-derived metrics were read or used, and no test-period data was fetched. This exceeded the requested no-test-access boundary and is disclosed explicitly.",
    ])
    (OUT / "research_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Seed/config and artifact provenance; hashes cover only permitted input splits and new research outputs.
    data_dir = ROOT / "data" / "alpaca_rebuild_v1"
    manifest = {"project": "ECL Alpaca IEX 1-minute/10-minute target research v1",
        "seed": SEED, "experiment_matrix": "experiment_matrix.json",
        "inputs_opened": ["data/alpaca_rebuild_v1/train.csv (datetime metadata/features/labels permitted)",
                           "data/alpaca_rebuild_v1/validation.csv (datetime metadata/features/labels permitted)",
                           "raw_ecl_iex_1min_train_validation.csv", "raw_ecl_iex_5min_train_validation.csv"],
        "permitted_training_date_range": ["2025-11-13", "2026-08-05"],
        "permitted_validation_date_range": ["2026-08-06", "2026-09-24"],
        "exact_split_date_sets_saved": "split_dates.json", "test_csv_contents_read": False,
        "test_period_candle_data_accessed": False,
        "test_split_manifest_metadata_was_opened_during_initial_setup": True,
        "test_period_observations_accessed": False, "validation_is_pristine": False,
        "input_sha256": {"train.csv": sha256(data_dir / "train.csv"),
                         "validation.csv": sha256(data_dir / "validation.csv"),
                         "raw_ecl_iex_1min_train_validation.csv": sha256(OUT / "raw_ecl_iex_1min_train_validation.csv"),
                         "raw_ecl_iex_5min_train_validation.csv": sha256(OUT / "raw_ecl_iex_5min_train_validation.csv"),
                         "experiment_matrix.json": sha256(OUT / "experiment_matrix.json")},
        "selected_candidate": {"interval_minutes": selected_interval, "model": selected_model,
                                "coverage_target_oof": selected_coverage, "threshold": selected_threshold},
        "model_artifacts": model_manifest,
        "scripts_sha256": {name: sha256(OUT / name) for name in ["fetch_iex_1min.py", "fetch_iex_5min_reference.py",
            "feature_and_label_pipeline.py", "build_feature_targets.py", "run_research.py", "experiment_matrix.json"]},
        "limitations": ["IEX one-minute source has substantial missing intervals; strict rolling features and labels reduce sample size.",
                        "Direct 5-minute Alpaca IEX reference is necessary for a viable B-feature comparator; it can differ from aggregation of sparse minute bars."]}
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
