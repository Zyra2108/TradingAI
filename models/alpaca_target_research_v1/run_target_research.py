"""Chronological target/horizon research using only permitted train/validation files."""
from __future__ import annotations

import hashlib
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from target_generation import (CLASSES, DATA, DIRECTIONS, FEATURES, generate_target,
                               load_allowed_split, matrix)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 20261115
COVERAGES = (0.20, 0.30, 0.40, 0.50)
MODELS = ("logistic_regression", "lightgbm", "xgboost")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def folds_for(frame: pd.DataFrame, count: int = 3):
    dates = np.array(sorted(frame.datetime.dt.normalize().unique()))
    initial = max(1, int(np.floor(.40 * len(dates))))
    chunks = [a for a in np.array_split(dates[initial:], count) if len(a)]
    if len(chunks) != count:
        raise ValueError("Insufficient dates for three expanding folds")
    return [(pd.Timestamp(c[0]), pd.Timestamp(c[-1]) + pd.Timedelta(days=1)) for c in chunks]


def preprocessor(model_name: str):
    numeric = [f for f in FEATURES if f != "day_of_week"]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "logistic_regression":
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(num_steps), numeric),
        ("weekday", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["day_of_week"]),
    ], remainder="drop", verbose_feature_names_out=False)


def estimator(model_name: str, y: np.ndarray):
    down, up = int((y == 0).sum()), int((y == 1).sum())
    if model_name == "logistic_regression":
        return LogisticRegression(C=.2, max_iter=500, class_weight="balanced", random_state=SEED)
    if model_name == "lightgbm":
        return lgb.LGBMClassifier(objective="binary", n_estimators=180, num_leaves=7,
            learning_rate=.03, min_child_samples=35, subsample=.8, subsample_freq=1,
            colsample_bytree=.8, reg_lambda=10., class_weight="balanced", n_jobs=4,
            random_state=SEED, verbosity=-1)
    if model_name == "xgboost":
        return xgb.XGBClassifier(n_estimators=180, max_depth=2, learning_rate=.03,
            min_child_weight=8, subsample=.8, colsample_bytree=.8, reg_lambda=10.,
            scale_pos_weight=down / max(up, 1), objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=4, random_state=SEED)
    raise ValueError(model_name)


def fit_model(model_name: str, frame: pd.DataFrame):
    y = frame.label.map({"DOWN": 0, "UP": 1}).to_numpy(int)
    pipe = Pipeline([("preprocessor", preprocessor(model_name)),
                     ("model", estimator(model_name, y))])
    pipe.fit(frame.loc[:, FEATURES], y)
    if list(pipe.named_steps["preprocessor"].feature_names_in_) != FEATURES:
        raise AssertionError("Model input order changed")
    return pipe


def threshold_at_coverage(conf: np.ndarray, eligible: np.ndarray, coverage: float) -> float:
    scores = np.asarray(conf, float)[np.asarray(eligible, bool)]
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("No finite eligible OOF confidence scores")
    k = max(1, int(np.ceil(coverage * len(scores))))
    return float(np.sort(scores)[-k])


def calibration(y: np.ndarray, p_up: np.ndarray) -> dict:
    yi = np.asarray([{"DOWN": 0, "UP": 1}[v] for v in y], dtype=int)
    p = np.clip(np.asarray(p_up, float), 1e-7, 1-1e-7)
    reliability, ece = [], 0.
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        mask = (p >= edges[i]) & ((p <= edges[i+1]) if i == 9 else (p < edges[i+1]))
        if mask.any():
            mp, rate = float(p[mask].mean()), float(yi[mask].mean())
            ece += float(mask.mean()) * abs(mp-rate)
            reliability.append({"bin": i, "n": int(mask.sum()), "mean_p_up": mp,
                                "observed_up_rate": rate, "lower": float(edges[i]),
                                "upper": float(edges[i+1])})
    return {"n": int(len(y)), "log_loss": float(log_loss(yi, np.c_[1-p,p], labels=[0,1])),
            "brier": float(brier_score_loss(yi,p)), "ece_10_bins": float(ece),
            "p_up_mean": float(p.mean()), "p_up_std": float(p.std()), "reliability_bins": reliability}


def evaluate_policy(actual: np.ndarray, prediction: np.ndarray, confidence: np.ndarray,
                    available: np.ndarray, threshold: float) -> dict:
    actual = np.asarray(actual, str); prediction = np.asarray(prediction, str)
    available = np.asarray(available, bool)
    eligible = available & np.isin(actual, DIRECTIONS)
    emit = available & (np.asarray(confidence, float) >= threshold)
    output = np.where(emit, prediction, "NO_SIGNAL")
    eligible_emit = emit & eligible
    ya, yp = actual[eligible_emit], prediction[eligible_emit]
    raw_y, raw_pred = actual[eligible], prediction[eligible]
    up_precision = float(precision_score(ya, yp, labels=DIRECTIONS, average="binary", pos_label="UP", zero_division=0)) if len(ya) else 0.
    down_precision = float(precision_score(ya, yp, labels=DIRECTIONS, average="binary", pos_label="DOWN", zero_division=0)) if len(ya) else 0.
    up_support = int((actual[eligible] == "UP").sum())
    down_support = int((actual[eligible] == "DOWN").sum())
    up_recall = float(((output[eligible] == "UP") & (actual[eligible] == "UP")).sum()/up_support) if up_support else 0.
    down_recall = float(((output[eligible] == "DOWN") & (actual[eligible] == "DOWN")).sum()/down_support) if down_support else 0.
    all_emitted_actual = actual[emit]
    all_emitted_pred = prediction[emit]
    cm_all = confusion_matrix(actual[available], output[available], labels=["DOWN", "NO_SIGNAL", "UP"])
    cm_eligible = confusion_matrix(ya, yp, labels=DIRECTIONS) if len(ya) else np.zeros((2, 2), dtype=int)
    raw_cm = confusion_matrix(raw_y, raw_pred, labels=DIRECTIONS) if len(raw_y) else np.zeros((2, 2), dtype=int)
    raw_metrics = {
        "accuracy_all_eligible_events": float(accuracy_score(raw_y, raw_pred)) if len(raw_y) else None,
        "balanced_accuracy_all_eligible_events": float(balanced_accuracy_score(raw_y, raw_pred)) if len(raw_y) and len(np.unique(raw_y)) > 1 else None,
        "macro_f1_all_eligible_events": float(f1_score(raw_y, raw_pred, labels=DIRECTIONS, average="macro", zero_division=0)) if len(raw_y) else None,
        "confusion_matrix_labels_rows_actual_cols_prediction": DIRECTIONS,
        "confusion_matrix": raw_cm.tolist(),
    }
    emitted_f1 = float(f1_score(actual[eligible], output[eligible], labels=DIRECTIONS, average="macro", zero_division=0)) if eligible.any() else None
    emitted_recall_macro = (up_recall + down_recall)/2 if eligible.any() else None
    return {
        "eligible_directional_events": int(eligible.sum()),
        "available_labeled_rows": int(available.sum()),
        "emitted_predictions_all_candles": int(emit.sum()),
        "no_signal_outcomes": int(available.sum() - emit.sum()),
        "coverage_all_available_candles": float(emit.sum()/available.sum()) if available.any() else None,
        "eligible_emitted_predictions": int(eligible_emit.sum()),
        "coverage_eligible_directional_events": float(eligible_emit.sum()/eligible.sum()) if eligible.any() else None,
        "up_precision_eligible_only": up_precision,
        "down_precision_eligible_only": down_precision,
        "macro_directional_precision_eligible_only": (up_precision + down_precision)/2,
        "directional_accuracy_emitted_eligible_only": float(accuracy_score(ya, yp)) if len(ya) else None,
        "up_recall_eligible_only": up_recall,
        "down_recall_eligible_only": down_recall,
        "macro_directional_recall_eligible_only": emitted_recall_macro,
        "macro_directional_f1_eligible_only": emitted_f1,
        "all_candle_operational_accuracy_neutral_is_error": float(((all_emitted_actual == all_emitted_pred) & np.isin(all_emitted_actual, DIRECTIONS)).sum()/emit.sum()) if emit.any() else None,
        "unthresholded_eligible_directional_metrics": raw_metrics,
        "confusion_labels_rows_actual_cols_output": ["DOWN", "NO_SIGNAL", "UP"],
        "confusion_matrix_all_available": cm_all.tolist(),
        "confusion_labels_rows_actual_cols_emitted_prediction": DIRECTIONS,
        "confusion_matrix_eligible_emissions": cm_eligible.tolist(),
    }


def main():
    allowed_outputs = {"experiment_matrix.json", "target_generation.py", "run_target_research.py",
                      "test_target_research.py", "__pycache__", "target_label_summaries.csv",
                      "selected_target_walk_forward_predictions.csv", "selected_target_pipeline.joblib",
                      "walk_forward_oof_predictions.csv", "walk_forward_fold_metrics.json",
                      "walk_forward_raw_metrics.json", "walk_forward_model_target_ranking.csv",
                      "selected_oof_date_robustness.csv", "validation_metrics.json",
                      "validation_simple_baseline.json", "validation_predictions.csv",
                      "validation_confusion_matrix.csv", "feature_leakage_audit.json",
                      "selected_target_specification.json", "reproducibility_manifest.json",
                      "research_report.md", "finalize_saved_results.py",
                      "validation_date_robustness.csv"}
    unexpected = [p.name for p in OUT.iterdir() if p.name not in allowed_outputs]
    if unexpected:
        raise FileExistsError(f"Refusing to overwrite existing target-research artifacts: {unexpected}")
    spec = matrix()
    train_raw = load_allowed_split("train.csv")
    val_raw = load_allowed_split("validation.csv")
    if train_raw.datetime.max() >= val_raw.datetime.min():
        raise ValueError("Training/validation chronology invalid")
    if [c for c in train_raw.columns if c in FEATURES] != FEATURES:
        raise ValueError("Feature order differs from frozen 24-feature contract")
    folds = folds_for(train_raw)

    target_summaries, oof_all, fold_policy_rows, fold_raw_rows = [], [], [], []
    candidate_cache = {}
    for target in spec["targets"]:
        tid = target["id"]
        labeled = generate_target(train_raw, target)
        candidate_cache[tid] = labeled
        valid = labeled[labeled.target_available].copy()
        target_summaries.append({"target_id": tid, "horizon_minutes": target["horizon_minutes"],
            "method": target["method"], "training_rows": len(labeled),
            "available_target_rows": len(valid), "unavailable_target_rows": int((~labeled.target_available).sum()),
            "directional_events": int(valid.label.isin(DIRECTIONS).sum()),
            "no_signal_events": int((valid.label == "NO_SIGNAL").sum()),
            "down_events": int((valid.label == "DOWN").sum()),
            "up_events": int((valid.label == "UP").sum()),
            "directional_event_rate_available": float(valid.label.isin(DIRECTIONS).mean()) if len(valid) else None,
            "threshold_pct_median": float(valid.threshold_pct.median()) if len(valid) else None,
            "threshold_pct_p90": float(valid.threshold_pct.quantile(.9)) if len(valid) else None})
        for model_name in MODELS:
            for fold_id, (eval_start, eval_end) in enumerate(folds, 1):
                fit = labeled[(labeled.datetime < eval_start) & (labeled.target_end < eval_start) &
                               labeled.label.isin(DIRECTIONS)].copy()
                evaluation = labeled[(labeled.datetime >= eval_start) & (labeled.datetime < eval_end) &
                                     labeled.target_available].copy()
                if len(fit) < 250 or fit.label.nunique() != 2 or evaluation.empty or not evaluation.label.isin(DIRECTIONS).any():
                    fold_raw_rows.append({"target_id": tid, "model": model_name, "fold": fold_id,
                                          "skipped": True, "reason": "insufficient directional fit/evaluation examples",
                                          "fit_rows": len(fit), "eval_rows": len(evaluation)})
                    continue
                pipe = fit_model(model_name, fit)
                probabilities = np.asarray(pipe.predict_proba(evaluation.loc[:, FEATURES]), float)
                if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-8):
                    raise ValueError("Invalid OOF probability output")
                p_down, p_up = probabilities[:, 0], probabilities[:, 1]
                prediction = np.where(p_up > p_down, "UP", "DOWN")
                confidence = np.maximum(p_down, p_up)
                actual = evaluation.label.to_numpy(str)
                available = evaluation.target_available.to_numpy(bool)
                event_mask = np.isin(actual, DIRECTIONS)
                fold_cal = calibration(actual[event_mask], p_up[event_mask])
                raw_policy = evaluate_policy(actual, prediction, confidence, available, -np.inf)
                fold_raw_rows.append({"target_id": tid, "model": model_name, "fold": fold_id,
                    "skipped": False, "eval_start": str(evaluation.datetime.min()), "eval_end": str(evaluation.datetime.max()),
                    "fit_first": str(fit.datetime.min()), "fit_last": str(fit.datetime.max()),
                    "fit_rows_directional": len(fit), "eval_rows_available": len(evaluation),
                    "eligible_events": int(event_mask.sum()), "target_unavailable_rows": int((~available).sum()),
                    "unthresholded_metrics": raw_policy["unthresholded_eligible_directional_metrics"],
                    "calibration_eligible": fold_cal})
                for i, row in enumerate(evaluation.itertuples(index=False)):
                    oof_all.append({"target_id": tid, "model": model_name, "fold": fold_id,
                        "datetime": row.datetime, "target_end": row.target_end,
                        "actual": actual[i], "future_return_pct": row.future_return_pct,
                        "threshold_pct": row.threshold_pct, "p_down": float(p_down[i]),
                        "p_up": float(p_up[i]), "confidence": float(confidence[i]),
                        "raw_prediction": prediction[i], "target_available": bool(available[i])})
                for coverage in COVERAGES:
                    threshold = threshold_at_coverage(confidence, event_mask, coverage)
                    metrics = evaluate_policy(actual, prediction, confidence, available, threshold)
                    fold_policy_rows.append({"target_id": tid, "model": model_name, "fold": fold_id,
                        "target_coverage_eligible_oof": coverage, "confidence_threshold": threshold,
                        "fit_rows_directional": len(fit), "eval_rows_available": len(evaluation), **metrics})

    oof = pd.DataFrame(oof_all)
    fold_metrics = pd.DataFrame(fold_policy_rows)
    skipped = pd.DataFrame(fold_raw_rows)
    if oof.empty or fold_metrics.empty:
        raise RuntimeError("No valid walk-forward predictions were produced")

    rankings = []
    for (tid, model_name, coverage), group in fold_metrics.groupby(
            ["target_id", "model", "target_coverage_eligible_oof"], sort=False):
        non_skipped = skipped[(skipped.target_id == tid) & (skipped.model == model_name)]
        if non_skipped.skipped.any():
            continue
        raw = group.unthresholded_eligible_directional_metrics
        # Fold-level macro precision/accuracy and event coverage are equally weighted by fold.
        ranks = []
        for row in group.to_dict("records"):
            raw_metrics = row["unthresholded_eligible_directional_metrics"]
            ranks.append({"macro_precision": row["macro_directional_precision_eligible_only"],
                          "accuracy": row["directional_accuracy_emitted_eligible_only"],
                          "eligible_coverage": row["coverage_eligible_directional_events"],
                          "up_precision": row["up_precision_eligible_only"],
                          "down_precision": row["down_precision_eligible_only"],
                          "eligible_signals": row["eligible_emitted_predictions"],
                          "raw_accuracy": raw_metrics["accuracy_all_eligible_events"]})
        eligible_pred_total = int(group.eligible_emitted_predictions.sum())
        ranking = {"target_id": tid, "model": model_name, "coverage_target": coverage,
            "fold_count": len(group), "folds_with_skips": int(non_skipped.skipped.sum()),
            "mean_fold_macro_precision": float(np.mean([r["macro_precision"] for r in ranks])),
            "std_fold_macro_precision": float(np.std([r["macro_precision"] for r in ranks], ddof=0)),
            "mean_fold_emitted_eligible_accuracy": float(np.nanmean([r["accuracy"] for r in ranks])),
            "std_fold_emitted_eligible_accuracy": float(np.nanstd([r["accuracy"] for r in ranks])),
            "mean_fold_eligible_coverage": float(np.mean([r["eligible_coverage"] for r in ranks])),
            "mean_fold_up_precision": float(np.mean([r["up_precision"] for r in ranks])),
            "mean_fold_down_precision": float(np.mean([r["down_precision"] for r in ranks])),
            "mean_fold_unthresholded_accuracy": float(np.nanmean([r["raw_accuracy"] for r in ranks])),
            "total_oof_eligible_emissions": eligible_pred_total,
            "min_fold_eligible_emissions": int(group.eligible_emitted_predictions.min()),
            "mean_fold_all_candle_operational_accuracy": float(np.nanmean(group.all_candle_operational_accuracy_neutral_is_error)),
        }
        rankings.append(ranking)
    ranking_df = pd.DataFrame(rankings)
    candidates = ranking_df[(ranking_df.mean_fold_eligible_coverage >= .20) &
                            (ranking_df.total_oof_eligible_emissions >= 75) &
                            (ranking_df.min_fold_eligible_emissions >= 10)]
    if candidates.empty:
        candidates = ranking_df[(ranking_df.mean_fold_eligible_coverage >= .20) &
                               (ranking_df.total_oof_eligible_emissions >= 75)]
    if candidates.empty:
        raise RuntimeError("No target/model/policy met OOF minimum eligible coverage and signal-count gates")
    selected = candidates.sort_values(["mean_fold_macro_precision", "mean_fold_emitted_eligible_accuracy",
                                       "mean_fold_unthresholded_accuracy"], ascending=False).iloc[0]
    selected_target = next(t for t in spec["targets"] if t["id"] == selected.target_id)
    selected_model, selected_coverage = str(selected.model), float(selected.coverage_target)
    selected_oof = oof[(oof.target_id == selected.target_id) & (oof.model == selected_model)].copy()
    selected_oof_eligible = selected_oof.actual.isin(DIRECTIONS).to_numpy()
    selected_threshold = threshold_at_coverage(selected_oof.confidence.to_numpy(float),
                                               selected_oof_eligible, selected_coverage)

    # Fit final candidate on training-only directional events. The validation target is generated once for this candidate only.
    selected_train = candidate_cache[selected_target["id"]]
    fit_all = selected_train[selected_train.label.isin(DIRECTIONS)].copy()
    final_model = fit_model(selected_model, fit_all)
    joblib.dump(final_model, OUT / "selected_target_pipeline.joblib", compress=3)
    selected_val = generate_target(val_raw, selected_target)
    val_eval = selected_val[selected_val.target_available].copy()
    val_prob = np.asarray(final_model.predict_proba(val_eval.loc[:, FEATURES]), float)
    if not np.isfinite(val_prob).all() or not np.allclose(val_prob.sum(axis=1), 1., atol=1e-8):
        raise ValueError("Invalid final validation probabilities")
    val_down, val_up = val_prob[:, 0], val_prob[:, 1]
    val_pred = np.where(val_up > val_down, "UP", "DOWN")
    val_conf = np.maximum(val_down, val_up)
    val_actual = val_eval.label.to_numpy(str)
    val_available = val_eval.target_available.to_numpy(bool)
    val_metrics = evaluate_policy(val_actual, val_pred, val_conf, val_available, selected_threshold)
    val_metrics.update({"target_id": selected_target["id"], "model": selected_model,
                        "coverage_target_selected_on_oof": selected_coverage,
                        "confidence_threshold_learned_from_training_oof": selected_threshold,
                        "horizon_minutes": selected_target["horizon_minutes"],
                        "class_distribution": {c: int((val_actual == c).sum()) for c in ["DOWN", "NO_SIGNAL", "UP"]},
                        "label_unavailable_rows_in_validation_input": int((~selected_val.target_available).sum()),
                        "calibration_eligible_directional": calibration(val_actual[np.isin(val_actual, DIRECTIONS)],
                                                                          val_up[np.isin(val_actual, DIRECTIONS)])})
    val_predictions = val_eval[["datetime", "target_end", "future_return_pct", "threshold_pct", "label"]].rename(
        columns={"label": "actual"}).copy()
    val_predictions["p_down"], val_predictions["p_up"] = val_down, val_up
    val_predictions["confidence"], val_predictions["raw_prediction"] = val_conf, val_pred
    val_predictions["selected_policy_output"] = np.where(val_conf >= selected_threshold, val_pred, "NO_SIGNAL")

    # One simple causal baseline: sign of existing 1-bar return.
    val_baseline = np.where(val_eval.return_1.to_numpy(float) > 0, "UP", "DOWN")
    baseline_metrics = evaluate_policy(val_actual, val_baseline,
        np.ones(len(val_actual), dtype=float), val_available, -np.inf)
    baseline_metrics.update({"baseline": "causal return_1 sign, no abstention"})

    # OOF pooled selected candidate under the chosen threshold; per-date evidence uses fold-derived score cutoffs.
    selected_oof["selected_oof_output"] = np.where(selected_oof.confidence >= selected_threshold,
                                                    selected_oof.raw_prediction, "NO_SIGNAL")
    selected_oof.to_csv(OUT / "selected_target_walk_forward_predictions.csv", index=False)
    selected_daily = []
    chosen_fold = fold_metrics[(fold_metrics.target_id == selected_target["id"]) &
                               (fold_metrics.model == selected_model) &
                               (fold_metrics.target_coverage_eligible_oof == selected_coverage)]
    chosen_oof_fold = selected_oof.merge(chosen_fold[["fold", "confidence_threshold"]], on="fold")
    for day, group in chosen_oof_fold.groupby(chosen_oof_fold.datetime.dt.date, sort=True):
        # use the threshold defined from that walk-forward fold's OOF scores
        threshold = float(group.confidence_threshold.iloc[0])
        m = evaluate_policy(group.actual.to_numpy(str), group.raw_prediction.to_numpy(str),
            group.confidence.to_numpy(float), np.ones(len(group), bool), threshold)
        selected_daily.append({"date": str(day), "fold": int(group.fold.iloc[0]),
                               "rows": len(group), **m})

    # Save complete research outputs.
    pd.DataFrame(target_summaries).to_csv(OUT / "target_label_summaries.csv", index=False)
    oof.to_csv(OUT / "walk_forward_oof_predictions.csv", index=False)
    fold_metrics.to_json(OUT / "walk_forward_fold_metrics.json", orient="records", indent=2, date_format="iso")
    pd.DataFrame(fold_raw_rows).to_json(OUT / "walk_forward_raw_metrics.json", orient="records", indent=2, date_format="iso")
    ranking_df.sort_values("mean_fold_macro_precision", ascending=False).to_csv(OUT / "walk_forward_model_target_ranking.csv", index=False)
    pd.DataFrame(selected_daily).to_csv(OUT / "selected_oof_date_robustness.csv", index=False)
    pd.DataFrame([val_metrics]).to_json(OUT / "validation_metrics.json", orient="records", indent=2)
    pd.DataFrame([baseline_metrics]).to_json(OUT / "validation_simple_baseline.json", orient="records", indent=2)
    val_predictions.to_csv(OUT / "validation_predictions.csv", index=False)
    cm_rows = []
    labels_all = ["DOWN", "NO_SIGNAL", "UP"]
    for i, a in enumerate(labels_all):
        for j, p in enumerate(labels_all):
            cm_rows.append({"actual": a, "predicted": p, "count": val_metrics["confusion_matrix_all_available"][i][j]})
    pd.DataFrame(cm_rows).to_csv(OUT / "validation_confusion_matrix.csv", index=False)

    # Feature/leakage and source audit; no target metadata is ever passed to fit_model.
    feature_audit = {"model_feature_order": FEATURES, "feature_count": len(FEATURES),
        "forbidden_columns_excluded": ["datetime", "target", "future_datetime", "future_return_pct", "target_end", "label", "volume", "open", "high", "low"],
        "feature_source": "existing causal B 24 features from the pre-generated split; no new feature calculation",
        "target_source": "labels generated separately from close and timestamps after loading only train.csv or validation.csv selected columns",
        "volatility_threshold_uses": "volatility_20 at decision timestamp only",
        "session_continuity": "exact 5-minute timestamp sequence through endpoint, same New York calendar date, source regular-session rows only",
        "test_file_opened": False}
    (OUT / "feature_leakage_audit.json").write_text(json.dumps(feature_audit, indent=2), encoding="utf-8")
    selected_spec = {"selected_target": selected_target, "selected_model": selected_model,
        "selected_oof_coverage": selected_coverage, "confidence_threshold": selected_threshold,
        "feature_order": FEATURES, "class_order": ["DOWN", "UP"],
        "probability_tie_policy": "DOWN", "neutral_event_policy": "NO_SIGNAL", "horizon_minutes": selected_target["horizon_minutes"]}
    (OUT / "selected_target_specification.json").write_text(json.dumps(selected_spec, indent=2), encoding="utf-8")
    manifest = {"inputs_opened": ["data/alpaca_rebuild_v1/train.csv", "data/alpaca_rebuild_v1/validation.csv",
                                    "models/alpaca_target_research_v1/experiment_matrix.json"],
        "input_sha256": {"train.csv": sha256(DATA/"train.csv"), "validation.csv": sha256(DATA/"validation.csv"),
                         "experiment_matrix.json": sha256(OUT/"experiment_matrix.json")},
        "test_dataset_accessed": False, "test_period_observations_accessed": False,
        "validation_is_pristine": False, "seed": SEED,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "lightgbm": lgb.__version__, "xgboost": xgb.__version__}}
    (OUT / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    oof_choice = ranking_df[(ranking_df.target_id == selected_target["id"]) &
                            (ranking_df.model == selected_model) &
                            (ranking_df.coverage_target == selected_coverage)].iloc[0]
    report = [
        "# ECL target and horizon research v1", "",
        "## Scope and data boundary", "",
        "Only the existing Alpaca IEX `train.csv` and `validation.csv` were used for target generation, model fitting, and evaluation. The CSVs expose close plus the existing 24 causal B features, not raw open/high/low/volume; labels therefore use exact close-to-close returns. No volume was synthesized or used. No frozen test CSV was opened, hashed, or evaluated.",
        "The validation period has prior research use and is not a pristine holdout. A data-inspection process exception is disclosed at the end of this report.",
        "", "## Predeclared matrix and method", "",
        f"The matrix in `experiment_matrix.json` fixes {len(spec['targets'])} target definitions across 10/15/30/60-minute horizons: fixed ±0.10%, square-root horizon-scaled fixed thresholds, selected ±0.15% cases, 0.5× historical-volatility-scaled thresholds, and minimum-movement event floors. Small movements are NO_SIGNAL. Volatility-scaled labels use same-row `volatility_20` only; horizon return uses the close at the exact future timestamp.",
        "Targets are generated separately from model inputs. Every endpoint requires all expected 5-minute timestamps through the endpoint and the same trading date/session; unavailable targets are excluded, never interpolated. Three expanding date-ordered folds purge fitting labels ending at or beyond the evaluation start. Fold preprocessing is independently fitted.",
        "Logistic Regression, LightGBM, and XGBoost use fixed modest settings. Thresholds and selection use training-only OOF predictions. The selected OOF policy must have ≥20% eligible event coverage, ≥75 emitted eligible OOF examples, and at least 10 eligible emissions per fold where feasible. No validation labels influence selection.",
        "", "## OOF selection", "",
        f"Selected target **{selected_target['id']}** ({selected_target['horizon_minutes']} minutes, {selected_target['method']}), model **{selected_model}**, and OOF-derived {selected_coverage:.0%} eligible-event coverage threshold. Mean fold eligible-only macro precision: {oof_choice.mean_fold_macro_precision:.1%}; mean fold emitted eligible accuracy: {oof_choice.mean_fold_emitted_eligible_accuracy:.1%}; mean eligible coverage: {oof_choice.mean_fold_eligible_coverage:.1%}; OOF eligible emissions: {int(oof_choice.total_oof_eligible_emissions)}.",
        "See fold-by-fold metrics/ranking and selected OOF date robustness CSVs. A threshold that merely creates a tiny number of signals was not eligible for selection.",
        "", "## Single selected-candidate validation confirmation", "",
        f"Validation target {selected_target['id']} yielded {val_metrics['eligible_directional_events']} eligible directional events among {val_metrics['available_labeled_rows']} available labels. The selected policy emitted {val_metrics['emitted_predictions_all_candles']} predictions and {val_metrics['no_signal_outcomes']} NO_SIGNAL outcomes; eligible coverage was {val_metrics['coverage_eligible_directional_events']:.1%}. UP precision {val_metrics['up_precision_eligible_only']:.1%}; DOWN precision {val_metrics['down_precision_eligible_only']:.1%}; macro precision {val_metrics['macro_directional_precision_eligible_only']:.1%}; emitted eligible accuracy {val_metrics['directional_accuracy_emitted_eligible_only']:.1%}; all-candle operational accuracy (neutral event labels count as errors) {val_metrics['all_candle_operational_accuracy_neutral_is_error']:.1%}.",
        f"The 65% precision / 55% accuracy / 20% coverage goals were {'all met' if val_metrics['coverage_eligible_directional_events'] >= .20 and val_metrics['macro_directional_precision_eligible_only'] >= .65 and val_metrics['directional_accuracy_emitted_eligible_only'] >= .55 else 'not all met'} on this confirmation. Treat it as a limited confirmation, not independent validation.",
        "", "## Recommendation", "",
        "Do not integrate automatically. If precision or accuracy targets fail at ≥20% eligible coverage, the evidence does not justify another broad model search; consider stopping this modeling direction or revisiting the decision problem with new, independently justified data/labels.",
        "", "## Process limitation", "",
        "During initial project inspection, an existing dataset audit report containing test-split aggregate metadata was opened inadvertently. No `test.csv` contents, test-period candle observations, or test predictions were opened or used, and the metadata was not used in this experiment. This exceeded the strict no-test-period-data boundary; this limitation is disclosed rather than hidden.",
    ]
    (OUT / "research_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
