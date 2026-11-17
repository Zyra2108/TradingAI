"""Build comparison CSVs from saved diagnostic predictions/metrics only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "alpaca_all_models_test_diagnostic"


def main() -> None:
    payload = json.loads((OUT / "evaluation_metrics.json").read_text(encoding="utf-8"))
    records = payload["models"]
    dist_rows = []
    for name, record in records.items():
        path = OUT / "predictions" / f"{name}.csv"
        frame = pd.read_csv(path)
        if "sequence_eligible" in frame:
            frame = frame.loc[frame.sequence_eligible.astype(bool)]
            scope = "eligible sequence windows"
        else:
            scope = "all test rows"
        predicted = frame.prediction.astype(str)
        predicted = predicted[~predicted.str.startswith("NOT_EVALUATED")]
        counts = predicted.value_counts().to_dict()
        n = int(len(predicted))
        for label in ("DOWN", "NEUTRAL", "UP"):
            count = int(counts.get(label, 0))
            dist_rows.append({"model": name, "prediction_scope": scope, "class": label, "count": count, "share": count / n if n else 0.0, "total_predictions_in_scope": n})
    pd.DataFrame(dist_rows).to_csv(OUT / "prediction_distribution.csv", index=False)

    # Long CNN windows exclude some test timestamps. Re-score all other saved
    # predictions on each CNN's exact eligible directional timestamps so the
    # ranking cannot benefit merely from a different sample composition.
    prediction_files = {p.stem: pd.read_csv(p) for p in (OUT / "predictions").glob("*.csv")}
    matched_rows = []
    all_directional = sorted({name for name in prediction_files if not name.startswith("cnn_") and name != "baseline_always_neutral"})
    for length in (6, 12, 24, 36):
        cnn = prediction_files[f"cnn_{length}_candles"]
        eligible_times = set(pd.to_datetime(cnn.loc[cnn.sequence_eligible.astype(bool), "datetime"]))
        target = cnn.loc[cnn.sequence_eligible.astype(bool), ["datetime", "target"]].copy()
        target["datetime"] = pd.to_datetime(target.datetime)
        target = target[target.target.isin(["DOWN", "UP"])].sort_values("datetime")
        actual = target.target.astype(str).to_numpy()
        for model_name in [f"cnn_{length}_candles", *all_directional]:
            frame = prediction_files[model_name].copy()
            frame["datetime"] = pd.to_datetime(frame.datetime)
            frame = frame[frame.datetime.isin(eligible_times) & frame.target.isin(["DOWN", "UP"])].sort_values("datetime")
            if model_name.startswith("cnn_"):
                frame = frame[frame.sequence_eligible.astype(bool)]
            pred = frame.prediction.astype(str).to_numpy()
            y = frame.target.astype(str).to_numpy()
            labels = ["DOWN", "UP"]
            matrix = confusion_matrix(y, pred, labels=labels)
            matched_rows.append({
                "sequence_eligibility_reference": length,
                "model": model_name,
                "n_actual_directional": len(y),
                "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(y) else None,
                "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)) if len(y) else None,
                "down_precision": float(precision_score(y, pred, labels=labels, average=None, zero_division=0)[0]) if len(y) else None,
                "down_recall": float(recall_score(y, pred, labels=labels, average=None, zero_division=0)[0]) if len(y) else None,
                "up_precision": float(precision_score(y, pred, labels=labels, average=None, zero_division=0)[1]) if len(y) else None,
                "up_recall": float(recall_score(y, pred, labels=labels, average=None, zero_division=0)[1]) if len(y) else None,
                "confusion_matrix_actual_rows_predicted_columns_down_up": json.dumps(matrix.tolist()),
            })
    pd.DataFrame(matched_rows).to_csv(OUT / "sequence_matched_directional_metrics.csv", index=False)

    # Values are drawn exclusively from already-saved OOF/validation summaries.
    sources = {
        "logistic_regression_3class": ("reports/alpaca_baselines_v1/validation_comparison.csv", "logistic_regression", "accuracy", "validation"),
        "random_forest_classifier_3class": ("reports/alpaca_baselines_v1/validation_comparison.csv", "random_forest", "accuracy", "validation"),
        "xgboost_classifier_3class": ("reports/alpaca_baselines_v1/validation_comparison.csv", "xgboost", "accuracy", "validation"),
        "hist_gradient_boosting_classifier_3class": ("reports/alpaca_baselines_v1/validation_comparison.csv", "hist_gradient_boosting", "accuracy", "validation"),
        "binary_random_forest_v3": ("reports/alpaca_binary_research_v3/validation_model_comparison.csv", "random_forest", "accuracy", "validation"),
        "binary_xgboost_v3": ("reports/alpaca_binary_research_v3/validation_model_comparison.csv", "xgboost", "accuracy", "validation"),
        "binary_xgboost_v1_additional": ("models/alpaca_binary_research_v1/metrics.csv", "xgboost_binary", "accuracy", "validation"),
        "ridge_regression": ("reports/alpaca_return_regression_v1/validation_metrics.csv", "ridge", "directional_accuracy", "validation"),
        "random_forest_regression": ("reports/alpaca_return_regression_v1/validation_metrics.csv", "random_forest_regressor", "directional_accuracy", "validation"),
        "xgboost_regression": ("reports/alpaca_return_regression_v1/validation_metrics.csv", "xgboost_regressor", "directional_accuracy", "validation"),
        "hist_gradient_boosting_regression": ("reports/alpaca_return_regression_v1/validation_metrics.csv", "hist_gradient_boosting_regressor", "directional_accuracy", "validation"),
    }
    # The frozen B validation metric is recorded in its final freeze report,
    # not in the generic baseline comparison CSV.
    sources["frozen_balanced_lightgbm_B_reference"] = (
        "models/alpaca_final_freeze_v1/audit_report.md",
        "frozen_balanced_lightgbm_B_reference",
        "accuracy",
        "validation",
    )
    # CNN and binary model previous walk-forward summaries use different schemas.
    for length in (6, 12, 24, 36):
        sources[f"cnn_{length}_candles"] = ("reports/alpaca_sequence_research_v1/validation_metrics.csv", length, "accuracy", "validation")
    comparison_rows = []
    for model, (relative, key, metric, period) in sources.items():
        path = ROOT / relative
        if not path.exists():
            comparison_rows.append({"model": model, "previous_period": period, "previous_accuracy": None, "source": relative, "status": "UNAVAILABLE"})
            continue
        if model == "frozen_balanced_lightgbm_B_reference":
            import re
            text = path.read_text(encoding="utf-8")
            match = re.search(r"'accuracy':\s*([0-9.]+)", text)
            value = float(match.group(1)) if match else None
            comparison_rows.append({"model": model, "previous_period": period, "previous_accuracy": value, "source": relative, "status": "AVAILABLE" if value is not None else "NOT_FOUND"})
            continue
        frame = pd.read_csv(path)
        if model.startswith("cnn_"):
            selected = frame.loc[(frame.model == "causal_1d_cnn") & (frame.sequence_length == key)]
        else:
            col = "model" if "model" in frame.columns else "family"
            selected = frame.loc[frame[col] == key]
        if selected.empty:
            comparison_rows.append({"model": model, "previous_period": period, "previous_accuracy": None, "source": relative, "status": "NOT_FOUND"})
            continue
        row = selected.iloc[0]
        comparison_rows.append({"model": model, "previous_period": period, "previous_accuracy": float(row[metric]), "source": relative, "status": "AVAILABLE"})

    # Add training-period OOF summaries for models that have them.
    oof_paths = {
        "binary_random_forest_v3": ("reports/alpaca_binary_research_v3/walk_forward_config_comparison.csv", "random_forest", "rf_compact_unweighted", "mean_accuracy"),
        "binary_xgboost_v3": ("reports/alpaca_binary_research_v3/walk_forward_config_comparison.csv", "xgboost", "xgb_shallow_unweighted", "mean_accuracy"),
        "ridge_regression": ("reports/alpaca_return_regression_v1/walk_forward_overall_metrics.csv", "ridge", None, "directional_accuracy"),
        "random_forest_regression": ("reports/alpaca_return_regression_v1/walk_forward_overall_metrics.csv", "random_forest_regressor", None, "directional_accuracy"),
        "xgboost_regression": ("reports/alpaca_return_regression_v1/walk_forward_overall_metrics.csv", "xgboost_regressor", None, "directional_accuracy"),
        "hist_gradient_boosting_regression": ("reports/alpaca_return_regression_v1/walk_forward_overall_metrics.csv", "hist_gradient_boosting_regressor", None, "directional_accuracy"),
    }
    for length in (6, 12, 24, 36):
        oof_paths[f"cnn_{length}_candles"] = ("reports/alpaca_sequence_research_v1/walk_forward_overall_metrics.csv", "causal_1d_cnn", length, "accuracy")
    for model, (relative, family, config, metric) in oof_paths.items():
        path = ROOT / relative
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if model.startswith("cnn_"):
            selected = frame.loc[(frame.model == family) & (frame.sequence_length == config)]
        elif config:
            selected = frame.loc[(frame.family == family) & (frame.configuration == config)]
        else:
            selected = frame.loc[frame.model == family]
        if not selected.empty:
            comparison_rows.append({"model": model, "previous_period": "walk_forward_OOF", "previous_accuracy": float(selected.iloc[0][metric]), "source": relative, "status": "AVAILABLE"})
    pd.DataFrame(comparison_rows).to_csv(OUT / "previous_performance_comparison.csv", index=False)

    inventory = pd.read_csv(OUT / "model_inventory.csv")
    extra = [
        {"model": "alpaca_candidates_v1_lightgbm_conservative", "artifacts": "models/alpaca_candidates_v1/lightgbm_conservative.joblib", "status": "NOT_IN_REQUESTED_MODEL_SET", "reason": "Compatible 24-feature three-class LightGBM artifact, but LightGBM was not among the four named classifier families; frozen Balanced B was included as the production reference."},
        {"model": "alpaca_candidates_v1_multinomial_logistic", "artifacts": "models/alpaca_candidates_v1/multinomial_logistic.joblib", "status": "NOT_IN_REQUESTED_MODEL_SET", "reason": "Additional candidate logistic pipeline outside the named alpaca_baselines_v1 classifier set."},
        {"model": "alpaca_candidates_v1_dummy", "artifacts": "models/alpaca_candidates_v1/dummy_most_frequent.joblib", "status": "BASELINE_RECOMPUTED_FROM_TRAIN", "reason": "The training-majority and always-neutral baselines were computed directly from the canonical training labels and applied to the test rows."},
        {"model": "alpaca_binary_research_v2_selective_logistic", "artifacts": "models/alpaca_binary_research_v2/selected_binary_pipeline.joblib", "status": "NOT_IN_REQUESTED_MODEL_SET", "reason": "This artifact applies a validation-selected NO SIGNAL policy; no selective policy was requested for this diagnostic. The original raw binary RF/XGBoost baselines were evaluated."},
        {"model": "alpaca_binary_research_v1_lightgbm_binary", "artifacts": "models/alpaca_binary_research_v1/lightgbm_binary.joblib", "status": "NOT_IN_REQUESTED_MODEL_SET", "reason": "Separate binary LightGBM candidate was outside the specifically requested binary RF/XGBoost additions."},
        {"model": "alpaca_binary_research_v3_logistic_and_hist_gradient_boosting_binary", "artifacts": "models/alpaca_binary_research_v3/logistic_regression.joblib; models/alpaca_binary_research_v3/hist_gradient_boosting.joblib", "status": "NOT_IN_REQUESTED_MODEL_SET", "reason": "The extra binary artifacts requested were Random Forest and XGBoost; the four named classifier models are evaluated as three-class models."},
    ]
    inventory = pd.concat([inventory, pd.DataFrame(extra)], ignore_index=True)
    inventory = inventory.drop_duplicates(subset=["model"], keep="first")
    inventory.to_csv(OUT / "model_inventory.csv", index=False)


if __name__ == "__main__":
    main()
