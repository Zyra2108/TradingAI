"""Build metrics/report from saved fold and one-time validation outputs only."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("ecl_three_class_v1_report_module", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
MODEL_DIR = module.MODEL_DIR
REPORT_DIR = module.REPORT_DIR


def pct(v: object) -> str:
    try:
        return f"{float(v) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in cols) + " |\n"
    divider = "| " + " | ".join("---" for _ in cols) + " |\n"
    body = "".join("| " + " | ".join(str(row.get(k, "—")) for k, _ in cols) + " |\n" for row in rows)
    return header + divider + body


def main() -> None:
    train, validation, folds, features, _ = module.load_data()
    fold_payload = json.loads((REPORT_DIR / "walk_forward_fold_metrics.json").read_text(encoding="utf-8"))
    selection = json.loads((REPORT_DIR / "selection.json").read_text(encoding="utf-8"))
    validation_result = json.loads((REPORT_DIR / "validation_metrics.json").read_text(encoding="utf-8"))
    baseline_comparison = pd.read_csv(REPORT_DIR / "saved_baseline_comparison.csv")
    validation_y = validation.target.astype(str).to_numpy()
    validation_majority = train.target.astype(str).value_counts().idxmax()
    validation_baselines = []
    for name, prediction in (("always_neutral", np.full(len(validation_y), "NEUTRAL")),
                             ("training_majority_class", np.full(len(validation_y), validation_majority))):
        score = module.class_metrics(validation_y, prediction)
        validation_baselines.append({"baseline": name, "training_majority_class": validation_majority,
            "n": score["n"], "accuracy": score["accuracy"], "balanced_accuracy": score["balanced_accuracy"],
            "macro_f1": score["macro_f1"], "weighted_f1": score["weighted_f1"],
            "prediction_distribution": json.dumps(score["prediction_distribution"]),
            "confusion_matrix_actual_rows_predicted_columns": json.dumps(score["confusion_matrix_actual_rows_predicted_columns"])})
    pd.DataFrame(validation_baselines).to_csv(REPORT_DIR / "validation_baseline_metrics.csv", index=False)
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                    "directional_macro_precision", "directional_macro_recall", "directional_macro_f1")
    summary_rows = []
    for name, pooled in selection["all_candidate_oof_metrics"].items():
        row = {"model": name, "n": pooled["n"], "accuracy": pooled["accuracy"],
               "balanced_accuracy": pooled["balanced_accuracy"], "macro_f1": pooled["macro_f1"],
               "weighted_f1": pooled["weighted_f1"],
               "directional_macro_precision": pooled["directional_macro_precision"],
               "directional_macro_recall": pooled["directional_macro_recall"],
               "directional_macro_f1": pooled["directional_macro_f1"],
               "directional_prediction_coverage_all_rows": pooled["directional_prediction_coverage_all_rows"],
               "guardrail_passed": pooled["guardrail"]["passed"]}
        for metric in metric_names:
            fold_values = [r["oof_metrics"][metric] for r in fold_payload if r["model"] == name]
            row[f"mean_fold_{metric}"] = float(np.mean(fold_values))
            row[f"std_fold_{metric}"] = float(np.std(fold_values, ddof=0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "walk_forward_summary.csv", index=False)

    flat_folds = []
    confusion_rows = []
    for record in fold_payload:
        name, fold = record["model"], record["fold"]
        tr, ev = record["train_metrics"], record["oof_metrics"]
        row = {"model": name, "fold": fold, "n_train": record["n_train"], "n_oof": record["n_oof"],
               "purged_train_rows": record["purged_train_rows"], "first_oof_timestamp": record["first_oof_timestamp"],
               "train_accuracy": tr["accuracy"], "train_balanced_accuracy": tr["balanced_accuracy"],
               "train_macro_f1": tr["macro_f1"], "oof_accuracy": ev["accuracy"],
               "oof_balanced_accuracy": ev["balanced_accuracy"], "oof_macro_f1": ev["macro_f1"],
               "oof_weighted_f1": ev["weighted_f1"], "oof_directional_macro_precision": ev["directional_macro_precision"],
               "oof_directional_macro_recall": ev["directional_macro_recall"], "oof_directional_macro_f1": ev["directional_macro_f1"],
               "oof_directional_prediction_coverage_all_rows": ev["directional_prediction_coverage_all_rows"],
               "oof_actual_directional_accuracy": ev["directional_accuracy_actual_directional"],
               "oof_prediction_distribution": json.dumps(ev["prediction_distribution"]),
               "oof_confusion_matrix_actual_rows_predicted_columns": json.dumps(ev["confusion_matrix_actual_rows_predicted_columns"]),
               "oof_per_class": json.dumps(ev["per_class"])}
        if name == "ridge_regression_to_3class":
            row["ridge_zero_boundary_binary_accuracy_on_directional"] = ev.get("zero_boundary_binary_accuracy_on_directional")
            row["ridge_zero_boundary_confusion_down_up"] = json.dumps(ev.get("zero_boundary_directional_confusion_matrix_down_up"))
        flat_folds.append(row)
        for i, actual_class in enumerate(module.CLASS_ORDER):
            for j, predicted_class in enumerate(module.CLASS_ORDER):
                confusion_rows.append({"model": name, "fold": fold, "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "count": ev["confusion_matrix_actual_rows_predicted_columns"][i][j]})
    pd.DataFrame(flat_folds).to_csv(REPORT_DIR / "walk_forward_fold_table.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(REPORT_DIR / "walk_forward_confusion_matrices.csv", index=False)

    # Fixed baselines from training labels: no fit or inference involved.
    saved_oof = pd.read_csv(REPORT_DIR / "walk_forward_predictions.csv", parse_dates=["datetime"])
    baseline_fold_rows = []
    for foldrow in folds.itertuples(index=False):
        trmask, evmask, _ = module.fold_masks(train, foldrow.train_end, foldrow.validation_start, foldrow.validation_end)
        y = train.loc[evmask, "target"].astype(str).to_numpy()
        train_y = train.loc[trmask, "target"].astype(str).to_numpy()
        majority = pd.Series(train_y).value_counts().idxmax()
        row = saved_oof[(saved_oof.model == "xgboost_multiclass") & (saved_oof.scope == "fold_oof") & (saved_oof.fold == int(foldrow.fold))]
        if len(row) != len(y):
            raise ValueError("Saved OOF prediction count does not match fold labels")
        pred_xgb = row.prediction.astype(str).to_numpy()
        for baseline, pred in (("always_neutral", np.full(len(y), "NEUTRAL")),
                               ("training_fold_majority", np.full(len(y), majority))):
            score = module.class_metrics(y, pred)
            baseline_fold_rows.append({"fold": int(foldrow.fold), "baseline": baseline, "train_majority_class": majority,
                "n": len(y), "accuracy": score["accuracy"], "balanced_accuracy": score["balanced_accuracy"],
                "macro_f1": score["macro_f1"], "weighted_f1": score["weighted_f1"],
                "prediction_distribution": json.dumps(score["prediction_distribution"]),
                "confusion_matrix_actual_rows_predicted_columns": json.dumps(score["confusion_matrix_actual_rows_predicted_columns"])})
        # Include saved candidate XGBoost OOF next to the baselines for clarity.
        score = module.class_metrics(y, pred_xgb)
        baseline_fold_rows.append({"fold": int(foldrow.fold), "baseline": "new_xgboost_candidate", "train_majority_class": majority,
            "n": len(y), "accuracy": score["accuracy"], "balanced_accuracy": score["balanced_accuracy"],
            "macro_f1": score["macro_f1"], "weighted_f1": score["weighted_f1"],
            "prediction_distribution": json.dumps(score["prediction_distribution"]),
            "confusion_matrix_actual_rows_predicted_columns": json.dumps(score["confusion_matrix_actual_rows_predicted_columns"])})
    pd.DataFrame(baseline_fold_rows).to_csv(REPORT_DIR / "baseline_fold_metrics.csv", index=False)

    # Zero boundary for the newly evaluated Ridge regressor is binary only.
    ridge_oof = saved_oof[(saved_oof.model == "ridge_regression_to_3class") & (saved_oof.scope == "fold_oof")].copy()
    actual_dir = ridge_oof.actual.isin(["DOWN", "UP"])
    ridge_zero = module.binary_metrics(ridge_oof.loc[actual_dir, "actual"].to_numpy(str),
                                       ridge_oof.loc[actual_dir, "zero_boundary_binary_prediction"].to_numpy(str))
    ridge_zero["n_all_rows"] = len(ridge_oof)
    ridge_zero["all_candle_accuracy_neutral_counted_wrong"] = float((ridge_oof.zero_boundary_binary_prediction == ridge_oof.actual).mean())
    (REPORT_DIR / "ridge_zero_boundary_binary_diagnostic.json").write_text(json.dumps(ridge_zero, indent=2), encoding="utf-8")

    # Compact OOF candidate comparison.
    candidate_rows = []
    for r in summary.to_dict(orient="records"):
        candidate_rows.append({"model": r["model"], "OOF_accuracy": pct(r["accuracy"]),
            "OOF_balanced_accuracy": pct(r["balanced_accuracy"]), "OOF_macro_F1": pct(r["macro_f1"]),
            "OOF_weighted_F1": pct(r["weighted_f1"]), "mean_fold_accuracy": f"{float(r['mean_fold_accuracy'])*100:.2f}% ± {float(r['std_fold_accuracy'])*100:.2f} pp",
            "mean_fold_macro_F1": f"{float(r['mean_fold_macro_f1'])*100:.2f}% ± {float(r['std_fold_macro_f1'])*100:.2f} pp",
            "mean_fold_balanced_accuracy": f"{float(r['mean_fold_balanced_accuracy'])*100:.2f}% ± {float(r['std_fold_balanced_accuracy'])*100:.2f} pp",
            "mean_fold_weighted_F1": f"{float(r['mean_fold_weighted_f1'])*100:.2f}% ± {float(r['std_fold_weighted_f1'])*100:.2f} pp",
            "mean_fold_directional_precision": f"{float(r['mean_fold_directional_macro_precision'])*100:.2f}% ± {float(r['std_fold_directional_macro_precision'])*100:.2f} pp",
            "mean_fold_directional_recall": f"{float(r['mean_fold_directional_macro_recall'])*100:.2f}% ± {float(r['std_fold_directional_macro_recall'])*100:.2f} pp",
            "mean_fold_directional_F1": f"{float(r['mean_fold_directional_macro_f1'])*100:.2f}% ± {float(r['std_fold_directional_macro_f1'])*100:.2f} pp",
            "directional_macro_precision": pct(r["directional_macro_precision"]),
            "directional_macro_recall": pct(r["directional_macro_recall"]),
            "directional_macro_F1": pct(r["directional_macro_f1"]),
            "directional_prediction_coverage": pct(r["directional_prediction_coverage_all_rows"]),
            "guardrail_passed": bool(r["guardrail_passed"])})

    selected = validation_result["selected_candidate"]
    valm = validation_result["metrics"]
    class_rows = []
    for c in module.CLASS_ORDER:
        x = valm["per_class"][c]
        class_rows.append({"class": c, "precision": pct(x["precision"]), "recall": pct(x["recall"]),
                           "F1": pct(x["f1"]), "support": x["support"]})
    val_matrix = pd.DataFrame(valm["confusion_matrix_actual_rows_predicted_columns"], index=module.CLASS_ORDER, columns=module.CLASS_ORDER)
    val_matrix.index.name = "actual\\predicted"
    val_matrix.to_csv(REPORT_DIR / "validation_confusion_matrix.csv")
    baseline_comparison.to_csv(REPORT_DIR / "saved_baseline_comparison.csv", index=False)

    config = json.loads((MODEL_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    spec = json.loads((MODEL_DIR / "frozen_candidate_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((MODEL_DIR / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    fold_dates = pd.read_csv(module.FOLD_CSV)
    fold_date_rows = [{"fold": int(r.fold), "fit_through": r.train_end, "OOF_period": f"{r.validation_start}–{r.validation_end}"} for r in fold_dates.itertuples(index=False)]

    report = f"""# ECL Three-Class ML Research v1

## Decision

**No model demonstrates a repeatable, useful improvement over the existing baselines. Do not promote a candidate.** XGBoost Multiclass was selected from training OOF results for the single limited validation confirmation. Its validation accuracy was {pct(valm['accuracy'])}, far below the validation always-NEUTRAL baseline ({pct(validation.target.eq('NEUTRAL').mean())}). It was only {float(valm['accuracy'] - baseline_comparison.iloc[0]['accuracy'])*100:+.2f} percentage points relative to the stored existing three-class XGBoost baseline on validation, with lower directional macro-F1 ({pct(valm['directional_macro_f1'])} versus {pct(baseline_comparison.iloc[0]['directional_macro_f1'])}). This does not establish meaningful directional improvement.

This validation period has been repeatedly used in prior work and is **not a pristine holdout**. The experiment read allowlist excluded the protected test CSV and saved test predictions; no test rows were loaded or evaluated. No live or frozen model was modified.

## Data and integrity

- Alpaca IEX ECL, 5-minute bars; 15-minute future return and original strict ±0.10% thresholds.
- Training: {len(train):,} rows, {train.datetime.min().date()}–{train.datetime.max().date()}; validation: {len(validation):,} rows, {validation.datetime.min().date()}–{validation.datetime.max().date()}.
- Train labels: {json.dumps(train.target.value_counts().reindex(module.CLASS_ORDER, fill_value=0).to_dict())}; validation labels: {json.dumps(validation.target.value_counts().reindex(module.CLASS_ORDER, fill_value=0).to_dict())}.
- Exact ordered 24-feature contract is taken from the frozen specification. The split CSV has two `close` columns by role (label reference close and model feature close); they were verified equal, renamed in memory to distinct roles, and only the feature close entered the model. Volume and metadata were excluded.
- Split inputs match the frozen specification hashes. Decision times are unique and chronological; labels match saved `future_return`; endpoints equal exactly +15 minutes and remain on the same New York session date. Each fold uses only earlier training rows and purges endpoints at/after the first OOF decision timestamp.
- Four chronological expanding folds:

{md_table(fold_date_rows, [("fold","Fold"),("fit_through","Training through"),("OOF_period","OOF dates")])}

The existing 24 feature calculations were reused from the already-created Alpaca dataset; this experiment did not add indicators or recalculate features. Dataset provenance identifies Alpaca IEX. Feature names contain no target, future timestamp, or volume input.

## Packages and fixed configurations

- scikit-learn {sklearn_version(manifest)}, XGBoost {manifest['xgboost']}; XGBoost 3.4.1 was present and used with `multi:softprob`.
- CatBoost was not installed. It was not installed or silently substituted; therefore CatBoost was not run.
- Ridge alpha=10 with fold-local median imputation and scaling. XGBoost and HistGradientBoosting use fold-local median imputation and training-fold balanced sample weights. Configurations/seeds are in `experiment_config.json`; no broad search was run.

## Training-only walk-forward results

OOF rows are 2,248 pooled held-out training-period predictions across four folds. Scores below use the selected Ridge boundary where applicable. The Ridge threshold sweep used only these OOF predictions, with macro-F1 selection subject to directional coverage, neutral-rate and minimum-per-class guardrails. **No threshold in the predeclared positive threshold grid passed that guardrail.** The diagnostic unconstrained macro-F1 choice was {selection['ridge_neutral_threshold_percentage_points']:.3f} percentage points, but Ridge failed the guardrail and was not eligible as the candidate. The separate zero-boundary diagnostic is binary only and is not the Ridge three-class neutral boundary.

{md_table(candidate_rows, [("model","Candidate"),("OOF_accuracy","OOF accuracy"),("OOF_balanced_accuracy","Balanced acc."),("OOF_macro_F1","Macro F1"),("OOF_weighted_F1","Weighted F1"),("mean_fold_accuracy","Mean fold accuracy ± SD"),("mean_fold_macro_F1","Mean fold macro-F1 ± SD"),("mean_fold_balanced_accuracy","Mean fold balanced acc. ± SD"),("directional_macro_precision","Directional macro P"),("directional_macro_recall","Directional macro R"),("directional_macro_F1","Directional macro F1"),("directional_prediction_coverage","Directional output coverage"),("guardrail_passed","Guardrail")])}

Detailed fold-level training/OOF accuracy, balanced accuracy, macro/weighted F1, class P/R/F1 and confusion matrices are in `walk_forward_fold_table.csv` and `walk_forward_confusion_matrices.csv`. The OOF and in-sample training scores show substantial overfit for the tree models: XGBoost mean in-sample fold accuracy was about {np.mean([x['train_metrics']['accuracy'] for x in fold_payload if x['model']=='xgboost_multiclass']):.1%}, versus mean OOF fold accuracy {np.mean([x['oof_metrics']['accuracy'] for x in fold_payload if x['model']=='xgboost_multiclass']):.1%}; HistGradientBoosting showed the same pattern. Ridge did not overfit comparably, but produced too many neutral calls and weak UP/DOWN recognition under its three-class policy.

The new XGBoost candidate was the highest OOF mean-fold macro-F1 among guardrail-valid candidates ({pct(summary.loc[summary.model == selected, 'mean_fold_macro_f1'].iloc[0])}). Fold accuracy varied materially; it was particularly weak in fold 1 and did not exceed 50% in any fold. Its pooled OOF accuracy was below the training-period always-NEUTRAL accuracy ({pct((train.target == 'NEUTRAL').mean())}).

## Ridge zero-boundary diagnostic and saved controls

The new Ridge regressor's zero-sign diagnostic, restricted to actual UP/DOWN OOF examples, scored {pct(ridge_zero['accuracy'])}; all-candle accuracy (actual neutral counted incorrect) was {pct(ridge_zero['all_candle_accuracy_neutral_counted_wrong'])}. It is binary only and makes no neutral class prediction. The already-saved Ridge and binary XGBoost control outputs, plus the prior three-class XGBoost validation predictions, are in `saved_baseline_comparison.csv`; no comparator model was refit or reinferred.

Among saved controls, existing binary Ridge OOF directional accuracy was 54.04% pooled (fold mean differs due unequal fold sizes) and validation was 48.88%; existing binary XGBoost v3 was 52.61% pooled OOF and 47.21% validation. These binary models remain directional controls, not three-class competitors. The previous three-class XGBoost baseline scored 40.31% validation accuracy and 36.20% macro-F1.

## One-time limited validation confirmation

After selection and freezing the candidate config/hash, only the selected XGBoost Multiclass artifact was evaluated on validation. Class order is DOWN, NEUTRAL, UP.

- N={valm['n']:,}; overall accuracy {pct(valm['accuracy'])}; balanced accuracy {pct(valm['balanced_accuracy'])}; macro-F1 {pct(valm['macro_f1'])}; weighted-F1 {pct(valm['weighted_f1'])}.
- Directional macro precision/recall/F1: {pct(valm['directional_macro_precision'])} / {pct(valm['directional_macro_recall'])} / {pct(valm['directional_macro_f1'])}.
- Directional output coverage over all rows {pct(valm['directional_prediction_coverage_all_rows'])}; among actual directional rows {pct(valm['directional_prediction_coverage_actual_directional'])}; accuracy among emitted directional predictions {pct(valm['accuracy_among_directional_predictions'])}.
- Prediction distribution: `{json.dumps(valm['prediction_distribution'])}`.
- Log loss: {valm['log_loss']:.4f}.

{md_table(class_rows, [("class","Class"),("precision","Precision"),("recall","Recall"),("F1","F1"),("support","Support")])}

Validation confusion matrix, actual rows / predicted columns `[DOWN, NEUTRAL, UP]`:

```text
{json.dumps(valm['confusion_matrix_actual_rows_predicted_columns'])}
```

Validation always-NEUTRAL accuracy was {pct(validation.target.eq('NEUTRAL').mean())}. The selected model predicted {valm['prediction_distribution']['NEUTRAL']} NEUTRAL rows of {valm['n']}, so its weak accuracy is not caused by excessive NEUTRAL output; the model instead makes many incorrect directional calls. The result is near the old three-class XGBoost baseline, not a meaningful gain.

Existing binary controls on their saved validation predictions (actual directional rows only): Ridge zero-sign {pct(float(baseline_comparison.loc[baseline_comparison.model.str.startswith('existing_ridge'), 'accuracy'].iloc[0]))}; binary XGBoost v3 {pct(float(baseline_comparison.loc[baseline_comparison.model.str.startswith('existing_binary_xgboost'), 'accuracy'].iloc[0]))}. Validation is reused and only a limited confirmation.

## Integrity, artifacts and reproduction

- Frozen selected research candidate: `{spec['selected_candidate']}`; train-only artifact SHA256 `{spec['artifact_sha256']}`. This is a research artifact, not the production model.
- Existing protected model artifacts hashed before/after are unchanged (`reproducibility_manifest.json`). Train/validation input hashes and all new output hashes are recorded there.
- No test CSV/predictions/metrics were included in the read allowlist. This experiment code contains no path to the protected test split.
- CatBoost unavailable; no installation attempted.
- Focused tests: 7 passed. Python syntax compilation passed. The first execution completed all folds and one selected validation inference; its final saved-baseline join failed due string/datetime mismatch. The timestamps were parsed and the remaining reports finalized from the already saved outputs; no model fit or validation inference was repeated.

Reproduce the experiment from a clean new output directory with:

```powershell
.\\.venv\\Scripts\\python.exe models\\alpaca_three_class_v1\\run_experiment.py
```

The runner refuses to overwrite existing outputs. Outputs are isolated under `models/alpaca_three_class_v1/` and `reports/alpaca_three_class_v1/`.

## Conclusion

Ridge regression converted to three classes failed its OOF directional-coverage/class-distribution guardrail. XGBoost Multiclass and HistGradientBoosting passed the distribution guardrail but had low directional macro precision/recall and under-40% pooled OOF accuracy. XGBoost was the strongest by predeclared OOF macro-F1 tie hierarchy, but its validation accuracy (40.83%) remained well below always-NEUTRAL (53.66%), and its directional macro-F1 did not improve on the existing three-class XGBoost baseline. **No candidate demonstrated repeatable useful improvement; do not promote or integrate any candidate.** A future genuinely untouched evaluation would be needed before making out-of-sample claims.
"""
    (REPORT_DIR / "research_report.md").write_text(report, encoding="utf-8")


def sklearn_version(manifest: dict) -> str:
    return str(manifest.get("sklearn", "not recorded"))


if __name__ == "__main__":
    main()
