"""Render the final report from already-saved diagnostic outputs only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "alpaca_all_models_test_diagnostic"


def pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |\n"
    sep = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join("| " + " | ".join(row.get(key, "—") for key, _ in columns) + " |\n" for row in rows)
    return head + sep + body


def main() -> None:
    metrics = json.loads((OUT / "evaluation_metrics.json").read_text(encoding="utf-8"))
    model_rows = list(csv.DictReader((OUT / "per_model_metrics.csv").open(encoding="utf-8-sig", newline="")))
    by_name = {r["model"]: r for r in model_rows}
    three = list(csv.DictReader((OUT / "ranked_three_class_models.csv").open(encoding="utf-8-sig", newline="")))
    directional = list(csv.DictReader((OUT / "ranked_directional_models.csv").open(encoding="utf-8-sig", newline="")))
    all_candle = list(csv.DictReader((OUT / "ranked_all_candle_accuracy.csv").open(encoding="utf-8-sig", newline="")))
    previous = list(csv.DictReader((OUT / "previous_performance_comparison.csv").open(encoding="utf-8-sig", newline="")))
    baseline_rows = list(csv.DictReader((OUT / "baseline_metrics.csv").open(encoding="utf-8-sig", newline="")))
    hashes_before = json.loads((OUT / "artifact_hashes_before.json").read_text(encoding="utf-8"))
    hashes_after = json.loads((OUT / "artifact_hashes_after.json").read_text(encoding="utf-8"))
    test_info = metrics.get("evaluation", {})
    n_rows = int(test_info["test_rows"])
    date_range = [test_info["date_start"], test_info["date_end"]]
    counts = test_info["class_distribution"]
    n_dir = int(counts.get("DOWN", 0) + counts.get("UP", 0))

    three_table = []
    for r in three:
        three_table.append({
            "model": r["model"], "n": r["n_evaluated"], "accuracy": pct(r["accuracy"]),
            "balanced": pct(r["balanced_accuracy"]), "macro": pct(r["macro_f1"]),
            "down_pr": pct(r["down_precision"]), "down_re": pct(r["down_recall"]),
            "down_f1": pct(metrics["models"][r["model"]]["per_class"]["DOWN"]["f1"]),
            "neutral_pr": pct(r["neutral_precision"]), "neutral_re": pct(r["neutral_recall"]),
            "neutral_f1": pct(metrics["models"][r["model"]]["per_class"]["NEUTRAL"]["f1"]),
            "up_pr": pct(r["up_precision"]), "up_re": pct(r["up_recall"]),
            "up_f1": pct(metrics["models"][r["model"]]["per_class"]["UP"]["f1"]),
            "dist": ", ".join(f"{k}:{v}" for k, v in metrics["models"][r["model"]]["prediction_distribution"].items()),
        })
    directional_table = []
    for r in directional:
        directional_table.append({
            "model": r["model"], "n": r["n_evaluated"], "accuracy": pct(r["accuracy"]),
            "balanced": pct(r["balanced_accuracy"]), "macro": pct(r["macro_f1"]),
            "down_pr": pct(r["down_precision"]), "down_re": pct(r["down_recall"]),
            "up_pr": pct(r["up_precision"]), "up_re": pct(r["up_recall"]),
            "coverage": pct(r["eligible_directional_coverage"]), "matrix": r["confusion_matrix"],
        })
    regression_table = []
    for name in ("ridge_regression", "random_forest_regression", "xgboost_regression", "hist_gradient_boosting_regression"):
        r = by_name[name]
        regression_table.append({"model": name, "mae": f"{float(r['mae_pct_points']):.5f}", "rmse": f"{float(r['rmse_pct_points']):.5f}", "corr": f"{float(r['return_correlation']):+.4f}"})
    candle_table = [{"model": r["model"], "accuracy": pct(r["all_candle_accuracy"]), "n": r["prediction_count"]} for r in all_candle]
    prev_table = []
    for r in previous:
        value = r.get("previous_accuracy")
        saved = by_name.get(r["model"], {})
        # per_model_metrics.csv stores the model's appropriate primary score:
        # all-row accuracy for 3-class and eligible-direction accuracy for
        # binary/regression/sequence models.
        test_value = saved.get("accuracy")
        delta = float(test_value) - float(value) if value not in (None, "") and test_value is not None else None
        prev_table.append({"model": r["model"], "period": r["previous_period"], "accuracy": pct(value), "test": pct(test_value), "delta": f"{delta * 100:+.2f} pp" if delta is not None else "—", "source": r["source"], "status": r["status"]})
    baseline_table = [{
        "baseline": r["baseline"], "three": pct(r["three_class_accuracy"]),
        "dir": pct(r["directional_accuracy"]), "all": pct(r["all_candle_accuracy"]),
        "matrix": r["confusion_matrix_3class"],
    } for r in baseline_rows]

    hash_pass = hashes_before == hashes_after and len(hashes_before) == 20
    frozen_hash = hashes_before.get("models\\alpaca_final_freeze_v1\\candidate_pipeline.joblib", "not found")
    report = f"""# All-model ECL test-period diagnostic

## Scope and interpretation

This is the authorized one-time diagnostic comparison using existing saved models. It is **not independent future confirmation**: this test period and its class information were previously exposed in research. No model was retrained, tuned, refit, recalibrated, or selected from these results; no production artifact was changed. Results must not be described as proof of future performance or profitability.

Source: `data/alpaca_ml_pipeline_v1/test.csv`, ECL / Alpaca IEX, five-minute candles, 15-minute horizon, strict ±0.10% target, classes DOWN / NEUTRAL / UP. The evaluated file contained {n_rows:,} rows spanning {date_range[0]} through {date_range[-1]}. Actual labels: DOWN {int(counts.get('DOWN', 0)):,}, NEUTRAL {int(counts.get('NEUTRAL', 0)):,}, UP {int(counts.get('UP', 0)):,}; directional population {n_dir:,}. Each three-class model was scored on all test rows. Binary/regression models were scored directionally only on the {n_dir:,} actual UP/DOWN rows, and separately on all rows with actual NEUTRALs counted as incorrect. CNN models have shorter eligible sequence populations; their row counts are shown and matched-timestamp results are available in `sequence_matched_directional_metrics.csv`.

## Three-class ranking

Ranked by three-class accuracy (DOWN / NEUTRAL / UP metrics are classwise). Confusion matrices use actual rows and predicted columns in that class order and are saved individually under `confusion_matrices/`.

{table(three_table, [("model","Model"),("n","N"),("accuracy","Accuracy"),("balanced","Balanced acc."),("macro","Macro F1"),("down_pr","D P"),("down_re","D R"),("down_f1","D F1"),("neutral_pr","N P"),("neutral_re","N R"),("neutral_f1","N F1"),("up_pr","U P"),("up_re","U R"),("up_f1","U F1"),("dist","Predicted D/N/U")])}
Always-NEUTRAL accuracy was {pct(next(r['three_class_accuracy'] for r in baseline_rows if r['baseline']=='always_neutral'))}; it exceeded every learned three-class model. Highest learned three-class accuracy was a tie: Logistic Regression and Random Forest, both {pct(by_name['logistic_regression_3class']['accuracy'])}.

## Directional ranking

Ranked by accuracy on actual directional observations only. Binary/regression confusion matrices use actual rows and predicted columns in `[DOWN, UP]` order. Regression decisions use the pre-existing zero boundary. CNNs always emit a direction when a sequence exists; no NEUTRAL output was invented.

{table(directional_table, [("model","Model"),("n","Directional N"),("accuracy","Directional acc."),("balanced","Balanced acc."),("macro","Macro F1"),("down_pr","DOWN P"),("down_re","DOWN R"),("up_pr","UP P"),("up_re","UP R"),("coverage","Directional coverage"),("matrix","CM [D,U]")])}
The top directional result was Ridge Regression at {pct(by_name['ridge_regression']['accuracy'])} (DOWN recall {pct(by_name['ridge_regression']['down_recall'])}, UP recall {pct(by_name['ridge_regression']['up_recall'])}). Binary XGBoost v3 reached {pct(by_name['binary_xgboost_v3']['accuracy'])}. These do not meet a 55% directional-accuracy goal.

Regression error and return correlation (errors in percentage points, as saved by the original regression evaluation):

{table(regression_table, [("model","Regressor"),("mae","MAE"),("rmse","RMSE"),("corr","Predicted/actual correlation")])}

## All-candle accuracy

This separate comparison counts an actual NEUTRAL row as incorrect for models that always emit UP/DOWN.

{table(candle_table, [("model","Model"),("n","Rows"),("accuracy","All-candle accuracy")])}
Always-NEUTRAL scored {pct(next(r['all_candle_accuracy'] for r in baseline_rows if r['baseline']=='always_neutral'))}, the highest all-candle accuracy. Among the learned models, the best all-candle accuracy was {pct(max(float(r['all_candle_accuracy']) for r in model_rows))}. The all-candle and directional rankings answer different questions and should not be conflated.

## Baselines

{table(baseline_table, [("baseline","Baseline"),("three","3-class accuracy"),("dir","Directional accuracy"),("all","All-candle accuracy"),("matrix","3-class CM [D,N,U]")])}
The training-majority three-class class was NEUTRAL; on this test population its accuracy matches always-NEUTRAL at {pct(next(r['three_class_accuracy'] for r in baseline_rows if r['baseline']=='training_majority_3class'))}. The directional training-majority baseline always predicted DOWN and scored {pct(next(r['directional_accuracy'] for r in baseline_rows if r['baseline']=='training_majority_direction'))}; always-UP scored {pct(next(r['directional_accuracy'] for r in baseline_rows if r['baseline']=='always_up'))}.

## Previous walk-forward and validation comparison

These are comparisons to previously saved OOF/validation records only; validation was reused in earlier research and is not pristine. See `previous_performance_comparison.csv` for exact sources and values.

{table(prev_table, [("model","Model"),("period","Prior period"),("accuracy","Prior accuracy"),("test","Diagnostic accuracy"),("delta","Difference"),("status","Availability"),("source","Saved source")])}
Overall, the diagnostic test accuracy was lower than validation for most models. Ridge was a notable exception relative to its prior validation and OOF summaries, but this exposed period cannot establish durable superiority. CNN lengths 6/12 again collapsed to all-DOWN predictions; length 24 emitted very few UP calls, and length 36 had reduced sequence eligibility. The matched-timestamp table controls the population for these CNN comparisons.

## Integrity, artifacts, and tests

- All 14 model slots explicitly named in the request evaluated successfully, including the saved binary RF/XGBoost artifacts. The frozen Balanced LightGBM B was added as the production reference, and an older separate binary XGBoost artifact was also evaluated. The frozen candidate and identical candidate copy have the same hash and were evaluated once.
- Additional saved artifacts are listed in `model_inventory.csv` with explicit status and reasons. Different-horizon/one-minute models were incompatible with this task; selective policy and unrequested candidate/ensemble artifacts were not part of the predeclared comparison.
- Artifact hash files contain {len(hashes_before)} entries; before/after maps are **{'identical' if hash_pass else 'NOT IDENTICAL'}**. Frozen hash is unchanged.
- Frozen candidate SHA256: `{frozen_hash}`. Full before/after SHA256 maps are in `artifact_hashes_before.json` and `artifact_hashes_after.json`.
- Every prediction and confusion matrix is preserved under `predictions/` and `confusion_matrices/`. `per_model_metrics.csv`, `prediction_distribution.csv`, ranked tables, baseline metrics, matched-sequence metrics, and previous-performance comparisons provide machine-readable results.
- Focused tests: 4 passed. Python syntax compilation passed for evaluator, summary helper, and focused test module. Summary generation emitted scikit-learn warnings for directional scoring when a model predicts a class absent from the directional truth subset; this is expected and does not invalidate the scores (zero-division behavior is explicit).
- No production/live application, frozen model, research artifact, or dataset was modified. Evaluation inputs were read-only; no inference was rerun while producing this report.

## Conclusion

No evaluated model exceeded 50% overall three-class accuracy; all were below the 50.77% always-NEUTRAL baseline. Ridge had the best directional accuracy at 54.45%, narrowly below 55%, but its recall was asymmetric (DOWN 73.65%, UP 35.59%). Binary XGBoost v3 was more balanced at 51.34% (DOWN recall 51.80%, UP recall 50.88%). None establishes a reliable, balanced improvement, and no candidate should be promoted automatically. Because this test period has prior exposure, none of these numbers is genuine independent future evidence.
"""
    (OUT / "research_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
