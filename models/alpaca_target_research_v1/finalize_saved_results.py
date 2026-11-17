"""Summarize saved OOF/validation outputs; does not load data or rerun inference."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
train_script = ROOT / "run_target_research.py"
spec = importlib.util.spec_from_file_location("target_research_saved", train_script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

metrics = json.loads((ROOT / "validation_metrics.json").read_text(encoding="utf-8"))[0]
selected = json.loads((ROOT / "selected_target_specification.json").read_text(encoding="utf-8"))
folds = json.loads((ROOT / "walk_forward_fold_metrics.json").read_text(encoding="utf-8"))
pred = pd.read_csv(ROOT / "validation_predictions.csv", parse_dates=["datetime", "target_end"])
threshold = float(selected["confidence_threshold"])
daily = []
for day, group in pred.groupby(pred.datetime.dt.date, sort=True):
    actual = group.actual.to_numpy(str)
    raw = group.raw_prediction.to_numpy(str)
    confidence = group.confidence.to_numpy(float)
    available = np.ones(len(group), dtype=bool)
    stats = module.evaluate_policy(actual, raw, confidence, available, threshold)
    daily.append({"date": str(day), **stats})
pd.DataFrame(daily).to_csv(ROOT / "validation_date_robustness.csv", index=False)

chosen_folds = [r for r in folds if r.get("target_id") == selected["selected_target"]["id"]
                and r.get("model") == selected["selected_model"]
                and r.get("target_coverage_eligible_oof") == selected["selected_oof_coverage"]]
date_data = pd.read_csv(ROOT / "selected_oof_date_robustness.csv")
sufficient_oof_dates = date_data[date_data.eligible_emitted_predictions >= 3]
sufficient_val_dates = pd.DataFrame(daily)
sufficient_val_dates = sufficient_val_dates[sufficient_val_dates.eligible_emitted_predictions >= 2]

report = (ROOT / "research_report.md").read_text(encoding="utf-8")
fold_lines = ["", "Selected-candidate fold results:", "", "| Fold | Eligible events | Signals on eligible events | UP precision | DOWN precision | Macro precision | Directional accuracy |", "|---:|---:|---:|---:|---:|---:|---:|"]
for row in sorted(chosen_folds, key=lambda r: r["fold"]):
    fold_lines.append(f"| {row['fold']} | {row['eligible_directional_events']} | {row['eligible_emitted_predictions']} | {row['up_precision_eligible_only']:.1%} | {row['down_precision_eligible_only']:.1%} | {row['macro_directional_precision_eligible_only']:.1%} | {row['directional_accuracy_emitted_eligible_only']:.1%} |")
date_note = (f"Across {len(sufficient_oof_dates)} OOF dates with at least three eligible signals, daily directional accuracy averaged "
             f"{sufficient_oof_dates.directional_accuracy_emitted_eligible_only.mean():.1%} (SD {sufficient_oof_dates.directional_accuracy_emitted_eligible_only.std(ddof=0):.1%}). "
             if len(sufficient_oof_dates) else "There were too few OOF dates with at least three eligible signals for a stable date-level summary. ")
val_date_note = (f"At validation, {len(sufficient_val_dates)} dates had at least two eligible emitted signals; their daily directional accuracy averaged "
                 f"{sufficient_val_dates.directional_accuracy_emitted_eligible_only.mean():.1%} (SD {sufficient_val_dates.directional_accuracy_emitted_eligible_only.std(ddof=0):.1%}). "
                 if len(sufficient_val_dates) else "At validation, too few dates had at least two eligible emitted signals for a stable date-level summary. ")
report = report.replace("See fold-by-fold metrics/ranking and selected OOF date robustness CSVs. A threshold that merely creates a tiny number of signals was not eligible for selection.",
    "See fold-by-fold metrics/ranking and selected OOF date robustness CSVs. A threshold that merely creates a tiny number of signals was not eligible for selection.\n" + "\n".join(fold_lines) + "\n\n" + date_note)
report = report.replace("Treat it as a limited confirmation, not independent validation.",
    "Treat it as a limited confirmation, not independent validation. " + val_date_note + "Daily results are saved in `validation_date_robustness.csv`.")
(ROOT / "research_report.md").write_text(report, encoding="utf-8")

manifest_path = ROOT / "reproducibility_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
manifest["selected_model_sha256"] = digest(ROOT / "selected_target_pipeline.joblib")
manifest["target_generation_script_sha256"] = digest(ROOT / "target_generation.py")
manifest["training_evaluation_script_sha256"] = digest(train_script)
manifest["experiment_matrix_sha256"] = digest(ROOT / "experiment_matrix.json")
manifest["saved_results_finalizer_sha256"] = digest(Path(__file__).resolve())
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
