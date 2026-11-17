"""Refresh the research decision text from saved v2 metrics; no data or inference is run."""
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
metrics_path = ROOT / "validation_metrics.json"
summary_path = ROOT / "summary.json"
manifest_path = ROOT / "reproducibility_manifest.json"
report_path = ROOT / "research_report.md"
training_script = ROOT / "train_binary_v2.py"
model_spec_path = ROOT / "model_specification.json"

spec = importlib.util.spec_from_file_location("v2_audit", training_script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
audit_rows, prior_audit = module.audit_previous_v1()
audit_rows.to_csv(ROOT / "previous_v1_metric_audit.csv", index=False)
(ROOT / "previous_v1_audit.json").write_text(json.dumps(prior_audit, indent=2), encoding="utf-8")

fold_metrics = json.loads((ROOT / "walk_forward_fold_metrics.json").read_text(encoding="utf-8"))
fold_bins = []
for record in fold_metrics:
    for bin_row in record["calibration_eligible"]["reliability_bins"]:
        fold_bins.append({"model": record["model"], "feature_set": record["feature_set"],
                          "fold": record["fold"], **bin_row})
import pandas as pd
pd.DataFrame(fold_bins).to_csv(ROOT / "walk_forward_calibration_bins.csv", index=False)

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["previous_v1_audit"] = prior_audit
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
feature_schema = json.loads((ROOT / "feature_schema.json").read_text(encoding="utf-8"))
training_config = json.loads((ROOT / "training_config.json").read_text(encoding="utf-8"))
policies = metrics["selective_policies"]
meaningful = [p for p in policies if p["coverage_eligible_directional"] >= .20]
best_p = max(p["eligible_directional_metrics_on_emissions"]["macro_directional_precision"] for p in meaningful)
best_a = max(p["eligible_directional_metrics_on_emissions"]["accuracy"] for p in meaningful)
all_best_p = max(p["eligible_directional_metrics_on_emissions"]["macro_directional_precision"] for p in policies)
all_best_a = max(p["eligible_directional_metrics_on_emissions"]["accuracy"] for p in policies)
met = any(p["coverage_eligible_directional"] >= .20 and
          p["eligible_directional_metrics_on_emissions"]["macro_directional_precision"] >= .65 and
          p["eligible_directional_metrics_on_emissions"]["accuracy"] >= .55 for p in policies)
attainment = {
    "minimum_eligible_directional_coverage": .20,
    "precision_target": .65,
    "accuracy_target": .55,
    "both_targets_met_at_minimum_coverage": met,
    "best_macro_precision_at_coverage_at_least_20pct": best_p,
    "best_accuracy_at_coverage_at_least_20pct": best_a,
    "best_macro_precision_any_coverage": all_best_p,
    "best_accuracy_any_coverage": all_best_a,
    "highest_precision_accuracy_policy": policies[0],
}
metrics["target_attainment"] = attainment
pd.DataFrame(metrics["calibration_eligible_directional"]["reliability_bins"]).to_csv(
    ROOT / "validation_calibration_bins.csv", index=False)
model_spec = {
    "model_file": "selected_binary_pipeline.joblib",
    "selected_model": metrics["selected_model"],
    "feature_set": metrics["selected_feature_set"],
    "feature_order": feature_schema["feature_sets"][metrics["selected_feature_set"]],
    "class_order": ["DOWN", "UP"],
    "encoded_classes": {"DOWN": 0, "UP": 1},
    "target": training_config["target"],
    "preprocessing": training_config["preprocessing"],
    "threshold_source": "Training-only chronological OOF confidence quantiles among actual UP/DOWN rows.",
    "selective_policy_thresholds": {
        str(p["target_oof_coverage_eligible_directional"]): p["threshold_learned_from_training_oof"]
        for p in metrics["selective_policies"]
    },
    "decision": "Return UP or DOWN when max class probability meets the selected confidence threshold; otherwise NO SIGNAL. Equal probabilities predict DOWN.",
    "test_data_accessed": False,
}
model_spec_path.write_text(json.dumps(model_spec, indent=2), encoding="utf-8")
manifest["model_specification_sha256"] = module.file_sha256(model_spec_path)
summary["target_attainment"] = attainment
metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

manifest["v2_validation_inference_passes"] = 2
manifest["training_script_sha256"] = module.file_sha256(training_script)
manifest["training_config_sha256"] = module.file_sha256(ROOT / "training_config.json")
manifest["feature_schema_sha256"] = module.file_sha256(ROOT / "feature_schema.json")
manifest["input_sha256"]["frozen_candidate_spec_sha256"] = module.file_sha256(
    ROOT.parents[0] / "alpaca_final_freeze_v1" / "frozen_candidate_spec.json")
manifest["inputs_read"] = sorted(set(manifest["inputs_read"] + [
    "models/alpaca_binary_research_v1/metrics.json"]))
manifest["validation_process_note"] = (
    "One provisional v2 pass used an all-candle coverage denominator and was discarded. "
    "The corrected reported thresholds are derived from training-only OOF eligible-direction scores. "
    "Model/feature ranking was OOF-only; provisional metrics were not used to rank candidates. "
    "The threshold denominator was corrected to comply with the requested eligible-direction coverage definition."
)
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

report = report_path.read_text(encoding="utf-8")
report = report.replace("validation was evaluated once after selection.",
    "the corrected validation confirmation follows training-only selection. A provisional v2 evaluation was discarded after an all-candle versus eligible-direction coverage-denominator mismatch was identified; it did not affect model ranking, but the correction means validation predictions were produced in two v2 passes.")
report = report.replace("## One-time validation confirmation", "## Corrected validation confirmation")
old = next(line for line in report.splitlines() if line.startswith(("Across the four OOF-derived thresholds,", "A nominal 20% OOF eligible-coverage threshold")))
new = (f"A nominal 20% OOF eligible-coverage threshold produced only {policies[0]['coverage_eligible_directional']:.1%} validation eligible coverage "
       f"({policies[0]['eligible_emitted_predictions']} eligible emitted examples); its highest macro precision/accuracy ({all_best_p:.1%}/{all_best_a:.1%}) "
       f"does not meet the coverage requirement. Among policies reaching at least 20% validation eligible coverage, best macro precision was {best_p:.1%} "
       f"and best eligible emitted accuracy was {best_a:.1%}. The three goals were {'met together' if met else 'not met together'}.")
report = report.replace(old, new)
process_note = " The validation set was run through inference once provisionally and once for the corrected coverage policy; the provisional results were discarded and not used for model ranking or threshold selection."
report = report.replace(process_note, "")
report = report.replace("This is variable-date diagnostic evidence, not independent confirmation.",
    "This is variable-date diagnostic evidence, not independent confirmation." + process_note)
report = report.replace("it did not establish the stated precision and accuracy targets.",
    "it did not establish the stated precision and accuracy targets at the required eligible-direction coverage.")
needle = "Added features use only the 24 existing same-candle causal inputs."
report = report.replace(needle, needle + " Reliability-bin CSVs for OOF folds and validation are included; aggregate calibration results are in the JSON metrics.")
report_path.write_text(report, encoding="utf-8")
