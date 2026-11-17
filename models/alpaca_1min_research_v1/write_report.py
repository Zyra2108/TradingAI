from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent
metrics = json.loads((OUT / "validation_metrics.json").read_text(encoding="utf-8"))
ranking = pd.read_csv(OUT / "walk_forward_model_policy_ranking.csv")
folds = pd.read_csv(OUT / "walk_forward_fold_metrics.csv")
cohort = json.loads((OUT / "comparison_cohort.json").read_text(encoding="utf-8"))
selected = metrics["selected_model"]
interval = int(metrics["selected_interval_minutes"])
threshold = float(metrics["confidence_threshold_from_training_oof"])
coverage_target = float(metrics["selected_oof_coverage_target"])

best_by_pair = (ranking.sort_values(
    ["interval_minutes", "model", "selection_eligible", "mean_fold_macro_directional_precision",
     "mean_fold_emitted_eligible_accuracy", "mean_fold_macro_f1", "oof_log_loss_eligible"],
    ascending=[True, True, False, False, False, False, True], kind="stable")
    .groupby(["interval_minutes", "model"], sort=True).head(1))
lines = [
    "# ECL 1-minute input / 10-minute horizon research v1", "",
    "## Summary", "",
    f"Training-only OOF selected **{interval}-minute {selected}** with confidence threshold {threshold:.6f} (nominal {coverage_target:.0%} eligible-coverage policy). It did not meet the requested validation precision or emitted directional accuracy. Eligible validation coverage was {metrics['metrics']['eligible_directional_coverage']:.1%}.",
    "This is a research candidate only. Do not integrate it; keep the frozen three-class model operational.", "",
    "## Data source and permitted dates", "",
    "The source was Alpaca IEX. The fetched bars covered only the existing train/validation date span, 2025-11-13 through 2026-09-24. Training dates were 2025-11-13 through 2026-08-05; validation dates were 2026-08-06 through 2026-09-24. The validation set was previously used in earlier research and is not pristine.",
    "The minute fetch returned 49,348 records, 49,347 valid regular-session rows, 216 trading dates, no duplicate timestamps, no invalid OHLC, one out-of-session record, and 34,533 missing expected regular-session minutes (58.83% coverage). Volume was nonzero and present on all valid rows. The direct 5-minute reference returned 16,060 rows, with 16,059 regular aligned rows, 717 missing expected intervals, and 95.73% coverage.",
    "Strict 5-minute reconstruction from 1-minute data required all five source minutes; only 2,662 complete aggregates remained, and no reconstructed rows survived original B feature warm-up. Therefore the fair 5-minute comparison uses direct Alpaca IEX 5Min bars fetched over the identical allowed date window; no gaps were filled.",
    "", "## Common-date comparison and target", "",
    f"To compare intervals fairly, all variants were restricted to {len(cohort['train_shared_dates'])} shared training dates and {len(cohort['validation_shared_dates'])} shared validation dates. The candle grids differ naturally (1m versus 5m), but every candidate uses the same trading dates. Feature-ready data before this restriction was available on 65 / 15 dates for each interval.",
    "At candle close, the reference is that candle's close. The target uses the close of the candle timestamped exactly 10 minutes later, on the same regular NYSE session, requiring every expected timestamp in between. UP is return > +0.10%; DOWN is < -0.10%; the inclusive interval is NO_SIGNAL and excluded from binary fitting. No cross-session, missing-path, or fabricated labels are allowed.",
    "The 1-minute candidate has 30 causal trailing features, including lagged returns, rolling volatility, candle shape, volume-relative-history, moving-average/EMA relationships, RSI and session-time context. Rolling features reset at session boundaries and missing-minute gaps. Volume is used only as an input feature in the 1-minute candidate; the 5-minute candidate uses the original 24 B price features and excludes volume.",
    "", "## Method", "",
    "Six fixed candidates were compared: Logistic Regression, LightGBM, and XGBoost on each input interval. Preprocessing was fit separately within each expanding chronological fold; the logistic pipeline scales after median imputation. Training rows whose target endpoint reached the fold evaluation start were purged. OOF probabilities were ordered DOWN/UP. Confidence thresholds were determined from training OOF confidence distributions at nominal 20%, 30%, 40%, and 50% eligible coverage. Candidate selection followed experiment_matrix.json and required at least 75 OOF emissions total and at least 10 per fold. No validation labels selected a model or threshold.",
    "", "## Walk-forward results", "",
    "Best OOF policy per fixed input/model pair (the ranking CSV contains every policy level):", "",
    "| Input | Model | Nominal OOF coverage | Achieved OOF coverage | Macro directional precision | Emitted eligible accuracy | Binary accuracy before abstention | Macro F1 | Log loss | Eligible emissions |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in best_by_pair.itertuples(index=False):
    lines.append(f"| {r.interval_minutes}m | {r.model} | {r.coverage_target_oof:.0%} | {r.mean_fold_eligible_coverage:.1%} | {r.mean_fold_macro_directional_precision:.1%} | {r.mean_fold_emitted_eligible_accuracy:.1%} | {r.oof_binary_accuracy_all_eligible:.1%} | {r.mean_fold_macro_f1:.1%} | {r.oof_log_loss_eligible:.4f} | {r.total_oof_eligible_emissions} |")
chosen = folds[(folds.interval_minutes == interval) & (folds.model == selected) & (folds.coverage_target_oof == coverage_target)].sort_values("fold")
lines += ["", f"Selected {interval}m/{selected} fold detail at the selected threshold:", "",
          "| Fold | Evaluation rows | Eligible events | Eligible signals | Coverage | Macro precision | Emitted accuracy | Binary accuracy | Macro F1 |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in chosen.itertuples(index=False):
    lines.append(f"| {int(r.fold)} | {int(r.n_labeled_rows)} | {int(r.n_eligible_directional)} | {int(r.n_emitted_on_eligible)} | {r.eligible_directional_coverage:.1%} | {r.macro_directional_precision:.1%} | {r.directional_accuracy_emitted_eligible:.1%} | {r.binary_accuracy_all_eligible_before_abstention:.1%} | {r.macro_directional_f1_all_eligible_no_signal_is_miss:.1%} |")

m = metrics["metrics"]
pd.DataFrame(m["confusion_matrix_all_rows"], index=["DOWN", "NO_SIGNAL", "UP"],
             columns=["DOWN", "NO_SIGNAL", "UP"]).rename_axis("actual").reset_index().to_csv(
                 OUT / "validation_confusion_matrix.csv", index=False)
lines += ["", "## One final selected-candidate validation confirmation", "",
    f"The selected {interval}m {selected} pipeline emitted {m['n_emitted_signals_all_rows']} signals across {m['n_labeled_rows']} candles. Of {m['n_eligible_directional']} eligible UP/DOWN examples, {m['n_emitted_on_eligible']} received a directional signal ({m['eligible_directional_coverage']:.1%} eligible coverage); {m['n_emitted_on_neutral_actual']} signals were issued on neutral actuals. It returned NO_SIGNAL on {m['n_no_signal_outputs']} candles.",
    f"UP precision {m['up_precision_eligible']:.1%}; DOWN precision {m['down_precision_eligible']:.1%}; macro directional precision {m['macro_directional_precision']:.1%}; accuracy among emitted eligible signals {m['directional_accuracy_emitted_eligible']:.1%}; binary accuracy before abstention across all eligible examples {m['binary_accuracy_all_eligible_before_abstention']:.1%}; macro F1 with abstentions counted as directional misses {m['macro_directional_f1_all_eligible_no_signal_is_miss']:.1%}.",
    f"The validation distribution was DOWN={metrics['class_distribution']['DOWN']}, NO_SIGNAL={metrics['class_distribution']['NO_SIGNAL']}, UP={metrics['class_distribution']['UP']}. Predictions were DOWN={m['predicted_class_distribution'].get('DOWN',0)}, NO_SIGNAL={m['predicted_class_distribution'].get('NO_SIGNAL',0)}, UP={m['predicted_class_distribution'].get('UP',0)}.",
    f"Unthresholded binary accuracy on eligible examples: {m['binary_accuracy_all_eligible_before_abstention']:.1%}. Strict all-candle directional hit rate, counting abstentions and neutral actual rows as misses: {m['all_candle_operational_accuracy_no_signal_and_actual_neutral_are_errors']:.1%}. Three-outcome accuracy, with neutral actual mapped to NO_SIGNAL: {m['all_candle_three_outcome_accuracy_neutral_target_maps_to_no_signal']:.1%}.",
    f"Calibration on eligible examples: log loss {m['calibration_eligible_directional']['log_loss']:.4f}, Brier {m['calibration_eligible_directional']['brier_score']:.4f}, 10-bin ECE {m['calibration_eligible_directional']['ece_10_bins']:.4f}. Confusion matrix rows and columns are actual/output [DOWN, NO_SIGNAL, UP]. See validation_confusion_matrix.csv.",
    "", "## Decision", "",
    "No: this candidate does not reach 65% macro directional precision or 55% emitted eligible accuracy. It does exceed the 20% eligible coverage requirement, but that does not compensate for weak precision and accuracy. OOF mean macro directional precision was below 65%; fold detail varies materially. Validation UP precision is materially higher than DOWN precision, and the lower DOWN recall shows asymmetric errors. Do not integrate this model. These classification metrics say nothing by themselves about profitability.",
    "", "## Process exception and data boundary disclosure", "",
    "During initial setup, a directory listing exposed the test.csv path/size metadata and the existing dataset reproducibility manifest was opened, revealing aggregate test-split metadata. The test.csv contents, test-period candle rows, test predictions, and test-derived metrics were not read or used, and no test-period data was fetched. The setup metadata inspection exceeded the strict no-test-access instruction; it is disclosed here. The final report must not be described as having had zero test-related metadata access.",
    "A preliminary draft run also calculated validation metrics before the same-date-cohort correction. It was not used to select a model or threshold; the corrected run selected solely from OOF and produced the final validation confirmation. Because validation was viewed in that preliminary run, it was technically consulted more than once; validation is not pristine and no further validation-driven choices were made.",
    ""]
(OUT / "research_report.md").write_text("\n".join(lines), encoding="utf-8")

# Correct the manifest disclosure without touching any original project manifests.
manifest_path = OUT / "reproducibility_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest.pop("test_dataset_accessed", None)
manifest["test_csv_contents_read"] = False
manifest["test_period_candles_or_predictions_accessed"] = False
manifest["test_split_manifest_metadata_opened_during_setup"] = True
manifest["validation_preliminary_metrics_viewed_before_same_date_correction"] = True
manifest["final_validation_selection_used_oof_only"] = True
manifest["limitations"] = list(manifest.get("limitations", [])) + [
    "Initial setup exposed test file path/size metadata and read the existing split manifest's aggregate test metadata; no test file contents or test-period candle observations were read.",
    "A preliminary validation metric pass occurred before the common-date correction; it did not affect model/threshold selection, which used training OOF only. The corrected final confirmation is disclosed separately."]
manifest["scripts_sha256"]["write_report.py"] = __import__("hashlib").sha256((OUT / "write_report.py").read_bytes()).hexdigest() if (OUT / "write_report.py").exists() else "will_be_updated_after_script_save"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
