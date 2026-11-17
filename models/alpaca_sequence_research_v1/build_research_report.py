from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT / "reports" / "alpaca_sequence_research_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md_table(frame: pd.DataFrame, columns: list[tuple[str, str, str | None]]) -> str:
    """columns: (key, label, format), format is pct/float/str."""
    lines = ["| " + " | ".join(label for _, label, _ in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for key, _, fmt in columns:
            value = row[key]
            if fmt == "pct":
                value = f"{100 * float(value):.1f}%"
            elif fmt == "float":
                value = f"{float(value):.3f}"
            else:
                value = str(value)
            vals.append(value)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    report = REPORT_DIR / "research_report.md"
    source = PROJECT / "models" / "alpaca_1min_research_v1" / "raw_ecl_iex_5min_train_validation.csv"
    source_meta = source.with_name("fetch_5min_metadata.json")
    metadata = json.loads(source_meta.read_text(encoding="utf-8"))
    if (metadata.get("symbol", "").upper(), metadata.get("feed", "").lower(), metadata.get("timeframe")) != ("ECL", "iex", "5Min"):
        raise ValueError("Approved source metadata does not match ECL Alpaca IEX 5Min.")
    if metadata.get("permitted_new_york_date_range") != ["2025-11-13", "2026-09-24"]:
        raise ValueError("Source metadata is outside the approved pre-test period.")
    if digest(source) != metadata.get("raw_csv_sha256"):
        raise ValueError("Approved raw source hash mismatch.")
    raw = pd.read_csv(source)
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True, errors="raise")
    raw["datetime_ny"] = raw.timestamp_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    raw["session_date"] = raw.datetime_ny.dt.date
    if raw.timestamp_utc.max() > pd.Timestamp("2026-09-24 23:59:59", tz="UTC"):
        raise ValueError("Raw data extends beyond the permitted cutoff.")
    ohlc = ["open", "high", "low", "close"]
    invalid = (
        (raw.low > raw[["open", "close"]].min(axis=1))
        | (raw.high < raw[["open", "close"]].max(axis=1))
        | (raw.low > raw.high)
        | (raw[ohlc] <= 0).any(axis=1)
        | (raw.volume < 0)
    )
    local_min = raw.datetime_ny.dt.hour * 60 + raw.datetime_ny.dt.minute
    gaps = raw.groupby("session_date", sort=True).timestamp_utc.diff().dropna()
    gaps = gaps[gaps > pd.Timedelta(minutes=5)]
    source_audit = {
        "source_path": str(source.relative_to(PROJECT)),
        "symbol": "ECL", "feed": "IEX", "timeframe": "5Min",
        "rows": int(len(raw)),
        "first_timestamp_utc": raw.timestamp_utc.min().isoformat(),
        "last_timestamp_utc": raw.timestamp_utc.max().isoformat(),
        "new_york_trading_dates": int(raw.session_date.nunique()),
        "duplicate_timestamps": int(raw.timestamp_utc.duplicated().sum()),
        "invalid_ohlc_rows": int(invalid.sum()),
        "zero_volume_rows": int((raw.volume == 0).sum()),
        "out_of_regular_session_rows": int((~local_min.between(570, 955)).sum()),
        "within_session_gaps_over_five_minutes": int(len(gaps)),
        "gap_minutes_distribution": {str(k): int(v) for k, v in gaps.dt.total_seconds().div(60).value_counts().sort_index().items()},
        "gaps_are_preserved_as_sequence_resets": True,
        "source_sha256": digest(source),
        "metadata_sha256": digest(source_meta),
    }
    (REPORT_DIR / "source_data_audit.json").write_text(json.dumps(source_audit, indent=2), encoding="utf-8")

    eligibility = pd.read_csv(REPORT_DIR / "sequence_eligibility.csv")
    overall = pd.read_csv(REPORT_DIR / "walk_forward_overall_metrics.csv")
    folds = pd.read_csv(REPORT_DIR / "walk_forward_fold_metrics.csv")
    validation = pd.read_csv(REPORT_DIR / "validation_metrics.csv")
    date_summary = pd.read_csv(REPORT_DIR / "date_stability_summary.csv")

    eligibility_table = md_table(eligibility, [
        ("split", "Split", None), ("sequence_length", "L", None), ("total_labeled_rows", "Labeled", None),
        ("eligible_rows", "Sequence eligible", None), ("directional_total", "Directional total", None),
        ("directional_eligible", "Directional eligible", None), ("eligible_directional_coverage", "Coverage", "pct"),
        ("eligible_dates", "Dates", None),
    ])

    comp = overall[overall.model.isin(["causal_1d_cnn", "random_forest", "xgboost"])].sort_values(["sequence_length", "model"])
    comparison = md_table(comp, [
        ("model", "Model", None), ("sequence_length", "L", None), ("fold_oof_directional_n", "OOF n", None),
        ("sequence_eligibility_coverage", "Coverage", "pct"), ("accuracy", "Accuracy", "pct"),
        ("balanced_accuracy", "Balanced acc.", "pct"), ("macro_f1", "Macro F1", "pct"),
        ("directional_macro_precision", "Macro precision", "pct"), ("down_recall", "DOWN recall", "pct"),
        ("down_precision", "DOWN precision", "pct"), ("up_recall", "UP recall", "pct"),
        ("up_precision", "UP precision", "pct"), ("log_loss", "Log loss", "float"),
        ("confusion_matrix", "Confusion [DOWN,UP]", None),
    ])

    cnn_folds = folds[folds.model == "causal_1d_cnn"].sort_values(["sequence_length", "fold"]).copy()
    for cls in ("DOWN", "UP"):
        values = cnn_folds.class_metrics.map(ast.literal_eval)
        cnn_folds[f"{cls.lower()}_precision"] = values.map(lambda x: x[cls]["precision"])
        cnn_folds[f"{cls.lower()}_recall"] = values.map(lambda x: x[cls]["recall"])
    fold_table = md_table(cnn_folds, [
        ("sequence_length", "L", None), ("fold", "Fold", None), ("n_directional", "n", None),
        ("actual_down", "Actual D", None), ("actual_up", "Actual U", None),
        ("prediction_down", "Pred D", None), ("prediction_up", "Pred U", None),
        ("accuracy", "Accuracy", "pct"), ("balanced_accuracy", "Balanced acc.", "pct"), ("macro_f1", "Macro F1", "pct"),
        ("down_precision", "D P", "pct"), ("down_recall", "D R", "pct"),
        ("up_precision", "U P", "pct"), ("up_recall", "U R", "pct"),
        ("log_loss", "Log loss", "float"), ("confusion_matrix_actual_rows_predicted_columns_down_up", "Confusion [D,U]", None),
    ])

    validation = validation[validation.model.isin(["causal_1d_cnn", "random_forest", "xgboost"])].sort_values(["sequence_length", "model"]).copy()
    parsed_classes = validation.class_metrics.map(ast.literal_eval)
    for cls in ("DOWN", "UP"):
        validation[f"{cls.lower()}_precision"] = parsed_classes.map(lambda x: x[cls]["precision"])
        validation[f"{cls.lower()}_recall"] = parsed_classes.map(lambda x: x[cls]["recall"])
    validation_table = md_table(validation, [
        ("model", "Model", None), ("sequence_length", "L", None), ("n_directional", "Directional n", None),
        ("sequence_eligibility_coverage_of_directional_rows", "Coverage", "pct"), ("accuracy", "Accuracy", "pct"),
        ("balanced_accuracy", "Balanced acc.", "pct"), ("macro_f1", "Macro F1", "pct"),
        ("down_precision", "DOWN P", "pct"), ("down_recall", "DOWN R", "pct"),
        ("up_precision", "UP P", "pct"), ("up_recall", "UP R", "pct"),
        ("directional_macro_precision", "Macro precision", "pct"), ("log_loss", "Log loss", "float"),
        ("confusion_matrix_actual_rows_predicted_columns_down_up", "Confusion [D,U]", None),
    ])

    date_table = md_table(date_summary[date_summary.model == "causal_1d_cnn"].sort_values("sequence_length"), [
        ("sequence_length", "CNN L", None), ("dates_with_at_least_10_directional", "Dates n≥10", None),
        ("median_daily_accuracy", "Median daily accuracy", "pct"),
        ("fraction_dates_accuracy_gt_50", "Dates >50%", "pct"),
        ("fraction_dates_accuracy_gt_55", "Dates >55%", "pct"),
    ])

    text = f"""# ECL Alpaca IEX OHLCV Sequence Research v1

## Decision

**No sequence candidate achieved reliable out-of-sample accuracy above 55%. Do not promote or integrate a sequence model.** The highest mean fold accuracy was 54.3% for a 12-candle CNN, but its mean fold balanced accuracy was 50.5% and macro-F1 was 38.4%. It predicted UP only 70 times in 1,188 OOF directional cases. On validation it predicted DOWN for every case at lengths 6 and 12. Longer windows were more balanced in some folds but did not produce stable accuracy above 55% and substantially reduced eligible coverage.

Chronological OOF predictions are out-of-sample within the training period. The existing validation period has prior research exposure and is **not a pristine holdout**. No protected test rows, predictions or metrics were accessed. This experiment does not establish independent future performance.

## Data and eligibility

Source: `models/alpaca_1min_research_v1/raw_ecl_iex_5min_train_validation.csv`, whose metadata specifies ECL, Alpaca IEX, 5Min, November 13, 2025–September 24, 2026. It contains {len(raw):,} bars over {raw.session_date.nunique()} New York dates; duplicate timestamps: {int(raw.timestamp_utc.duplicated().sum())}; invalid OHLCV rows: {int(invalid.sum())}; zero-volume rows: {int((raw.volume == 0).sum())}. True missing intervals were kept as gaps and reset the sequence. See `source_data_audit.json`.

The experiment reuses the approved train/validation labels. It verifies that decision and future endpoint candles exist in the permitted source, the endpoint is exactly 15 minutes later in the same New York session date, and the saved `future_return` maps to the strict existing ±0.10% labels. NEUTRAL rows are excluded from binary fitting and directional metrics.

| Split | All labels | Sequence length | Sequence rows | Directional total | Directional eligible | Coverage | Eligible dates |
|---|---:|---:|---:|---:|---:|---:|---:|
{eligibility_table.splitlines()[2:] and chr(10).join(eligibility_table.splitlines()[2:])}

L6/L12 retained all original labels. L24 retained 87.7% of validation directional samples; L36 retained 53.3%. Every input window ends at its decision candle and requires exact consecutive 5-minute timestamps within the same New York session. No gap is filled or bridged. Volume is the IEX-feed volume only and must not be interpreted as consolidated market-wide volume.

## Input and model

The learned input is a fixed rolling window of raw OHLCV, not indicator formulas: OHLC is expressed as percent distance from the window's final close; volume uses `log1p`. Channel scaling is fitted only on training windows within each fold. The compact CNN is causal Conv1D(8 filters, kernel 3), global average pooling, a regularized 8-unit dense layer, dropout, and sigmoid output. Settings are fixed at 20 epochs, batch 32, Adam 0.001, no class weighting, and no threshold search. UP is predicted when P(UP) ≥ 0.5; otherwise DOWN.

Four established expanding chronological folds were reused. Training examples whose 15-minute endpoint reached the first held-out timestamp were purged. RF and XGBoost predictions saved from the existing v3 experiment were joined by exact timestamp for matched comparisons; they were not re-inferred here. The v3 baseline configuration had been selected using the aggregate of those same OOF folds, so baseline fold results are a descriptive reference rather than a nested, configuration-selection-unbiased comparison. The validation baseline configuration was selected using training OOF only.

## Matched OOF comparison

This table aggregates the four OOF folds on identical eligible timestamps per length. Complete per-fold rows for the CNN and both baselines are saved in `walk_forward_fold_metrics.csv`; overall metrics are in `walk_forward_overall_metrics.csv`.

{comparison}

The CNN's accuracy did not translate to balanced direction prediction. At L6/L12, UP recall was especially weak (0.4%/6.3% pooled OOF), while XGBoost was better on balanced accuracy and macro-F1 on the same rows. L24 remained behind XGBoost on balanced accuracy and macro-F1. L36 had only about 57.7% OOF eligibility coverage and unstable folds.

## CNN fold results

Confusion matrices use actual rows and predicted columns in DOWN, UP order.

{fold_table}

The later folds show substantial instability. In particular, a DOWN-heavy fold can make a model that predicts mostly DOWN look accurate; fold 4 drops below 50% for shorter windows and to 42.4% for L36. No length produces consistent >55% accuracy.

## Validation confirmation (limited diagnostic)

Each predeclared length was trained on train only and evaluated once on the existing validation set. The validation period was not used to change architecture, length, preprocessing or threshold. Saved RF/XGBoost predictions were aligned on the same timestamp subsets.

{validation_table}

CNN validation accuracy was 51.5%, 51.5%, 52.3% and 49.5% for lengths 6, 12, 24 and 36. L6/L12 predicted all DOWN; L24 predicted UP only 7 times among 472 directional cases; L36 had 53.3% sequence eligibility and only 9.7% UP recall. Matched Random Forest validation accuracy was higher at each length. All-candle operational accuracy, where neutral actuals count as wrong, was 23.9%, 23.9%, 23.8% and 21.9% for the CNN.

## Date stability

Daily statistics include dates with at least 10 directional labels; the full date-level details are in `date_stability_metrics.csv`.

{date_table}

Daily medians are not enough to establish an edge: minima and maxima vary widely, and fold-level performance remains unstable.

## Reproduction and safeguards

Run from the project root to reproduce the training and evaluation, then rebuild the narrative report from its saved outputs:

```powershell
.\\.venv\\Scripts\\python.exe models\\alpaca_sequence_research_v1\\run_sequence_research.py
.\\.venv\\Scripts\\python.exe models\\alpaca_sequence_research_v1\\build_research_report.py
```

The script allowlists only the pre-test raw cache and metadata, train/validation splits, established fold schedule, and saved v3 baseline prediction outputs. It rejects other input paths. It makes no network calls, does not read `test.csv` or post-September 24 data, and does not access or modify the live application or frozen production model.

Focused tests: **8 passed**. Syntax compilation passed. TensorFlow {json.loads((MODEL_DIR / 'reproducibility_manifest.json').read_text()).get('tensorflow_version', 'version recorded in manifest')} was available; a small synthetic CNN trained, saved and reloaded with matching probabilities. Six Keras/NumPy deprecation warnings occurred in tests, and TensorFlow emitted a graph-attribute compatibility warning during fitting. The installed TensorFlow build is a release candidate, a reproducibility limitation. The full project suite was not run.

No model is selected or promoted. The existing frozen three-class model remains unchanged. No profitability or live performance is claimed.
"""
    report.write_text(text, encoding="utf-8")
    (MODEL_DIR / "README.md").write_text(
        "# ECL OHLCV Sequence Research v1\n\n"
        "Isolated Alpaca IEX raw 5-minute OHLCV sequence research using the existing 15-minute ±0.10% labels. "
        "Compares 6/12/24/36 candle causal CNN inputs with saved Random Forest and XGBoost predictions on matched timestamps. "
        "No test data or post-2026-09-24 observations are used, and no live or frozen model files are changed.\n\n"
        "From project root, run the experiment and then build its report: ` .\\.venv\\Scripts\\python.exe models\\alpaca_sequence_research_v1\\run_sequence_research.py ` followed by ` .\\.venv\\Scripts\\python.exe models\\alpaca_sequence_research_v1\\build_research_report.py `.\n\n"
        "Results: `reports/alpaca_sequence_research_v1/research_report.md`. No sequence model met the objective; none is promoted.\n",
        encoding="utf-8",
    )

    manifest_path = MODEL_DIR / "reproducibility_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    approved_inputs = [PROJECT / path for path in manifest["source_constraints"]["approved_input_paths"]]
    manifest["input_sha256"] = {str(p.relative_to(PROJECT)): digest(p) for p in approved_inputs}
    artifact_paths = [
        MODEL_DIR / name for name in (
            "configuration.json", "feature_schema.json", "fold_training_config.json",
            "run_sequence_research.py", "build_research_report.py", "test_sequence_research.py", "README.md",
            "causal_1d_cnn_len6.keras", "causal_1d_cnn_len12.keras", "causal_1d_cnn_len24.keras", "causal_1d_cnn_len36.keras",
            "channel_scaler_len6.joblib", "channel_scaler_len12.joblib", "channel_scaler_len24.joblib", "channel_scaler_len36.joblib",
        )
    ]
    manifest["artifact_sha256"] = {str(p.relative_to(PROJECT)): digest(p) for p in artifact_paths}
    manifest["report_outputs"] = [str(p.relative_to(PROJECT)) for p in sorted(REPORT_DIR.iterdir()) if p.is_file()]
    manifest["research_report_sha256"] = digest(report)
    manifest["source_data_audit_sha256"] = digest(REPORT_DIR / "source_data_audit.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
