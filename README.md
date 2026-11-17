# ECL AI Trading Indicator

A local student research and paper-trading dashboard for Ecolab Inc. (ECL). The active market-data architecture uses **Alpaca IEX exclusively** for historical and live ECL data. The application never places orders.

## Architecture

- Historical ECL 5-minute OHLCV: Alpaca IEX REST cache and manifests under `reports/alpaca_iex_1y_audit_v1/`, `models/alpaca_1min_research_v1/`, and `data/alpaca_backfill_v1/`.
- Live data: Alpaca IEX minute-bar WebSocket, persisted in `data/live/ecl_alpaca_iex.sqlite3`, aggregated to finalized 5-minute candles, then passed to the existing feature and prediction pipeline.
- Model: integrity-checked frozen candidate at `models/alpaca_final_freeze_v1/candidate_pipeline.joblib`.
- Prediction: completed 5-minute candles, 15-minute horizon, DOWN / NEUTRAL / UP, frozen raw argmax policy and immutable prediction/outcome records.
- Dashboard: FastAPI bound to loopback, unified historical/live chart and read-only Historical Replay.

## Setup on Windows

1. Create and activate a virtual environment, then install the pinned dependencies:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`, then fill `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` locally. Keep credentials private. `ALPACA_IEX_DISPLAY_AUTHORIZED` defaults to `false`; enable private local display only when your account permissions allow it.
3. Start the dashboard from the project root:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn live_app.app:app --host 127.0.0.1 --port 8000
   ```

4. Open `http://127.0.0.1:8000`. Stop with Ctrl+C. Do not bind to `0.0.0.0`, use a tunnel, or expose the app publicly.

When display authorization is off, the app remains in Alpaca mode and reports that the gate is disabled. It does not select another provider. Missing credentials or stream failures appear as explicit status errors; they do not trigger provider fallback.

## Historical Replay and indicators

The unified timeline uses only available cached Alpaca IEX candles and live persisted Alpaca candles. The chart indicates its actual available range and preserves missing intervals. Model Indicator markers come from saved historical-backtest or genuine live prediction records; selecting candles does not rerun inference. SMA 20 is a separate indicator based on 20 consecutive valid closes.

Historical Replay uses permitted pre-test cached data and the frozen candidate. It does not write to the live prediction ledger. A missing saved prediction is shown as such rather than generated retrospectively.

## Data-source limitations

IEX is a venue-specific feed, not consolidated-market coverage. An absent IEX minute or 5-minute bar does not prove the overall market had no trades. No prices are filled or fabricated. Sparse observed-minute aggregation has passed the existing native IEX 5-minute parity audit for its audited pre-test period; later periods require their own overlap checks before data is merged or used for training.

The reproducible native 1-minute and 5-minute research fetches cover Nov 13, 2025 through Sep 24, 2026. Additional backfill exists for Sep 25 through Nov 12, 2026. The one-year source is retained, but its original acquisition script is not present; consult its manifest and audit before relying on it as a reproducible canonical training source.

## Alpaca-only baseline rebuild (research)

The isolated `alpaca_ml_pipeline_v1` dataset was assembled from existing Alpaca IEX 5-minute files only. The one-year audit file is the primary source; the separately fetched native 5-minute cache is used to verify overlap; the Sep 25–Nov 12, 2026 Alpaca IEX backfill contributes only non-overlapping timestamps. Conflicting overlapping OHLCV records stop the builder. Missing IEX candles remain missing. Dataset lineage, hashes, split rules and quality checks are saved in `data/alpaca_ml_pipeline_v1/reproducibility_manifest.json` and `quality_report.json`.

Features use the frozen 24-column order and the existing causal project feature functions. Labels compare the decision candle close with the exact close 15 minutes later in the same continuous session segment: UP above +0.10%, DOWN below -0.10%, NEUTRAL otherwise. Chronological dates are divided 70/15/15 with target-horizon purging. The generated test split is kept separate from training and validation.

To reproduce the dataset and fixed baseline comparison from the project root:

```powershell
.\.venv\Scripts\python.exe data\alpaca_ml_pipeline_v1\build_dataset.py
.\.venv\Scripts\python.exe -m pytest data\alpaca_ml_pipeline_v1\test_alpaca_ml_pipeline.py models\alpaca_baselines_v1\test_metrics.py -q
.\.venv\Scripts\python.exe models\alpaca_baselines_v1\train_compare.py
```

The baseline comparison is Logistic Regression, Random Forest, XGBoost and HistGradientBoosting, plus always-neutral and a fixed RSI reference rule. Validation is used for model selection; it has appeared in previous research and is not a pristine holdout. After selection is frozen, the new test split is evaluated exactly once with:

```powershell
.\.venv\Scripts\python.exe models\alpaca_baselines_v1\final_evaluate_selected.py
```

Dataset files are in `data/alpaca_ml_pipeline_v1/`; model artifacts and configuration are in `models/alpaca_baselines_v1/`; comparisons and final evaluation are in `reports/alpaca_baselines_v1/`. This research candidate does not replace or modify the live model in `models/alpaca_final_freeze_v1/`.

The current rebuilt split has 4,883 training rows (Nov 13, 2025–Aug 5, 2026), 1,161 validation rows (Aug 6–Sep 24, 2026), and 1,369 new test rows (Sep 25–Nov 12, 2026). On validation, XGBoost ranked first by the predeclared macro-F1 / balanced-accuracy rule (macro-F1 0.362, balanced accuracy 0.366); its accuracy was 0.403 versus 0.537 for always-neutral. On the one-time new test evaluation, XGBoost scored 0.386 accuracy, 0.362 balanced accuracy and 0.356 macro-F1; always-neutral accuracy was 0.508. These results do not justify replacing the live model. The rebuilt test dates overlap dates used in earlier project test research, and its class distribution was emitted during initial dataset QA; the rebuilt test labels were not used for fitting or model selection, but this is not a pristine project-level holdout.

## Research decision

The frozen Alpaca three-class model remains operational. The binary research candidates did not meet the stated precision, accuracy and meaningful-coverage goals consistently, so none was promoted. Historical classification scores are not evidence of trading profitability. Live end-to-end operation requires a real open-market prediction and subsequent settlement trial; synthetic tests do not establish live readiness.

## Data preprocessing images (pre-test source only)

To regenerate the before/after preprocessing charts from the approved Alpaca IEX 5-minute training/validation cache, open PowerShell in the project root and run:

```powershell
.\.venv\Scripts\python.exe scripts\visualize_alpaca_preprocessing.py
```

The script verifies the cache hash and its ECL/IEX/5-minute provenance, and refuses dates outside Nov 13, 2025–Sep 24, 2026. It does not access test data, train a model, or alter any dataset. It writes `before_after_preprocessing.png`, `volume_and_return_distributions.png`, and `preprocessing_visual_report.json` under `reports/data_preprocessing_visuals_v1/`. Missing market intervals remain gaps; the plots do not fill them. To write to a different new folder, pass `--output reports/my_preprocessing_run`.

## Safety

This is an educational paper-trading indicator, not financial advice. It does not submit orders. Keep API credentials in the local `.env`, do not commit that file, and keep the server loopback-only.
