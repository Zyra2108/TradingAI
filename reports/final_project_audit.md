# ECL project audit after provider cleanup

## Active architecture

The active application is Alpaca IEX-only. `live_app.app:create_app` always constructs or accepts `AlpacaPredictionEngine`; authorization no longer selects a provider. The separate `ALPACA_IEX_DISPLAY_AUTHORIZED` gate remains false by default. If credentials or the stream are unavailable, the status API reports that condition; the application has no provider fallback.

The dashboard uses `models/alpaca_final_freeze_v1/candidate_pipeline.joblib`. Its six frozen artifact hashes match `SHA256SUMS.json`. The live path remains ECL IEX minute bars → persisted observations → finalized five-minute candles → 24 ordered features → immutable 15-minute predictions and settlements.

The local `.env` no longer contains the retired provider key entry. Alpaca environment-variable names remain present; values were not intentionally printed or included in this report. The first broad repository search inadvertently emitted the obsolete key line; treat that retired key as exposed and revoke it if it remains valid.

## Cleanup and archive

Clearly legacy provider modules, data products, model/research artifacts, collection scripts, old dashboard assets and related reports/tests were backed up and moved out of active paths. The external archive and exact inventory are at:

`C:\Users\sasi stuthika\AppData\Local\ECL_AlpacaOnlyCleanupBackup\20261116`

Alpaca datasets, rebuild artifacts, research, backfill, provider, aggregation, replay, model, database, and Alpaca-specific reports remain in place. Old ZIP archives, existing backups and some historical Alpaca manifests/reports were retained because they are archival or have uncertain provenance; they are not active imports or provider-selection paths.

## Verification

- Python syntax compilation for the active configuration, FastAPI app, Alpaca pipeline, historical feature utilities, dataset builder and Alpaca freeze audit script: passed.
- Focused Alpaca/application/replay suite: **152 passed**, 3 warnings.
- Full project pytest suite: **222 passed**, 10 warnings.
- Chart JavaScript test: **16 passed**, 0 failed.
- Frozen artifact hash verification: **6/6 match**.
- No live server, Alpaca connection, data download, model training or inference was started for this cleanup.

## Protected test-data limitation

`data/alpaca_rebuild_v1/test.csv` was not deliberately opened with a CSV reader, copied, hashed, modified or deleted. However, the initial broad repository text search did not exclude that CSV and may have traversed it. Therefore this cleanup cannot claim strict zero access to its contents. No test results were used for changes.
