# Alpaca ECL dataset rebuild audit v1

No models were trained and no historical test results were accessed.

Source SHA-256: `757b35e5426426ed5840282b9817f888664789ffe5f50f17898f8178ed1bd4c2`
Source candles: 18,735; duplicates removed 0; invalid OHLC removed 0; outside regular session removed 1.
Regular-session candles: 18,734; zero-volume values marked missing: 0.
Rows entering feature/label readiness filtering: 18,734; usable feature-ready and labeled rows before split purge: 7,358; exact 15-minute target integrity checks passed: 7,358.

## Splits

| Split | Rows | Trading dates | Date range | DOWN | NEUTRAL | UP | Missing feature cells | Purged |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| train | 4,883 | 154 | 2025-11-13–2026-08-05 | 1,323 (27.09%) | 2,364 (48.41%) | 1,196 (24.49%) | 0 | 0 |
| validation | 1,161 | 33 | 2026-08-06–2026-09-24 | 277 (23.86%) | 623 (53.66%) | 261 (22.48%) | 0 | 0 |
| test | 1,314 | 33 | 2026-09-25–2026-11-11 | 319 (24.28%) | 660 (50.23%) | 335 (25.49%) | 0 | 0 |

## Checks

- Feature input order equals frozen B's 24-column specification; timestamp, target and future-return metadata are excluded from model inputs.
- Features use the established causal historical calculations; volume is not used.
- Every retained label passed exact +15-minute timestamp, within-session, continuous 5-minute sequence, close-derived return and threshold checks.
- Missing candles are not synthesized. Rolling features reset on session/gap segments.
- Feature values contain no missing or infinite cells in retained split rows; per-feature dtypes are recorded in the manifest.
- Split boundaries are by whole trading dates, with explicit target-horizon purge.
- Old Twelve Data comparison (if present) uses only the old prepared training split's target column; descriptive only, no old validation/test metrics.
- Test CSV is created once from the in-memory split. No training, model selection or backtest was performed.

## Artifacts

- `train.csv`, `validation.csv`, `test.csv`: each includes datetime/future target audit metadata and exactly 24 B features.
- `class_distributions.csv`: per-split class counts and percentages.
- `reproducibility_manifest.json`: source hash, definitions, split boundaries, purge rules and audits.
- `build_alpaca_dataset.py`: reproducible builder; refuses to overwrite a nonempty output directory.

