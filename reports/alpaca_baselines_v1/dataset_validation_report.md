# Alpaca IEX dataset validation (pre-training)

The old training/validation/test CSVs were not used. The protected old test CSV was not opened. The new test split was written from the canonical source but its labels were withheld from this pre-training report.

Canonical candles: **18,789** (2025-11-12 11:25:00 through 2026-11-12 15:55:00 New York time).
Target-ready, feature-ready rows before date splits: **7,413**.
Source rows rejected from canonical: **1**; all audited source files include **2** rejected observations because the same invalid row appears in the verification cache.
Missing expected NYSE five-minute slots: **795**. These are preserved gaps, not filled candles.
Feature warm-up/nonfinite exclusions: **10,506**; rows without an exact valid 15-minute endpoint: **2,438** (these exclusion groups can overlap).

## Split validation

| Split | Rows | Trading dates | Date range | Missing/nonfinite feature cells | Class counts (DOWN / NEUTRAL / UP) |
|---|---:|---:|---|---:|---|
| train | 4,883 | 154 | 2025-11-13–2026-08-05 | 0 | 1323 / 2364 / 1196 |
| validation | 1,161 | 33 | 2026-08-06–2026-09-24 | 0 | 277 / 623 / 261 |
| test (summary withheld) | 1,369 | 34 | 2026-09-25–2026-11-12 | validated in builder memory; no class summary emitted | withheld until final evaluation |

All splits use the exact same 24 frozen feature columns and 15-minute target. Purged rows: train 0, validation 0.
The one out-of-session row, 2026-09-18 09:25 ET, was rejected. No missing candle was fabricated. Feature generation reused the existing causal feature implementation in the frozen feature order.
