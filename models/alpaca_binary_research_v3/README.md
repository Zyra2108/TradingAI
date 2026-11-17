# Binary UP/DOWN research v3

This is an isolated research experiment. It uses the Alpaca IEX pre-test training and validation splits only; no final test data is accessed. NEUTRAL examples are excluded from binary training and directional-only metrics, but the classifier still emits UP or DOWN for every row. A low-confidence flag is informational and does not abstain.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe models\alpaca_binary_research_v3\run_binary_research.py
```

The design, model settings, folds, guards and seeds are in `configuration.json`. The research conclusion and fold/validation reports are in `reports/alpaca_binary_research_v3/research_report.md`. No candidate met the predeclared validation guardrails; none is promoted or integrated.
