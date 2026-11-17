# ECL OHLCV Sequence Research v1

Isolated Alpaca IEX raw 5-minute OHLCV sequence research using the existing 15-minute ±0.10% labels. Compares 6/12/24/36 candle causal CNN inputs with saved Random Forest and XGBoost predictions on matched timestamps. No test data or post-2026-09-24 observations are used, and no live or frozen model files are changed.

From project root, run the experiment and then build its report: ` .\.venv\Scripts\python.exe models\alpaca_sequence_research_v1\run_sequence_research.py ` followed by ` .\.venv\Scripts\python.exe models\alpaca_sequence_research_v1\build_research_report.py `.

Results: `reports/alpaca_sequence_research_v1/research_report.md`. No sequence model met the objective; none is promoted.
