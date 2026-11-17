# ECL Dashboard (Alpaca IEX)

The active dashboard uses Alpaca IEX only. It combines the permitted Alpaca historical cache with persisted live Alpaca IEX candles on one timeline.

## Run locally

From `D:\ML_project`, configure `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in `.env`. Set `ALPACA_IEX_DISPLAY_AUTHORIZED=true` only when your account permissions allow private local display. The default is false. The gate controls display/stream startup; it never selects a different provider.

```powershell
.\.venv\Scripts\python.exe -m uvicorn live_app.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`; press Ctrl+C to stop. Keep the app on `127.0.0.1` and do not use public hosting or a tunnel.

## Data and prediction flow

The Alpaca IEX WebSocket subscribes to ECL minute bars. Valid observed minute bars are persisted in the separate Alpaca SQLite database and aggregated into NYSE-session-aligned, finalized 5-minute candles after the configured buffer. Missing minutes are not filled. Only eligible finalized candles reach the frozen model, which uses the exact 24-feature order, DOWN/NEUTRAL/UP class order, 15-minute horizon and raw argmax policy. Predictions and settled outcomes are immutable.

Historical Replay and saved historical Model Indicator signals are retrospective and separate from the live prediction ledger. The dashboard does not rerun inference when a candle is selected.

## Limitations

Alpaca IEX represents IEX venue observations rather than consolidated market data. Missing IEX bars are not evidence of market-wide inactivity. Provider availability, permissions, quotas and connectivity can affect live updates. The display authorization setting is separate from provider selection; there is no fallback provider.
