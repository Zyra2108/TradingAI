# Final Integrated Live-Market Verification — ECL Alpaca IEX

Trial date: **2026-11-13**  
Capture window: **11:06–11:37 America/New_York** (16:06–16:37 UTC), stopped early after a deterministic pipeline blocker was established.  
Result: **INCOMPLETE — no finalized candle, prediction, or settled outcome. This is not a successful end-to-end live verification.**

## Preflight and active model

| Check | Result | Evidence |
|---|---|---|
| Regular NYSE market open | PASS | Market calendar reported 09:30–16:00 New York; capture began at 11:06 ET. |
| Alpaca IEX / ECL / 5-minute application | PASS | `/api/health` and `/api/status` returned HTTP 200 in `alpaca_iex` mode, ECL/NYSE, IEX feed. |
| User display authorization | PASS | Effective setting was true; value was not changed. |
| Credentials configured | PASS | Both required credential variables were present; no values were displayed. |
| Local-only server | PASS | Exactly one server was started after stopping the verified prior project server tree; listener bound to `127.0.0.1:8000`. |
| Frozen model integrity | PASS | `models/alpaca_final_freeze_v1/candidate_pipeline.joblib`; SHA256 `31188a09eba09d58d441324fcfb130a23268a833a459709e4c0c6ad9ca45d08d`. |
| Model contract | PASS | 24 frozen features in exact order; classes DOWN/NEUTRAL/UP; 15-minute horizon; raw three-class argmax. |
| Separate Alpaca database | PASS | `data/live/ecl_alpaca_iex.sqlite3`; verification runner queried it read-only. |

No other Alpaca market-data client was found running on this machine at preflight. The earlier server process tree was verified by command line and stopped; port 8000 was confirmed free before starting one server.

## Real integrated stream observations

The application received the Alpaca `connected`, `authenticated`, and ECL bars subscription acknowledgements. During the capture, `/api/status` repeatedly reported connected/authenticated/subscribed and the last event receive delay was about 60 seconds. There were **22 unique new ECL IEX minute bars** persisted during the capture. Natural reconnect count increased from 1 to 3; after those reconnects, the stream was again active. No manual reconnect was forced.

The dashboard HTML, CSS, and JavaScript assets all returned HTTP 200. Automated polling of `/api/status`, `/api/candles`, and `/api/predictions` returned successfully. The browser automation surface was unavailable, so visual chart/label rendering remains **UNVERIFIED** and requires a manual browser check.

## Aggregation blocker and database evidence

The database contained a fully continuous real-minute sequence at **16:15, 16:16, 16:17, 16:18, and 16:19 UTC** (11:15–11:19 ET). Yet the live database recorded **0 finalized five-minute candles, 0 predictions, and 0 outcomes**. At capture start the dashboard had 0 prior candles and 0 prior predictions; no existing prediction/outcome records were deleted or changed. No synthetic records were inserted.

The cause is in the integration lifecycle: `AlpacaFiveMinuteAggregator.ingest()` tries to finalize only the bucket containing the minute just received. The last constituent minute is received before the configured 120-second finalization buffer expires. Later minute messages belong to the next five-minute bucket, so they do not retry the now-due previous bucket. `recover_aggregations()` runs at application startup only; the periodic maintenance task settles predictions but does not revisit due aggregation buckets. Thus even the complete 16:15–16:19 sequence remained unfinalized. Without a finalized candle, no feature row or model inference can occur, and no 15-minute outcome can later settle.

The status continued to show an older safe `connection_failed` warning while `connected`, `authenticated`, and `subscribed` were true after reconnection. The feed was active; the warning was stale status metadata, not evidence of current authentication rejection.

## Verification checklist

| Stage | Result |
|---|---|
| Server startup and localhost-only bind | PASS |
| Frozen model integrity / 24-feature contract | PASS |
| Alpaca WebSocket connection | PASS |
| Authentication | PASS |
| ECL IEX bar subscription | PASS |
| Real minute events persisted | PASS — 22 unique new bars |
| Natural reconnect observed | PASS — reconnect count rose by 2 |
| Five-minute aggregation | FAIL — complete sequence persisted but not finalized |
| Feature generation | NOT REACHED |
| Frozen-model prediction | NOT REACHED |
| Prediction timestamp/reference-price persistence | NOT REACHED |
| 15-minute outcome and settlement | NOT REACHED |
| SQLite read-only persistence audit | PASS — minute rows verified; prediction/outcome counts remained zero |
| Dashboard API/assets | PASS — localhost endpoints and static assets responded |
| Dashboard visual rendering | UNVERIFIED — manual browser check still needed |
| Missing-data recovery | PARTIAL — reconnects observed; no finalized aggregation-gap record was produced in this capture |

## Capture evidence

Evidence is stored locally in `reports/live_market_verification_runs/20261113T160612Z/`:

- `observations.jsonl` — safe status snapshots and capture start metadata; no credentials or raw provider payloads.
- `summary.json` — final counts, model identity, reconnect evidence, and the aggregation blocker.

The runner was intentionally stopped early once code-path inspection and the persisted complete minute sequence established that the current runtime would not finalize it. It did not run the full 60 minutes. The server was then shut down cleanly. No live trial success is claimed.

## Required before another live trial

The periodic due-bucket scheduling fix and its regression tests have since been implemented (159 project tests passed). The follow-up trial documented below did not verify the fix end to end because Alpaca returned a 406 connection-limit error before a complete bucket was finalized. Resolve the connection-limit condition and repeat an open-market integrated trial; preserve strict five-minute continuity and idempotency. A complete frozen-model prediction and subsequent settled outcome are still required before claiming live verification.

## Runner launch note

The first confirmed runner invocation exposed an import-path defect (`config` was not importable when launched as a script). The runner was updated to add the project root to `sys.path`; its 4 focused tests passed after the correction. The confirmed capture then ran and produced the evidence above.


## Tests and integrity

- Verification runner focused tests after the import-path correction: **4 passed**.
- Full project suite after the runner change and live capture: **155 passed, 7 warnings** (182.18 seconds).
- Frozen model hash remained the loader-verified hash listed above; no model files were changed.
- The verification script uses no credentials in output, wrote no prediction/candle/outcome data itself, and performed only read-only database checks. The application persisted the real IEX minute bars observed during the run.
- No historical or test dataset was accessed for this verification.

The runner fix changed `scripts/live_market_verification.py` so the exact requested invocation can import root `config.py`. Capture evidence and the report were created under the dedicated verification paths. The prior test suite result of 155 passed remains confirmed by this final run.

## Follow-up integrated live trial — 2026-11-13

This follow-up occurred after the periodic aggregation fix. **It did not complete an end-to-end prediction and settlement cycle.** The requested 60-minute capture stopped early when Alpaca returned WebSocket status **406**, classified by the app as `connection_limit_exceeded`.

### Preflight and startup

- The NYSE regular session was open (09:30–16:00 America/New_York); capture began at 11:57:43 ET.
- Alpaca IEX mode, ECL/NYSE, configured credentials, and the user's existing display authorization were present/enabled. No credential values were printed or changed.
- The frozen candidate integrity check passed: `models/alpaca_final_freeze_v1/candidate_pipeline.joblib`, SHA256 `31188a09eba09d58d441324fcfb130a23268a833a459709e4c0c6ad9ca45d08d`; 24 features, DOWN/NEUTRAL/UP class order, 15-minute horizon, raw three-class argmax.
- Port 8000 was free and no local Python project server/verification process was found before startup. Exactly one server was started on `127.0.0.1:8000`; health/status and dashboard assets responded successfully.
- Initial provider status was connected, authenticated, and subscribed to ECL bars. Reconnect count moved from 3 to 4 during this attempt.

### Capture evidence and outcome

- Capture ran from **11:57:43 to 12:24:35 ET** (16:57:43–17:24:35 UTC), approximately **26 minutes 52 seconds**, before being stopped after the 406 connection-limit response.
- **20** unique real Alpaca IEX one-minute bars were persisted, with event timestamps spanning 11:57–12:20 ET.
- **0** new five-minute candles were finalized. Five aggregation gap records were detected; incomplete minute sequences remained rejected.
- **0** new model predictions and **0** settled outcomes were produced. Prediction generation, reference-price persistence, and settlement therefore remain unverified in this live attempt.
- At the end, provider state was `connection_limit_exceeded` (status 406), disconnected, unauthenticated, and unsubscribed. The app server was stopped cleanly. No retries or bypasses were attempted after the blocker, and no synthetic rows were inserted.
- Dashboard HTML/CSS/JavaScript and API polling were verified through the runner; visual browser rendering was not checked.

The initial process check found no competing local Python client, but the evidence does not identify whether the 406 came from another account connection outside this machine, a provider-side connection slot that remained occupied, or another cause. Do not treat it as a credential failure based on this evidence alone.

Evidence is saved under `reports/live_market_verification_runs/20261113T165742Z/`. `observations.jsonl` contains safe app status observations; `summary.json` records the incomplete trial and read-only persistence counts. The runner was interrupted after the explicit 406 blocker, so `summary.json` was reconstructed from those observations and read-only SQLite queries. The app itself persisted only real provider bars.

**Conclusion:** the periodic finalization change has not yet been shown to produce a complete live prediction and outcome. A future trial requires resolving the account-level connection-limit condition and then observing a complete five-minute source sequence; the strict gap checks must remain in place. No live verification success is claimed.
