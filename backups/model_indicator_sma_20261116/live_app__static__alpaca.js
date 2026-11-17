const root = document.getElementById('chart');
const chartState = document.getElementById('chart-state');
const el = id => document.getElementById(id);
const VIEWPORT = window.ECLChartViewport;
const ny = value => value ? new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', dateStyle: 'medium', timeStyle: 'medium', hour12: false }).format(new Date(value)) : '—';
const nyEpoch = value => ny(Number(value) * 1000);
const dollars = value => value == null ? '—' : `$${Number(value).toFixed(2)}`;
const quantity = value => value == null ? '—' : Number(value).toLocaleString('en-US', { maximumFractionDigits: 0 });
const pct = value => value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`;
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let chart = null, candleSeries = null, indicatorSeries = null;
let timeline = [], candleByTime = new Map(), indicatorByTime = new Map();
let predictions = new Map(), selectedTime = null, viewportInitialized = false, previousCount = 0;
let historyLoaded = false, latestLiveTime = null, detailRequest = 0, refreshBusy = false, marketCoverage = null, marketIsOpen = false;
let latestPredictionLoaded = false;

function visibleAutoscaleInfoProvider(original) {
  const range = chart?.timeScale().getVisibleLogicalRange();
  const priceRange = VIEWPORT?.visiblePriceRange(timeline, range, indicatorByTime, el('indicator-toggle')?.checked !== false);
  if (!priceRange) return original();
  return { priceRange, margins: { above: 0, below: 0 } };
}

if (window.LightweightCharts?.createChart) {
  chart = LightweightCharts.createChart(root, {
    width: Math.max(1, root.clientWidth), height: Math.max(1, root.clientHeight),
    layout: { background: { color: '#0e1723' }, textColor: '#91a4ba' },
    grid: { vertLines: { color: '#1a2838' }, horzLines: { color: '#1a2838' } },
    rightPriceScale: { borderColor: '#29394c', autoScale: true, scaleMargins: { top: 0.08, bottom: 0.08 } },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 2,
      barSpacing: VIEWPORT?.candleSpacing(root.clientWidth - 70, 65) || 10,
      tickMarkFormatter: t => new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(Number(t) * 1000)) },
    localization: { timeFormatter: t => nyEpoch(t) }
  });
  candleSeries = chart.addCandlestickSeries({ upColor: '#37d6a0', downColor: '#fb7185', borderVisible: false, wickUpColor: '#37d6a0', wickDownColor: '#fb7185', autoscaleInfoProvider: visibleAutoscaleInfoProvider });
  indicatorSeries = chart.addLineSeries({ color: '#73b5ff', lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: true });
  if (window.ResizeObserver) new ResizeObserver(() => chart.applyOptions({ width: Math.max(1, root.clientWidth), height: Math.max(1, root.clientHeight) })).observe(root);
  chart.subscribeCrosshairMove(param => {
    const t = param?.time;
    el('crosshair-time').textContent = typeof t === 'number' ? nyEpoch(t) : 'Move over chart';
  });
  chart.subscribeClick(param => {
    if (!param?.time || typeof param.time !== 'number') return;
    if (param.seriesData?.get(candleSeries)) selectCandle(param.time);
  });
} else {
  chartState.textContent = 'Candlestick chart unavailable: the chart library could not be loaded.';
  chartState.classList.remove('hidden');
}

function renderIndicator() {
  const enabled = el('indicator-toggle').checked;
  el('indicator-state').textContent = enabled ? 'ON' : 'OFF';
  if (indicatorSeries) indicatorSeries.setData(VIEWPORT.indicatorData(timeline, 20, enabled));
  renderIndicatorExplanation();
}
function renderIndicatorExplanation() {
  const note = el('indicator-explanation');
  if (!note) return;
  if (!el('indicator-toggle').checked) { note.textContent = 'SMA 20 hidden; its calculation is unchanged.'; return; }
  const candle = candleByTime.get(Number(selectedTime)) || timeline.at(-1);
  if (!candle) { note.textContent = 'SMA 20 · waiting for completed candles.'; return; }
  const availability = VIEWPORT.indicatorAvailability(timeline, Number(candle.time), 20);
  const value = indicatorByTime.get(Number(candle.time));
  note.textContent = value == null
    ? `SMA 20 unavailable at ${nyEpoch(candle.time)} · ${availability.reason}`
    : `SMA 20 available at ${nyEpoch(candle.time)} · ${dollars(value)} · resets each NYSE session and after missing candles.`;
}
function updateTimeline(candles) {
  const previousLatestLiveTime = latestLiveTime;
  const byTime = new Map(timeline.map(row => [Number(row.time), row]));
  for (const candle of candles) {
    const stamp = Number(candle.time);
    const existing = byTime.get(stamp);
    // A persisted live candle is the authoritative observation for an overlap.
    byTime.set(stamp, existing && !candle.live ? existing : { ...candle, time: stamp });
  }
  const updated = [...byTime.values()].sort((a,b) => a.time-b.time);
  const changed = updated.length !== timeline.length || updated.some((row, i) => Number(row.time) !== Number(timeline[i]?.time) || row.close !== timeline[i]?.close || row.volume !== timeline[i]?.volume);
  if (!changed) return;
  const scale = chart?.timeScale();
  const oldRange = viewportInitialized && scale ? scale.getVisibleLogicalRange() : null;
  const newRange = VIEWPORT?.appendedRange(oldRange, previousCount, updated.length);
  timeline = updated;
  candleByTime = new Map(timeline.map(row => [Number(row.time), row]));
  latestLiveTime = timeline.reduce((latest, row) => row.live ? Math.max(latest ?? -Infinity, Number(row.time)) : latest, null);
  if (candleSeries) {
    if (!viewportInitialized && timeline.length) scale.applyOptions({ barSpacing: VIEWPORT?.candleSpacing(root.clientWidth - 70, timeline.length) || 10, rightOffset: 2 });
    candleSeries.setData(timeline.map(({time,open,high,low,close}) => ({time,open,high,low,close})));
    // Compute against the complete timeline, independent of viewport or selection.
    indicatorByTime = new Map(VIEWPORT.simpleMovingAverage(timeline, 20).map(row => [row.time, row.value]));
    renderIndicator();
    previousCount = timeline.length;
    if (timeline.length && !viewportInitialized) {
      const initial = VIEWPORT.initialRange(timeline.length);
      if (initial) scale.setVisibleLogicalRange(initial);
      viewportInitialized = true;
    } else if (newRange) scale.setVisibleLogicalRange(newRange);
    candleSeries.setMarkers([...predictions.values()].map(row => ({
      time: row.candle_time_epoch,
      position: row.prediction === 'UP' ? 'belowBar' : row.prediction === 'DOWN' ? 'aboveBar' : 'inBar',
      color: row.prediction === 'UP' ? '#37d6a0' : row.prediction === 'DOWN' ? '#fb7185' : '#f5c76b',
      shape: row.prediction === 'UP' ? 'arrowUp' : row.prediction === 'DOWN' ? 'arrowDown' : 'circle', text: row.prediction
    })).filter(marker => candleByTime.has(marker.time)).sort((a,b) => a.time-b.time));
  }
  if (selectedTime != null && candleByTime.has(selectedTime)) renderSelectedCandle();
  if (selectedTime != null && selectedTime === previousLatestLiveTime && latestLiveTime !== previousLatestLiveTime) selectCandle(selectedTime);
}
function resetView() {
  if (!chart || !timeline.length) return;
  const scale = chart.timeScale();
  scale.applyOptions({ barSpacing: VIEWPORT.candleSpacing(root.clientWidth - 70, timeline.length), rightOffset: 2 });
  const range = VIEWPORT.initialRange(timeline.length);
  if (range) scale.setVisibleLogicalRange(range);
}
el('chart-reset').addEventListener('click', resetView);
el('return-latest').addEventListener('click', () => {
  resetView();
  if (latestLiveTime != null) selectCandle(latestLiveTime);
  else if (timeline.length) selectCandle(Number(timeline.at(-1).time));
});
el('indicator-toggle').addEventListener('change', renderIndicator);

function setProbability(name, value, prefix = '') {
  el(`${prefix}prob-${name}`).textContent = pct(value);
  el(`${prefix}bar-${name}`).style.width = `${Math.max(0, Math.min(100, Number(value || 0) * 100))}%`;
}
function clearLatestPrediction(message, chipText = 'NO RECORD', chipClass = 'no-prediction') {
  el('latest-prediction-content').classList.add('hidden');
  el('latest-prediction-empty').textContent = message;
  el('latest-prediction-empty').classList.remove('hidden');
  const chip = el('latest-settlement-chip'); chip.textContent = chipText; chip.className = `state-chip ${chipClass}`;
}
function renderPipelineDiagnostics(status) {
  const providerUp = Boolean(status.connected && status.authenticated && status.subscribed);
  const providerReason = status.provider_error || status.state || 'awaiting connection';
  el('latest-provider-status').textContent = providerUp
    ? `Provider: connected · authenticated · ECL subscribed${status.provider_error ? ` · ${status.provider_error}` : ''}`
    : `Provider: disconnected or not ready · ${providerReason}`;
  const market = status.market_state || 'Market status unavailable';
  el('latest-market-status').textContent = `Market: ${market}${market !== 'Market open' ? ' · no new market candle is expected until the next open session.' : ''}`;

  const aggregation = status.last_aggregation_status;
  let aggregationMessage = 'No aggregation attempt recorded yet.';
  if (aggregation === 'waiting_for_finalization_buffer') aggregationMessage = 'Latest candle is awaiting finalization.';
  else if (['known_stream_interruption', 'minute_processing_failure', 'empty_bucket', 'missing_expected_minute',
    'outside_session_or_unaligned', 'mixed_source_or_feed', 'finalized_candle_conflict'].includes(aggregation)) {
    aggregationMessage = `Latest aggregation rejected: ${aggregation.replaceAll('_', ' ')}.`;
  } else if (aggregation) aggregationMessage = `Latest aggregation status: ${aggregation.replaceAll('_', ' ')}.`;
  el('latest-aggregation-status').textContent = aggregationMessage;

  const attempt = status.latest_prediction_attempt;
  let attemptMessage = 'No prediction attempt recorded for a newly finalized candle.';
  if (attempt?.status === 'feature_warmup_incomplete') {
    const names = (attempt.missing_features || []).join(', ') || 'required rolling features';
    attemptMessage = `Feature warm-up incomplete${attempt.candle_time_utc ? ` at ${ny(attempt.candle_time_utc)}` : ''}: consecutive completed candles are insufficient. Missing: ${names}.`;
  } else if (attempt?.status === 'prediction_saved') attemptMessage = 'Prediction successfully generated and saved.';
  else if (attempt?.status === 'prediction_already_saved') attemptMessage = 'A prediction is already saved for this candle.';
  else if (attempt?.status === 'prediction_error') attemptMessage = `Prediction generation or persistence error: ${attempt.reason || attempt.error_type || 'check application diagnostics'}.`;
  else if (attempt?.status === 'candle_awaiting_finalization') attemptMessage = 'Candle is awaiting its finalization buffer; prediction has not run.';
  else if (attempt?.status === 'retroactive_prediction_skipped') attemptMessage = 'Historical recovery was skipped; no retrospective prediction was created.';
  else if (attempt?.status) attemptMessage = `Latest prediction attempt: ${attempt.status.replaceAll('_', ' ')}${attempt.reason ? ` · ${attempt.reason.replaceAll('_', ' ')}` : ''}.`;
  else if (!status.model_ready) attemptMessage = `Prediction attempt unavailable because the model is not ready: ${status.engine_error || 'integrity check failed'}.`;
  el('latest-attempt-status').textContent = attemptMessage;
}
function renderLatestPrediction(prediction, status) {
  renderPipelineDiagnostics(status || {});
  if (!prediction) {
    clearLatestPrediction('No model prediction recorded yet. The panel reads saved records only; it does not run inference.');
    el('latest-prediction-time').textContent = 'No saved prediction timestamp';
    latestPredictionLoaded = true;
    return;
  }
  el('latest-prediction-empty').classList.add('hidden');
  el('latest-prediction-content').classList.remove('hidden');
  const signal = el('latest-signal-pill'); signal.textContent = prediction.prediction;
  signal.className = `signal-pill ${String(prediction.prediction).toLowerCase()}`;
  el('latest-prediction-time').textContent = `Candle ${ny(prediction.candle_time_utc)} · ${prediction.horizon_minutes}-minute horizon`;
  el('latest-created-time').textContent = `Predicted ${ny(prediction.prediction_created_utc)}`;
  el('latest-confidence').textContent = `Confidence ${pct(prediction.confidence)}`;
  el('latest-reference-label').textContent = `Reference close · ${prediction.horizon_minutes}-minute horizon`;
  el('latest-reference-close').textContent = dollars(prediction.reference_close);
  el('latest-outcome-time').textContent = `Endpoint ${ny(prediction.outcome_candle_time_utc)}`;
  setProbability('down', prediction.probability_down, 'latest-');
  setProbability('neutral', prediction.probability_neutral, 'latest-');
  setProbability('up', prediction.probability_up, 'latest-');
  const chip = el('latest-settlement-chip'), outcome = el('latest-outcome-context');
  outcome.className = 'outcome-context hidden'; outcome.textContent = '';
  if (prediction.actual_target != null) {
    chip.textContent = 'SETTLED'; chip.className = 'state-chip settled';
    outcome.className = `outcome-context ${prediction.correct ? 'correct' : 'incorrect'}`;
    outcome.textContent = `${prediction.correct ? 'CORRECT' : 'INCORRECT'} · Actual ${prediction.actual_target} · return ${Number(prediction.actual_return_pct).toFixed(3)}%`;
  } else {
    chip.textContent = 'PENDING'; chip.className = 'state-chip pending';
    outcome.className = 'outcome-context pending';
    outcome.textContent = `Pending settlement · endpoint ${ny(prediction.outcome_candle_time_utc)}.`;
  }
  latestPredictionLoaded = true;
}
function clearPrediction(message, chipText = 'NO RECORD', chipClass = 'no-prediction') {
  el('prediction-content').classList.add('hidden');
  el('prediction-empty').textContent = message;
  el('prediction-empty').classList.remove('hidden');
  const chip = el('settlement-chip'); chip.textContent = chipText; chip.className = `state-chip ${chipClass}`;
}
function renderSelectedCandle() {
  const candle = candleByTime.get(Number(selectedTime));
  if (!candle) return;
  const isLatestLive = candle.live && Number(candle.time) === latestLiveTime && marketIsOpen;
  el('selected-time').textContent = `Selected ${nyEpoch(candle.time)}${isLatestLive ? ' · LATEST LIVE' : ''}`;
  el('detail-kind').textContent = isLatestLive ? 'LATEST LIVE CANDLE' : 'HISTORICAL CANDLE';
  el('detail-heading').textContent = isLatestLive ? 'Latest completed candle' : 'Selected candle';
  el('detail-time').textContent = nyEpoch(candle.time);
  el('detail-open').textContent = dollars(candle.open); el('detail-high').textContent = dollars(candle.high);
  el('detail-low').textContent = dollars(candle.low); el('detail-close').textContent = dollars(candle.close);
  el('detail-volume').textContent = quantity(candle.volume);
  const indicator = indicatorByTime.get(Number(candle.time)); el('detail-indicator').textContent = indicator == null ? 'Unavailable' : dollars(indicator);
  renderIndicatorExplanation();
}
async function selectCandle(time) {
  const stamp = Number(time);
  if (!candleByTime.has(stamp)) return;
  selectedTime = stamp;
  renderSelectedCandle();
  const candle = candleByTime.get(stamp);
  const request = ++detailRequest;
  const isLatestLive = candle.live && Number(candle.time) === latestLiveTime && marketIsOpen;
  clearPrediction(isLatestLive ? 'Waiting for a recorded prediction for this completed live candle.' : 'Checking the immutable prediction ledger…', 'CHECKING', 'pending');
  try {
    const response = await fetch(`/api/market-candle-details?candle_time_utc=${encodeURIComponent(candle.candle_time_utc || new Date(stamp * 1000).toISOString())}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not load recorded prediction details');
    if (request !== detailRequest || selectedTime !== stamp) return;
    if (!data.prediction) {
      clearPrediction(isLatestLive ? 'No prediction has been recorded for this completed live candle.' : 'No prediction recorded', 'NO PREDICTION', 'no-prediction');
      return;
    }
    const prediction = data.prediction;
    // Future-facing latest live signals never display an actual result, even if
    // the selected record changes while the chart is refreshing.
    const liveFuture = !VIEWPORT.shouldShowSettledOutcome(
      candle, latestLiveTime, marketIsOpen, prediction.settlement_status === 'SETTLED'
    );
    el('prediction-empty').classList.add('hidden'); el('prediction-content').classList.remove('hidden');
    const pill = el('signal-pill'); pill.textContent = prediction.prediction; pill.className = `signal-pill ${prediction.prediction.toLowerCase()}`;
    el('signal-time').textContent = `Created ${ny(prediction.prediction_created_utc)}`;
    el('reference-close').textContent = dollars(prediction.reference_close);
    el('outcome-time').textContent = `15-minute endpoint ${ny(prediction.outcome_candle_time_utc)}`;
    setProbability('down', prediction.probability_down); setProbability('neutral', prediction.probability_neutral); setProbability('up', prediction.probability_up);
    const chip = el('settlement-chip'); const outcome = el('outcome-context');
    outcome.className = 'outcome-context hidden'; outcome.textContent = '';
    if (liveFuture) {
      chip.textContent = 'FORWARD LOOKING'; chip.className = 'state-chip pending';
      outcome.classList.add('hidden');
    } else if (prediction.settlement_status === 'SETTLED') {
      chip.textContent = 'SETTLED'; chip.className = 'state-chip settled';
      outcome.classList.remove('hidden'); outcome.className = `outcome-context ${prediction.correct ? 'correct' : 'incorrect'}`;
      outcome.textContent = `${prediction.correct ? 'CORRECT' : 'INCORRECT'} · Actual ${prediction.actual_target} · endpoint close ${dollars(prediction.actual_close)} · return ${Number(prediction.actual_return_pct).toFixed(3)}%`;
    } else {
      chip.textContent = 'PENDING'; chip.className = 'state-chip pending';
      outcome.classList.remove('hidden'); outcome.className = 'outcome-context pending';
      outcome.textContent = `Pending settlement · waiting for a verified continuous path through ${ny(prediction.outcome_candle_time_utc)}.`;
    }
  } catch (error) {
    if (request === detailRequest) clearPrediction(`Prediction ledger unavailable: ${error.message}`, 'UNAVAILABLE', 'pending');
  }
}

function updateStatus(status, coverage) {
  marketIsOpen = status.market_state === 'Market open';
  el('stream-state').textContent = (status.state || 'unknown').replaceAll('_',' ');
  el('stream-message').textContent = status.message || status.last_safe_error || '';
  el('market-state').textContent = status.market_state || '—';
  el('last-event').textContent = status.last_event_time_utc ? ny(status.last_event_time_utc) : '—';
  el('event-delay').textContent = status.receive_delay_seconds == null ? 'Receive delay —' : `Receive delay ${Math.max(0,Number(status.receive_delay_seconds)).toFixed(1)} sec`;
  el('last-candle').textContent = `Last completed candle ${status.last_completed_candle_time_et ? ny(status.last_completed_candle_time_et) : '—'} · Reconnects: ${status.reconnect_count || 0}`;
  if (!status.display_authorized) { el('auth-notice').textContent = status.message || 'Alpaca display is disabled.'; el('auth-notice').classList.remove('hidden'); }
  else el('auth-notice').classList.add('hidden');
  const gaps = status.missing_data_warnings || [];
  if (gaps.length) { el('warnings').innerHTML = `<strong>Missing IEX activity / data warnings</strong><span>${gaps.length} bucket(s) withheld. No prices are filled in.</span>`; el('warnings').classList.remove('hidden'); }
  else el('warnings').classList.add('hidden');
  const first = coverage?.first_candle_utc, last = coverage?.last_candle_utc;
  el('history-range').textContent = first && last ? `${ny(first)} – ${ny(last)}` : 'History unavailable';
  el('chart-coverage').textContent = coverage ? `${Number(coverage.historical_candle_count || 0).toLocaleString()} permitted historical candles · ${Number(coverage.live_candle_count || 0)} persisted live · history ends ${last ? ny(last) : 'unavailable'}; not all-time history` : 'Historical coverage metadata unavailable';
}
async function loadHistoricalTimeline() {
  const response = await fetch('/api/market-timeline');
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Historical timeline could not be loaded');
  updateTimeline(data.candles || []);
  marketCoverage = data.coverage || null;
  return data.coverage;
}
async function refresh() {
  if (refreshBusy) return;
  refreshBusy = true;
  try {
    if (!historyLoaded) {
      await loadHistoricalTimeline();
      historyLoaded = true;
    }
    const urls = ['/api/status','/api/candles?limit=2000','/api/predictions?limit=100000'];
    const responses = await Promise.all(urls.map(url => fetch(url)));
    const data = await Promise.all(responses.map(async response => { const json = await response.json(); if (!response.ok) throw new Error(json.detail || 'Dashboard request failed'); return json; }));
    const [status, liveCandles, predictionRows] = data;
    predictions = new Map(predictionRows.map(row => [Number(row.candle_time_epoch), row]));
    renderLatestPrediction(predictionRows[0] || null, status);
    updateTimeline(liveCandles);
    // Keep the history extent explicit while status and new live candles update.
    updateStatus(status, marketCoverage);
    if (candleSeries) candleSeries.setMarkers([...predictions.values()].map(row => ({
      time: row.candle_time_epoch, position: row.prediction === 'UP' ? 'belowBar' : row.prediction === 'DOWN' ? 'aboveBar' : 'inBar',
      color: row.prediction === 'UP' ? '#37d6a0' : row.prediction === 'DOWN' ? '#fb7185' : '#f5c76b',
      shape: row.prediction === 'UP' ? 'arrowUp' : row.prediction === 'DOWN' ? 'arrowDown' : 'circle', text: row.prediction
    })).filter(marker => candleByTime.has(marker.time)).sort((a,b) => a.time-b.time));
    if (selectedTime != null && candleByTime.has(selectedTime)) renderSelectedCandle();
  } catch (error) {
    el('stream-state').textContent = 'Dashboard unavailable'; el('stream-message').textContent = String(error.message || error);
    if (!latestPredictionLoaded) clearLatestPrediction(`Latest saved prediction could not be loaded: ${error.message || error}`, 'UNAVAILABLE', 'pending');
    if (!timeline.length) { chartState.textContent = `Market timeline unavailable: ${error.message || error}`; chartState.classList.remove('hidden'); }
  } finally { refreshBusy = false; }
}
refresh(); setInterval(refresh, 10000);
