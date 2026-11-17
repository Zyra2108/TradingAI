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
let chart = null, candleSeries = null, indicatorSeries = null, volumeSeries = null, currentPriceLine = null;
let timeline = [], candleByTime = new Map(), indicatorByTime = new Map();
let predictions = new Map(), historicalSignals = new Map(), savedResearchPreviews = new Map(), selectedTime = null, viewportInitialized = false, previousCount = 0;
let historyLoaded = false, historicalSignalsLoaded = false, latestLiveTime = null, refreshBusy = false, marketCoverage = null, marketIsOpen = false;
let latestPredictionLoaded = false;
let selectedResearchModel = 'xgboost_multiclass', researchCatalog = null, researchCatalogPromise = null;
let previewHistoryRefreshAt = 0;

function visibleAutoscaleInfoProvider(original) {
  const range = chart?.timeScale().getVisibleLogicalRange();
  const priceRange = VIEWPORT?.visiblePriceRange(timeline, range, indicatorByTime, el('indicator-select')?.value === 'sma');
  if (!priceRange) return original();
  return { priceRange, margins: { above: 0, below: 0 } };
}

if (window.LightweightCharts?.createChart) {
  chart = LightweightCharts.createChart(root, {
    width: Math.max(1, root.clientWidth), height: Math.max(1, root.clientHeight),
    layout: { background: { color: '#ffffff' }, textColor: '#586571' },
    grid: { vertLines: { color: '#edf0f2' }, horzLines: { color: '#edf0f2' } },
    rightPriceScale: { borderColor: '#dce2e7', autoScale: true, scaleMargins: { top: 0.08, bottom: 0.22 } },
    timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 2,
      barSpacing: VIEWPORT?.candleSpacing(root.clientWidth - 70, 65) || 10,
      tickMarkFormatter: t => new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(Number(t) * 1000)) },
    localization: { timeFormatter: t => nyEpoch(t) }
  });
  candleSeries = chart.addCandlestickSeries({ upColor: '#13805d', downColor: '#c7434f', borderVisible: false, wickUpColor: '#13805d', wickDownColor: '#c7434f', autoscaleInfoProvider: visibleAutoscaleInfoProvider });
  indicatorSeries = chart.addLineSeries({ color: '#957018', lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: true });
  volumeSeries = chart.addHistogramSeries({ priceScaleId: 'volume', priceFormat: { type: 'volume' }, priceLineVisible: false, lastValueVisible: false, base: 0 });
  chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 }, visible: false, borderVisible: false });
  if (window.ResizeObserver) new ResizeObserver(() => {
    chart.applyOptions({ width: Math.max(1, root.clientWidth), height: Math.max(1, root.clientHeight) });
    renderSupportResistance(selectedTime == null ? Number(timeline.at(-1)?.time) : Number(selectedTime));
  }).observe(root);
  chart.subscribeCrosshairMove(param => {
    const t = param?.time;
    el('crosshair-time').textContent = typeof t === 'number' ? nyEpoch(t) : 'Move over chart';
  });
  chart.subscribeClick(param => {
    if (!param?.time || typeof param.time !== 'number') return;
    if (param.seriesData?.get(candleSeries)) selectCandle(param.time);
  });
  chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
    renderSupportResistance(selectedTime == null ? Number(timeline.at(-1)?.time) : Number(selectedTime));
  });
} else {
  chartState.textContent = 'Candlestick chart unavailable: the chart library could not be loaded.';
  chartState.classList.remove('hidden');
}

function renderIndicator() {
  const selected = el('indicator-select').value;
  const smaSelected = selected === 'sma';
  el('indicator-context-title').textContent = smaSelected ? 'SMA 20' : 'Model Indicator';
  el('indicator-state').textContent = smaSelected ? '20 consecutive closes' : 'Saved AI predictions';
  if (indicatorSeries) indicatorSeries.setData(VIEWPORT.indicatorData(timeline, 20, smaSelected));
  el('model-marker-key')?.classList.toggle('hidden', smaSelected);
  renderModelMarkers();
  const indicatorRow = el('detail-indicator-row');
  if (indicatorRow) indicatorRow.classList.toggle('hidden', !smaSelected);
  renderIndicatorExplanation();
}
function renderIndicatorExplanation() {
  const note = el('indicator-explanation');
  if (!note) return;
  if (el('indicator-select').value === 'model') {
    note.textContent = 'Displays saved AI-generated UP, DOWN and NEUTRAL prediction markers. HB means Historical Backtest; LIVE means genuine Live Prediction. Historical backtests are retrospective, not predictions originally issued live.';
    return;
  }
  note.textContent = 'Simple Moving Average calculated using 20 consecutive valid closing prices. The line appears after 20 valid candles and resets after genuine data gaps or session boundaries.';
}
function renderModelMarkers() {
  if (!candleSeries) return;
  if (el('indicator-select').value !== 'model') { candleSeries.setMarkers([]); return; }
  const markers = [];
  const appendMarker = (row, sourceType, time) => {
    const historical = sourceType === 'historical_backtest';
    const mainPreview = sourceType === 'research_preview';
    const modelLabel = row.model_id === 'xgboost_multiclass' ? 'MAIN' : row.model_id === 'ridge_return' ? 'RIDGE' : 'XGB-B';
    return {
      time: Number(time),
      position: row.prediction === 'UP' ? 'belowBar' : row.prediction === 'DOWN' ? 'aboveBar' : 'inBar',
      color: mainPreview ? (row.prediction === 'UP' ? '#13805d' : row.prediction === 'DOWN' ? '#c7434f' : '#b07b12')
        : row.prediction === 'UP' ? (historical ? '#83b7ff' : '#37d6a0')
        : row.prediction === 'DOWN' ? (historical ? '#c39aff' : '#fb7185') : '#f5c76b',
      shape: historical || mainPreview ? 'square' : row.prediction === 'UP' ? 'arrowUp' : row.prediction === 'DOWN' ? 'arrowDown' : 'circle',
      text: mainPreview ? `${modelLabel} ${row.prediction}` : `${historical ? 'HB' : 'LIVE'} ${row.prediction}`
    };
  };
  for (const row of historicalSignals.values()) {
    const time = Number(row.candle_time_epoch);
    if (candleByTime.has(time)) markers.push(appendMarker(row, 'historical_backtest', time));
  }
  for (const row of predictions.values()) {
    const time = Number(row.candle_time_epoch);
    if (candleByTime.has(time)) markers.push(appendMarker(row, 'live', time));
  }
  for (const row of savedResearchPreviews.values()) {
    if (row.model_id !== selectedResearchModel) continue;
    const time = Math.floor(new Date(row.candle_time_utc).getTime() / 1000);
    if (candleByTime.has(time)) markers.push(appendMarker(row, 'research_preview', time));
  }
  markers.sort((a,b) => a.time-b.time);
  candleSeries.setMarkers(markers);
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
    if (volumeSeries) volumeSeries.setData(timeline.filter(row => Number.isFinite(Number(row.volume)) && Number(row.volume) >= 0).map(row => ({
      time: Number(row.time), value: Number(row.volume),
      color: Number(row.close) >= Number(row.open) ? 'rgba(19,128,93,.28)' : 'rgba(199,67,79,.26)',
    })));
    if (currentPriceLine) candleSeries.removePriceLine(currentPriceLine);
    const latest = timeline.at(-1);
    if (latest) currentPriceLine = candleSeries.createPriceLine({ price: Number(latest.close), color: '#8d9aa6', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'Last' });
    // Compute against the complete timeline, independent of viewport or selection.
    indicatorByTime = new Map(VIEWPORT.simpleMovingAverage(timeline, 20).map(row => [row.time, row.value]));
    renderIndicator();
    previousCount = timeline.length;
    if (timeline.length && !viewportInitialized) {
      const initial = VIEWPORT.initialRange(timeline.length);
      if (initial) scale.setVisibleLogicalRange(initial);
      viewportInitialized = true;
    } else if (newRange) scale.setVisibleLogicalRange(newRange);
    renderMarketOverview();
    renderMarketContext(selectedTime == null ? Number(latest?.time) : Number(selectedTime));
    renderSupportResistance(selectedTime == null ? Number(latest?.time) : Number(selectedTime));
    renderModelMarkers();
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
el('indicator-select').addEventListener('change', renderIndicator);
el('chart-fullscreen').addEventListener('click', async () => {
  const panel = document.querySelector('.chart-panel');
  try {
    if (!document.fullscreenElement) await panel.requestFullscreen();
    else await document.exitFullscreen();
  } catch { /* Fullscreen can be unavailable in embedded or restricted browsers. */ }
});
document.addEventListener('fullscreenchange', () => {
  if (chart) chart.applyOptions({ width: Math.max(1, root.clientWidth), height: Math.max(1, root.clientHeight) });
  renderSupportResistance(selectedTime == null ? Number(timeline.at(-1)?.time) : Number(selectedTime));
});
el('model-select').addEventListener('change', () => {
  selectedResearchModel = el('model-select').value;
  updateResearchModelDescription();
  renderSelectedSavedResearchPreview();
  renderModelMarkers();
});
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
  const productionPanel = el('production-panel');
  if (productionPanel) productionPanel.classList.toggle('hidden', !prediction);
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
function nyDate(epoch) {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(Number(epoch) * 1000));
}
function percentNumber(value, digits = 2) {
  return value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(digits)}%`;
}
function renderMarketOverview() {
  if (!timeline.length) return;
  const last = timeline.at(-1);
  const session = timeline.filter(row => nyDate(row.time) === nyDate(last.time));
  const base = session[0]?.open;
  const delta = base == null ? null : Number(last.close) - Number(base);
  const change = delta == null ? null : delta / Number(base) * 100;
  for (const id of ['overview-last-price', 'chart-last-price']) el(id).textContent = dollars(last.close);
  const changeText = change == null ? 'Session change unavailable' : `${delta >= 0 ? '+' : ''}${dollars(delta)} (${change >= 0 ? '+' : ''}${change.toFixed(2)}%)`;
  for (const id of ['overview-change', 'chart-price-change']) {
    const node = el(id); node.textContent = changeText;
    node.className = `change-${change == null || delta === 0 ? 'flat' : delta > 0 ? 'up' : 'down'}`;
  }
}
function renderMarketContext(stamp) {
  const candle = candleByTime.get(Number(stamp));
  if (!candle) return;
  el('context-open').textContent = dollars(candle.open); el('context-high').textContent = dollars(candle.high);
  el('context-low').textContent = dollars(candle.low); el('context-close').textContent = dollars(candle.close);
  el('context-volume').textContent = quantity(candle.volume);
  const sessionRows = timeline.filter(row => nyDate(row.time) === nyDate(candle.time) && Number(row.time) <= Number(candle.time));
  const last20 = sessionRows.slice(-21);
  const returns = [];
  for (let i = 1; i < last20.length; i++) {
    const earlier = Number(last20[i - 1].close), later = Number(last20[i].close);
    if (earlier > 0) returns.push(later / earlier - 1);
  }
  const meanVolume = sessionRows.slice(-20).map(row => Number(row.volume)).filter(Number.isFinite);
  const averageVolume = meanVolume.length ? meanVolume.reduce((sum, value) => sum + value, 0) / meanVolume.length : null;
  const recentVolatility = returns.length > 1 ? Math.sqrt(returns.reduce((sum, value) => sum + value * value, 0) / returns.length) * 100 : null;
  el('activity-summary').textContent = `${recentVolatility == null ? 'Volatility: insufficient closes' : `20-bar realized volatility: ${recentVolatility.toFixed(3)}%`} · ${averageVolume == null ? 'Volume: unavailable' : `mean volume ${quantity(averageVolume)}`}`;
}
function renderSupportResistance(stamp) {
  const overlay = el('level-overlay');
  if (!overlay || !window.ECLMarketLevels || !timeline.length || !Number.isFinite(Number(stamp))) return;
  const result = window.ECLMarketLevels.calculateZones(timeline, Number(stamp));
  const support = result.support || [], resistance = result.resistance || [];
  const summaries = [
    ['support-summary', support, 'support'], ['resistance-summary', resistance, 'resistance'],
  ];
  for (const [id, zones, kind] of summaries) {
    const node = el(id); node.replaceChildren();
    if (!zones.length) {
      const empty = document.createElement('span'); empty.textContent = `No repeated ${kind} zone in the current continuous session window.`; node.append(empty);
    } else {
      for (const zone of zones) {
        const line = document.createElement('div'); line.className = kind === 'support' ? 'zone-price' : 'zone-price resistance-text';
        line.textContent = `${dollars(zone.low)}–${dollars(zone.high)} · ${zone.touches} touches`; node.append(line);
      }
    }
  }
  el('context-method').textContent = result.method + ' · levels use candles no later than the selected timestamp';
  overlay.replaceChildren();
  if (!chart || !candleSeries) return;
  const leftCoordinate = chart.timeScale().timeToCoordinate(Number(stamp));
  const plotWidth = Math.max(0, root.clientWidth - 62);
  if (leftCoordinate == null) return;
  const left = leftCoordinate == null ? 0 : Math.max(0, Math.min(plotWidth, leftCoordinate));
  for (const [kind, zones] of [['support', support], ['resistance', resistance]]) {
    for (const zone of zones) {
      const yHigh = candleSeries.priceToCoordinate(Number(zone.high));
      const yLow = candleSeries.priceToCoordinate(Number(zone.low));
      if (yHigh == null || yLow == null || plotWidth <= left) continue;
      const band = document.createElement('div'); band.className = `level-zone ${kind}`;
      band.style.top = `${Math.max(0, Math.min(yHigh, yLow))}px`;
      band.style.height = `${Math.max(2, Math.abs(yLow - yHigh))}px`;
      band.style.left = `${left}px`; band.style.right = `${Math.max(0, root.clientWidth - plotWidth)}px`;
      const label = document.createElement('span'); label.textContent = `${kind === 'support' ? 'SUPPORT' : 'RESISTANCE'} ${dollars(zone.low)}–${dollars(zone.high)}`;
      band.append(label); overlay.append(band);
    }
  }
}
function renderResearchCatalog(data) {
  researchCatalog = data;
  const models = new Map((data.models || []).map(model => [model.id, model]));
  updateResearchModelDescription(models);
  const grid = el('model-comparison-grid'); grid.replaceChildren();
  if (data.baselines) {
    const baseline = document.createElement('p'); baseline.className = 'comparison-disclosure';
    baseline.textContent = `Validation references (same period): always-NEUTRAL all-row accuracy ${percentNumber(data.baselines.always_neutral_accuracy_all_rows)}; majority directional class ${percentNumber(data.baselines.directional_majority_accuracy_eligible_rows)} on ${data.baselines.directional_rows} eligible UP/DOWN rows.`;
    grid.append(baseline);
  }
  for (const model of (data.models || []).filter(item => item.id === 'xgboost_multiclass')) {
    const validation = model.evaluation.validation || {}, wf = model.evaluation.walk_forward || {};
    const multiclass = model.kind === 'multiclass';
    const metrics = [
      [multiclass ? 'ALL-ROW ACC.' : 'DIRECTIONAL ACC.', validation.accuracy],
      ['BALANCED ACC.', validation.balanced_accuracy], ['MACRO F1', validation.macro_f1],
    ];
    const other = multiclass ? [
      ['DOWN P / R', validation.per_class?.DOWN ? `${percentNumber(validation.per_class.DOWN.precision)} / ${percentNumber(validation.per_class.DOWN.recall)}` : null],
      ['NEUTRAL P / R', validation.per_class?.NEUTRAL ? `${percentNumber(validation.per_class.NEUTRAL.precision)} / ${percentNumber(validation.per_class.NEUTRAL.recall)}` : null],
      ['UP P / R', validation.per_class?.UP ? `${percentNumber(validation.per_class.UP.precision)} / ${percentNumber(validation.per_class.UP.recall)}` : null],
      ['DIR. COVERAGE', validation.coverage], ['OOF MACRO F1', wf.mean_fold_macro_f1],
      ['OOF ACC. ± SD', wf.mean_fold_accuracy == null ? null : `${percentNumber(wf.mean_fold_accuracy)} ± ${percentNumber(wf.std_fold_accuracy)}`],
    ] : [
      ['UP P / R', validation.up_precision == null ? null : `${percentNumber(validation.up_precision)} / ${percentNumber(validation.up_recall)}`],
      ['DOWN P / R', validation.down_precision == null ? null : `${percentNumber(validation.down_precision)} / ${percentNumber(validation.down_recall)}`],
      ['ELIGIBLE COVERAGE', validation.coverage], ['ALL-CANDLE ACC.', validation.all_candle_accuracy],
      ['OOF ACC. ± SD', wf.mean_fold_accuracy == null ? null : `${percentNumber(wf.mean_fold_accuracy)} ± ${percentNumber(wf.std_fold_accuracy)}`],
      ['OOF MACRO F1', wf.mean_fold_macro_f1],
    ];
    const card = document.createElement('article'); card.className = 'model-card';
    const title = document.createElement('div'); title.className = 'model-card-head';
    const h = document.createElement('h3'); h.textContent = model.name;
    title.append(h); card.append(title);
    const note = document.createElement('p'); note.textContent = `${multiclass ? 'All-row three-class validation' : 'Eligible UP/DOWN validation; neutral errors shown separately'} · ${validation.date_start || 'Aug 6, 2026'}–${validation.date_end || 'Sep 24, 2026'} · N=${validation.n ?? '—'}; validation reused in previous research.`; card.append(note);
    const training = model.evaluation.training || {};
    const trainNote = document.createElement('p'); trainNote.className = 'model-card-training';
    trainNote.textContent = training.mean_fold_accuracy == null
      ? (training.note || 'In-sample training metrics are not available in the saved report; OOF metrics below are used for comparison.')
      : `In-sample fold mean: ${percentNumber(training.mean_fold_accuracy)} accuracy · ${percentNumber(training.mean_fold_macro_f1)} macro F1 (not out-of-fold).`;
    card.append(trainNote);
    const metricGrid = document.createElement('div'); metricGrid.className = 'metric-grid';
    for (const [label, value] of [...metrics, ...other]) {
      const cell = document.createElement('div'), name = document.createElement('span'), val = document.createElement('b');
      name.textContent = label; val.textContent = typeof value === 'string' ? value : percentNumber(value);
      cell.append(name, val); metricGrid.append(cell);
    }
    card.append(metricGrid);
    const folds = document.createElement('div'); folds.className = 'fold-row';
    for (const fold of wf.folds || []) {
      const pill = document.createElement('span'); pill.className = 'fold-pill'; pill.textContent = `F${fold.fold} ${percentNumber(fold.accuracy)} acc · ${percentNumber(fold.macro_f1)} F1`; folds.append(pill);
    }
    card.append(folds);
    if (validation.confusion_matrix) {
      const details = document.createElement('details'), summary = document.createElement('summary'), table = document.createElement('table');
      details.className = 'confusion-details'; summary.textContent = `Confusion matrix · actual rows / predicted columns · ${multiclass ? 'DOWN, NEUTRAL, UP' : 'DOWN, UP'}`;
      table.className = 'cm-table';
      const labels = multiclass ? ['DOWN', 'NEUTRAL', 'UP'] : ['DOWN', 'UP'];
      table.innerHTML = '<thead><tr><th>ACTUAL ↓ / PREDICTED →</th>' + labels.map(label => `<th>${escapeHtml(label)}</th>`).join('') + '</tr></thead>' + '<tbody>' + validation.confusion_matrix.map((row, i) => `<tr><th>${escapeHtml(labels[i])}</th>${row.map(value => `<td>${Number(value).toLocaleString()}</td>`).join('')}</tr>`).join('') + '</tbody>';
      details.append(summary, table); card.append(details);
    }
    grid.append(card);
  }
}
function updateResearchModelDescription(models = null) {
  if (!researchCatalog && !models) return;
  const byId = models || new Map((researchCatalog.models || []).map(model => [model.id, model]));
  const selected = byId.get(selectedResearchModel);
  if (selected) el('research-model-description').textContent = `${selected.name} · ${selected.summary} 15-minute horizon · ${selected.features} ordered causal features.`;
}
async function loadResearchCatalog() {
  if (researchCatalog) return;
  if (researchCatalogPromise) return researchCatalogPromise;
  researchCatalogPromise = (async () => {
    const response = await fetch('/api/research-models'); const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Research model summaries unavailable');
    renderResearchCatalog(data);
    return data;
  })();
  try { return await researchCatalogPromise; }
  catch (error) { researchCatalogPromise = null; throw error; }
}
function showResearchUnavailable(message, badge = 'UNAVAILABLE') {
  el('research-preview-content').classList.add('hidden');
  el('research-preview-empty').textContent = message; el('research-preview-empty').classList.remove('hidden');
  const status = el('research-model-status'); status.textContent = badge; status.className = 'badge badge-muted';
}
function renderResearchPreview(data) {
  const status = el('research-model-status');
  if (data.status !== 'preview') {
    const missing = (data.missing_features || []).length ? ` Missing features: ${data.missing_features.join(', ')}.` : '';
    showResearchUnavailable(`${data.message || 'Model preview unavailable.'}${missing}`, data.status === 'insufficient_features' ? 'WARMUP' : 'NO DATA');
    return;
  }
  el('research-preview-empty').classList.add('hidden'); el('research-preview-content').classList.remove('hidden');
  const pill = el('research-signal-pill'); pill.textContent = data.signal; pill.className = `signal-pill ${String(data.prediction).toLowerCase()}`;
  el('research-original-output').textContent = `${data.prediction} · saved model output`;
  el('research-preview-time').textContent = `${ny(data.candle_time_utc)} · computed ${ny(data.generated_at_utc)}`;
  el('research-reference-price').textContent = dollars(data.reference_price); el('research-horizon').textContent = `${data.horizon_minutes} min`;
  el('research-persisted').textContent = data.persisted ? 'Saved to research log' : 'Unrecorded preview';
  const isRegression = data.kind === 'regression';
  el('research-return-row').classList.toggle('hidden', !isRegression);
  if (isRegression) el('research-return').textContent = `${Number(data.predicted_return_percentage_points) > 0 ? '+' : ''}${Number(data.predicted_return_percentage_points).toFixed(4)}%`;
  el('research-probability-block').classList.toggle('hidden', isRegression);
  if (!isRegression) {
    const probabilities = data.probabilities || {};
    for (const name of ['down', 'neutral', 'up']) {
      const key = name.toUpperCase(), available = probabilities[key] != null;
      const row = el(`research-prob-${name}`).closest('.prob-row'); row.classList.toggle('hidden', !available);
      if (available) setProbability(name, probabilities[key], 'research-');
    }
  }
  status.textContent = 'SAVED PREVIEW'; status.className = 'badge badge-warning';
}
function renderSelectedSavedResearchPreview() {
  const rows = [...savedResearchPreviews.values()].filter(row => row.model_id === selectedResearchModel)
    .sort((a, b) => new Date(b.candle_time_utc) - new Date(a.candle_time_utc));
  const row = rows[0];
  if (!row) {
    showResearchUnavailable('No saved prediction for this model yet. New completed ECL IEX candles are recorded after inference succeeds.', 'NO RECORD');
    return;
  }
  renderResearchPreview({ ...row, status: 'preview', persisted: true, experimental: true,
    kind: row.model_id === 'ridge_return' ? 'regression' : 'classifier' });
}
function renderPredictionHistory() {
  const body = el('prediction-history-body'); if (!body) return;
  const rows = [];
  for (const row of predictions.values()) rows.push({ ...row, source_type: 'live' });
  for (const row of historicalSignals.values()) rows.push({ ...row, source_type: 'historical_backtest' });
  const filtered = rows.sort((a, b) => Number(b.candle_time_epoch || 0) - Number(a.candle_time_epoch || 0)).slice(0, 40);
  body.replaceChildren();
  if (!filtered.length) { const tr = document.createElement('tr'), td = document.createElement('td'); td.colSpan = 5; td.className = 'table-empty'; td.textContent = 'No saved prediction records in this ledger.'; tr.append(td); body.append(tr); return; }
  for (const row of filtered) {
    const tr = document.createElement('tr');
    const settled = row.settlement_status === 'SETTLED' || row.actual_target != null;
    const outcome = settled ? String(row.actual_target || '—') : row.settlement_status === 'UNAVAILABLE' ? 'Unavailable' : 'Pending';
    const cells = [
      row.prediction_created_et || row.prediction_created_utc || row.candle_time_et || row.candle_time_utc || '—',
      row.prediction || '—', dollars(row.reference_close), `${row.horizon_minutes || 15} min`, outcome,
    ];
    for (let i = 0; i < cells.length; i++) {
      const td = document.createElement('td'); td.textContent = cells[i];
      if (i === 4) td.className = outcome === 'UP' ? 'outcome-up' : outcome === 'DOWN' ? 'outcome-down' : outcome === 'NEUTRAL' ? 'outcome-neutral' : '';
      tr.append(td);
    }
    body.append(tr);
  }
}
function renderResearchPreviewHistory(records) {
  const body = el('research-preview-history-body'); if (!body) return;
  body.replaceChildren();
  if (!records?.length) {
    const tr = document.createElement('tr'), td = document.createElement('td');
    td.colSpan = 7; td.className = 'table-empty'; td.textContent = 'No experimental model outputs have been logged yet.';
    tr.append(td); body.append(tr); return;
  }
  for (const row of records) {
    const tr = document.createElement('tr'); tr.className = 'research-preview-history-row';
    let estimate = '—';
    if (row.probabilities) estimate = ['DOWN','NEUTRAL','UP'].filter(k => row.probabilities[k] != null)
      .map(k => `${k} ${(Number(row.probabilities[k]) * 100).toFixed(2)}%`).join(' · ');
    else if (row.predicted_return_percentage_points != null) estimate = `${Number(row.predicted_return_percentage_points).toFixed(5)} pp`;
    const outcome = row.outcome_status === 'SETTLED'
      ? `${row.actual_target} · ${row.correct ? 'CORRECT' : 'INCORRECT'} · ${Number(row.actual_return_pct).toFixed(3)}%`
      : row.outcome_status === 'UNAVAILABLE' ? 'Unavailable' : 'Pending';
    const values = [ny(row.candle_time_utc), row.model_name, `${row.prediction} · ${row.signal}`, estimate,
      dollars(row.reference_price), `${row.horizon_minutes || 15} min`, outcome];
    for (const value of values) { const td = document.createElement('td'); td.textContent = String(value ?? '—'); tr.append(td); }
    body.append(tr);
  }
}
async function refreshResearchPreviewHistory() {
  if (Date.now() - previewHistoryRefreshAt < 15000) return;
  previewHistoryRefreshAt = Date.now();
  try {
    const response = await fetch('/api/research-model-preview-history?limit=300');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Research preview log unavailable');
    const records = data.records || [];
    savedResearchPreviews = new Map(records.filter(row => row.status === 'preview')
      .map(row => [`${row.candle_time_utc}:${row.model_id}`, row]));
    renderResearchPreviewHistory(records);
    renderSelectedSavedResearchPreview();
    renderModelMarkers();
  } catch (error) {
    const body = el('research-preview-history-body');
    if (body) { body.replaceChildren(); const tr = document.createElement('tr'), td = document.createElement('td');
      td.colSpan = 7; td.className = 'table-empty'; td.textContent = `Research preview log unavailable: ${error.message}`; tr.append(td); body.append(tr); }
  }
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
  el('detail-indicator-label').textContent = 'SMA 20';
  renderMarketContext(Number(candle.time));
  renderSupportResistance(Number(candle.time));
  renderIndicatorExplanation();
}
function selectCandle(time) {
  const stamp = Number(time);
  if (!candleByTime.has(stamp)) return;
  selectedTime = stamp;
  renderSelectedCandle();
}

function updateStatus(status, coverage) {
  const recorderChip = el('research-recording-chip');
  if (recorderChip) {
    const recording = Boolean(status.research_preview_recording_enabled);
    recorderChip.textContent = recording ? 'AUTO LOG READY' : 'AUTO LOG OFF';
    recorderChip.className = `state-chip ${recording ? 'settled' : 'no-prediction'}`;
  }
  marketIsOpen = status.market_state === 'Market open';
  el('stream-state').textContent = (status.state || 'unknown').replaceAll('_',' ');
  el('stream-message').textContent = status.message || status.last_safe_error || '';
  const connected = Boolean(status.connected && status.authenticated && status.subscribed);
  el('connection-label').textContent = connected ? 'CONNECTED · IEX' : (status.state || 'WAITING').replaceAll('_', ' ').toUpperCase();
  el('connection-dot').className = `state-dot ${connected ? 'connected' : (status.provider_error || status.last_safe_error ? 'error' : '')}`;
  el('reconnect-label').textContent = `Reconnects: ${status.reconnect_count || 0}`;
  el('market-state').textContent = status.market_state || '—';
  const restRefresh = status.current_market_refresh || {};
  if (status.last_event_time_utc) {
    el('last-event').textContent = ny(status.last_event_time_utc);
    el('event-delay').textContent = status.receive_delay_seconds == null ? 'WebSocket event' : `WebSocket event delay · ${Math.max(0,Number(status.receive_delay_seconds)).toFixed(1)} sec`;
  } else if (restRefresh.latest_bar_utc) {
    el('last-event').textContent = ny(restRefresh.latest_bar_utc);
    el('event-delay').textContent = 'REST catch-up · completed IEX candle';
  } else {
    el('last-event').textContent = '—';
    el('event-delay').textContent = 'No recent IEX event';
  }
  const refreshError = restRefresh.last_safe_error ? ` · ${String(restRefresh.last_safe_error).replaceAll('_', ' ')}` : '';
  el('rest-refresh-status').textContent = `REST catch-up ${restRefresh.state || 'not configured'}${refreshError}`;
  el('last-candle').textContent = `Last completed candle ${status.last_completed_candle_time_et ? ny(status.last_completed_candle_time_et) : '—'} · Reconnects: ${status.reconnect_count || 0}`;
  if (!status.display_authorized) { el('auth-notice').textContent = status.message || 'Alpaca display is disabled.'; el('auth-notice').classList.remove('hidden'); }
  else el('auth-notice').classList.add('hidden');
  const first = coverage?.first_candle_utc, last = coverage?.last_candle_utc;
  el('history-range').textContent = first && last ? `${ny(first)} – ${ny(last)}` : 'History unavailable';
  const coverageMessage = coverage ? `${Number(coverage.historical_candle_count || 0).toLocaleString()} historical · ${Number(coverage.live_candle_count || 0)} persisted live · range ${first ? ny(first) : '—'} to ${last ? ny(last) : '—'}; available cache only` : 'Historical coverage metadata unavailable';
  el('chart-coverage').textContent = coverageMessage;
  el('timeline-coverage-note').textContent = coverageMessage;
}
async function loadHistoricalTimeline() {
  const response = await fetch('/api/market-timeline');
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Historical timeline could not be loaded');
  updateTimeline(data.candles || []);
  marketCoverage = data.coverage || null;
  return data.coverage;
}
async function refreshTimelineTail() {
  const response = await fetch('/api/market-timeline?tail=200');
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Latest Alpaca candle cache could not be loaded');
  marketCoverage = data.coverage || marketCoverage;
  updateTimeline(data.candles || []);
}
async function refresh() {
  if (refreshBusy) return;
  refreshBusy = true;
  try {
    if (!historyLoaded) {
      await loadHistoricalTimeline();
      historyLoaded = true;
    } else {
      // REST catch-up writes to the separate verified current-candle cache,
      // not to the production prediction ledger. Poll its tail so real new
      // bars appear in the already-loaded chart while preserving the viewport.
      try { await refreshTimelineTail(); }
      catch (error) { console.warn('Latest Alpaca candle cache refresh unavailable', error); }
    }
    const urls = ['/api/status','/api/candles?limit=2000','/api/predictions?limit=100000'];
    const responses = await Promise.all(urls.map(url => fetch(url)));
    const data = await Promise.all(responses.map(async response => { const json = await response.json(); if (!response.ok) throw new Error(json.detail || 'Dashboard request failed'); return json; }));
    const [status, liveCandles, predictionRows] = data;
    predictions = new Map(predictionRows.map(row => [Number(row.candle_time_epoch), row]));
    if (!historicalSignalsLoaded) {
      const response = await fetch('/api/historical-model-signals?limit=100000');
      const saved = await response.json();
      if (!response.ok) throw new Error(saved.detail || 'Historical model signals could not be loaded');
      historicalSignals = new Map((saved.signals || []).map(row => [Number(row.candle_time_epoch), row]));
      historicalSignalsLoaded = true;
    }
    renderLatestPrediction(predictionRows[0] || null, status);
    updateTimeline(liveCandles);
    // Keep the history extent explicit while status and new live candles update.
    updateStatus(status, marketCoverage);
    renderModelMarkers();
    renderPredictionHistory();
    if (!researchCatalog) {
      try { await loadResearchCatalog(); }
      catch { /* Evaluation cards are optional; candle, status and ledger data remain usable. */ }
    }
    if (selectedTime != null && candleByTime.has(selectedTime)) {
      renderSelectedCandle();
    }
    refreshResearchPreviewHistory();
  } catch (error) {
    el('stream-state').textContent = 'Dashboard unavailable'; el('stream-message').textContent = String(error.message || error);
    if (!latestPredictionLoaded) clearLatestPrediction(`Latest saved prediction could not be loaded: ${error.message || error}`, 'UNAVAILABLE', 'pending');
    if (!timeline.length) { chartState.textContent = `Market timeline unavailable: ${error.message || error}`; chartState.classList.remove('hidden'); }
  } finally { refreshBusy = false; }
}
refresh(); setInterval(refresh, 10000);
