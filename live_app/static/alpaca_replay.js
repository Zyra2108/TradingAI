// The replay chart is drawn locally so a blocked third-party chart CDN cannot
// prevent the date/candle API requests from running.
const replayRoot = document.getElementById('replay-chart');
const canvas = document.getElementById('replay-canvas');
const context = canvas?.getContext('2d');
const byId = id => document.getElementById(id);
const replayState = byId('replay-state');
const VIEWPORT = window.ECLChartViewport;
let replayCandles = [];
let selectedCandle = null;
let chartGeometry = null;
let visibleStart = 0;
let visibleCount = 65;
let dragState = null;
let draggedSinceDown = false;
const formatNy = value => value ? new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', dateStyle: 'medium', timeStyle: 'medium', hour12: false }).format(new Date(value)) : '—';
const percent = value => value == null ? '—' : `${(Number(value) * 100).toFixed(2)}%`;
const money = value => value == null ? '—' : `$${Number(value).toFixed(2)}`;
const setProbability = (label, value) => { byId(`replay-prob-${label}`).textContent = percent(value); byId(`replay-bar-${label}`).style.width = `${Math.max(0, Math.min(100, Number(value || 0) * 100))}%`; };

function clearResult(message) {
  byId('replay-signal').textContent = '—'; byId('replay-signal').className = 'signal-pill neutral';
  byId('replay-candle-time').textContent = message || 'Select a completed candle to begin';
  byId('replay-reference').textContent = '—'; byId('replay-confidence').textContent = '—';
  for (const label of ['down','neutral','up']) setProbability(label, null);
  byId('replay-target-close').textContent = '—'; byId('replay-target-time').textContent = 'Target time —';
  byId('replay-actual').textContent = 'Not evaluated'; byId('replay-actual').className = 'state-chip pending';
  byId('replay-correct').textContent = '—';
}

function drawCandles() {
  if (!context || !canvas || !replayRoot) {
    replayState.textContent = 'This browser cannot render the historical chart canvas.';
    return;
  }
  const width = Math.max(1, replayRoot.clientWidth), height = Math.max(1, replayRoot.clientHeight);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#0e1723'; context.fillRect(0, 0, width, height);
  if (!replayCandles.length) {
    context.fillStyle = '#91a4ba'; context.font = '13px Segoe UI, sans-serif';
    context.fillText('Select a trading date to load actual Alpaca IEX candles.', 22, 34);
    chartGeometry = null; return;
  }
  visibleCount = Math.max(1, Math.min(replayCandles.length, visibleCount));
  visibleStart = VIEWPORT
    ? VIEWPORT.panWindow(replayCandles.length, visibleStart, visibleCount, 0).start
    : Math.max(0, Math.min(visibleStart, replayCandles.length - visibleCount));
  const visibleCandles = replayCandles.slice(visibleStart, visibleStart + visibleCount);
  const left = 70, right = width - 16, top = 18, bottom = height - 32;
  const lows = visibleCandles.map(c => Number(c.low)), highs = visibleCandles.map(c => Number(c.high));
  let min = Math.min(...lows), max = Math.max(...highs);
  if (max === min) { min -= 0.01; max += 0.01; }
  const pad = (max - min) * 0.06; min -= pad; max += pad;
  const y = price => bottom - (price - min) / (max - min) * (bottom - top);
  const step = (right - left) / visibleCandles.length;
  chartGeometry = { left, right, top, bottom, step, visibleStart, visibleCount };
  context.font = '10px Segoe UI, sans-serif'; context.textAlign = 'right'; context.textBaseline = 'middle';
  for (let tick = 0; tick <= 4; tick++) {
    const py = top + tick * (bottom - top) / 4;
    const value = max - tick * (max - min) / 4;
    context.strokeStyle = '#1a2838'; context.lineWidth = 1; context.beginPath(); context.moveTo(left, py); context.lineTo(right, py); context.stroke();
    context.fillStyle = '#91a4ba'; context.fillText(`$${value.toFixed(2)}`, left - 8, py);
  }
  const bodyWidth = Math.max(3, Math.min(16, step * 0.68));
  visibleCandles.forEach((candle, visibleIndex) => {
    const index = visibleStart + visibleIndex;
    const x = left + step * (visibleIndex + 0.5);
    const open = Number(candle.open), close = Number(candle.close);
    const up = close >= open;
    const isSelected = selectedCandle?.candle_time_utc === candle.candle_time_utc;
    const color = isSelected ? '#73b5ff' : up ? '#37d6a0' : '#fb7185';
    context.strokeStyle = color; context.fillStyle = color; context.lineWidth = isSelected ? 2 : 1;
    context.beginPath(); context.moveTo(x, y(Number(candle.high))); context.lineTo(x, y(Number(candle.low))); context.stroke();
    const yo = y(open), yc = y(close), bodyTop = Math.min(yo, yc), bodyHeight = Math.max(1, Math.abs(yc - yo));
    context.fillRect(x - bodyWidth / 2, bodyTop, bodyWidth, bodyHeight);
    if (isSelected) {
      context.strokeStyle = '#73b5ff'; context.lineWidth = 1; context.setLineDash([3, 3]);
      context.beginPath(); context.moveTo(x, top); context.lineTo(x, bottom); context.stroke(); context.setLineDash([]);
      context.fillStyle = '#73b5ff'; context.beginPath(); context.arc(x, Math.max(top + 4, y(Number(candle.high)) - 8), 3, 0, Math.PI * 2); context.fill();
    }
  });
  context.textAlign = 'center'; context.textBaseline = 'top'; context.fillStyle = '#91a4ba';
  const labelCount = Math.min(6, replayCandles.length);
  for (let i = 0; i < labelCount; i++) {
    const visibleIndex = Math.round(i * (visibleCandles.length - 1) / Math.max(1, labelCount - 1));
    const index = visibleStart + visibleIndex;
    const candle = visibleCandles[visibleIndex];
    const time = new Date(candle.time * 1000).toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false });
    context.fillText(time, left + step * (visibleIndex + 0.5), bottom + 8);
  }
}

function selectCandle(candle) {
  selectedCandle = candle;
  const index = replayCandles.findIndex(row => row.candle_time_utc === candle.candle_time_utc);
  if (index >= 0 && (index < visibleStart || index >= visibleStart + visibleCount)) {
    visibleStart = Math.max(0, Math.min(index - Math.floor(visibleCount / 2), replayCandles.length - visibleCount));
  }
  byId('replay-candle').value = candle.candle_time_utc;
  drawCandles();
  clearResult(`Selected ${formatNy(candle.candle_time_utc)} · click Replay to calculate`);
  byId('run-replay').disabled = false;
}

async function getJson(url) {
  const response = await fetch(url);
  let body;
  try { body = await response.json(); } catch { throw new Error(`Request failed with HTTP ${response.status} (invalid JSON response)`); }
  if (!response.ok) throw new Error(body.detail || `Request failed with HTTP ${response.status}`);
  return body;
}

async function loadDate(day) {
  replayState.textContent = 'Loading completed five-minute candles…';
  byId('replay-candle').disabled = true; byId('run-replay').disabled = true;
  selectedCandle = null; replayCandles = []; drawCandles(); clearResult('Choose a completed candle.');
  replayCandles = await getJson(`/api/replay/candles?day=${encodeURIComponent(day)}`);
  replayCandles.sort((a, b) => a.time - b.time);
  const windowRange = VIEWPORT?.visibleWindow(replayCandles.length, 65)
    || { start: Math.max(0, replayCandles.length - 65), count: Math.min(65, replayCandles.length || 1) };
  visibleCount = Math.max(1, windowRange.count);
  visibleStart = windowRange.start;
  drawCandles();
  const select = byId('replay-candle'); select.innerHTML = '';
  for (const candle of replayCandles) {
    const option = document.createElement('option'); option.value = candle.candle_time_utc;
    option.textContent = `${formatNy(candle.candle_time_utc)} ET · close ${money(candle.close)}`; select.append(option);
  }
  select.disabled = replayCandles.length === 0;
  replayState.textContent = replayCandles.length ? `${replayCandles.length} actual Alpaca IEX candles · date is in New York time` : 'No candles are available for this trading date.';
  if (replayCandles.length) selectCandle(replayCandles[replayCandles.length - 1]);
}

async function runReplay() {
  if (!selectedCandle) return;
  byId('run-replay').disabled = true; replayState.textContent = 'Calculating frozen-model replay…';
  clearResult('Running the frozen model on information through the selected candle…');
  try {
    const day = byId('replay-date').value;
    const url = `/api/replay/predict?day=${encodeURIComponent(day)}&candle_time=${encodeURIComponent(selectedCandle.candle_time_utc)}`;
    const result = await getJson(url);
    if (result.status === 'unavailable') {
      const missing = (result.missing_features || []).join(', ');
      const context = result.feature_context;
      clearResult(`Prediction unavailable: ${result.reason.replaceAll('_',' ')}. Missing features: ${missing || 'not reported'}.`);
      replayState.textContent = context
        ? `Only ${context.continuous_candles} continuous candle(s) since ${context.reset_reason.replaceAll('_',' ')} at ${formatNy(context.segment_start_et)} ET. ${context.warmup_rule}`
        : 'Required historical features are not ready.';
      return;
    }
    const pill = byId('replay-signal'); pill.textContent = result.prediction; pill.className = `signal-pill ${result.prediction.toLowerCase()}`;
    byId('replay-candle-time').textContent = `${formatNy(result.candle_time_et)} ET · retrospective prediction`;
    byId('replay-reference').textContent = money(result.reference_close);
    for (const label of ['down','neutral','up']) setProbability(label, result.probabilities[label.toUpperCase()]);
    byId('replay-confidence').textContent = percent(result.confidence);
    const outcome = result.outcome;
    if (outcome.status === 'available') {
      byId('replay-target-close').textContent = money(outcome.target_close);
      byId('replay-target-time').textContent = `${formatNy(outcome.target_time_et)} ET · exactly +15 minutes`;
      byId('replay-actual').textContent = outcome.actual;
      byId('replay-actual').className = `state-chip ${outcome.correct ? 'settled' : 'pending'}`;
      byId('replay-correct').textContent = outcome.correct ? 'Yes' : 'No';
      byId('replay-outcome-note').textContent = `Observed return ${(outcome.return * 100).toFixed(3)}%; actual label uses the frozen ±0.10% target definition.`;
    } else {
      byId('replay-target-time').textContent = `Target time ${formatNy(outcome.target_time_utc)} ET`;
      byId('replay-actual').textContent = 'Unavailable'; byId('replay-actual').className = 'state-chip pending';
      byId('replay-correct').textContent = 'Not evaluated';
      byId('replay-outcome-note').textContent = outcome.reason === 'session_boundary' ? 'The +15-minute endpoint would cross a valid NYSE session boundary.' : 'The continuous same-session candle path is incomplete; no prices were filled or interpolated.';
    }
    replayState.textContent = 'Replay complete · no live prediction record was created.';
  } catch (error) {
    clearResult(`Replay error: ${error.message}`); replayState.textContent = 'Historical replay could not be completed.';
  } finally { byId('run-replay').disabled = !selectedCandle; }
}

byId('replay-date').addEventListener('change', event => loadDate(event.target.value).catch(error => { replayState.textContent = `Could not load date: ${error.message}`; }));
byId('replay-candle').addEventListener('change', event => { const candle = replayCandles.find(row => row.candle_time_utc === event.target.value); if (candle) selectCandle(candle); });
byId('run-replay').addEventListener('click', runReplay);
canvas?.addEventListener('click', event => {
  if (draggedSinceDown) { draggedSinceDown = false; return; }
  if (!chartGeometry || !replayCandles.length) return;
  const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left;
  if (x < chartGeometry.left || x >= chartGeometry.right) return;
  const index = chartGeometry.visibleStart + Math.floor((x - chartGeometry.left) / chartGeometry.step);
  if (replayCandles[index]) selectCandle(replayCandles[index]);
});
canvas?.addEventListener('wheel', event => {
  if (!replayCandles.length || !chartGeometry) return;
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const fraction = Math.max(0, Math.min(1, (x - chartGeometry.left) / (chartGeometry.right - chartGeometry.left)));
  const amount = Math.max(1, Math.round(visibleCount * 0.2));
  const next = Math.max(1, Math.min(replayCandles.length, visibleCount + (event.deltaY < 0 ? -amount : amount)));
  const range = VIEWPORT?.zoomWindow(replayCandles.length, visibleStart, visibleCount, next, fraction)
    || { start: Math.round(anchor - fraction * next), count: next };
  visibleCount = range.count;
  visibleStart = range.start;
  drawCandles();
}, { passive: false });
canvas?.addEventListener('pointerdown', event => {
  if (event.button !== 0 || !chartGeometry) return;
  dragState = { x: event.clientX, start: visibleStart };
  draggedSinceDown = false;
  canvas.setPointerCapture?.(event.pointerId);
});
canvas?.addEventListener('pointermove', event => {
  if (!dragState || !chartGeometry) return;
  const dx = event.clientX - dragState.x;
  if (Math.abs(dx) > 4) draggedSinceDown = true;
  if (!draggedSinceDown) return;
  const shift = -Math.round(dx / chartGeometry.step);
  const range = VIEWPORT?.panWindow(replayCandles.length, dragState.start, visibleCount, shift)
    || { start: Math.max(0, Math.min(replayCandles.length - visibleCount, dragState.start + shift)) };
  visibleStart = range.start;
  drawCandles();
});
const finishDrag = () => { dragState = null; };
canvas?.addEventListener('pointerup', finishDrag);
canvas?.addEventListener('pointercancel', finishDrag);
byId('replay-chart-reset')?.addEventListener('click', () => {
  const range = VIEWPORT?.visibleWindow(replayCandles.length, 65)
    || { start: Math.max(0, replayCandles.length - 65), count: Math.min(65, replayCandles.length || 1) };
  visibleCount = Math.max(1, range.count);
  visibleStart = range.start;
  if (replayCandles.length) selectCandle(replayCandles[replayCandles.length - 1]);
  else drawCandles();
});
new ResizeObserver(drawCandles).observe(replayRoot);

(async function initialize() {
  try {
    replayState.textContent = 'Loading permitted historical dates…';
    const body = await getJson('/api/replay/dates');
    const select = byId('replay-date'); select.innerHTML = '';
    for (const day of body.dates) { const option = document.createElement('option'); option.value = day; option.textContent = day; select.append(option); }
    select.disabled = body.dates.length === 0;
    if (!body.dates.length) { replayState.textContent = 'No permitted historical dates are available.'; return; }
    await loadDate(body.dates[body.dates.length - 1]);
  } catch (error) { replayState.textContent = `Historical replay unavailable: ${error.message}`; }
})();
