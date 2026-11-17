const test = require('node:test');
const assert = require('node:assert/strict');
const viewport = require('../live_app/static/alpaca_chart_viewport.js');

test('initial live viewport shows a few actual candles with room at the right', () => {
  const range = viewport.initialRange(4);
  assert.deepEqual(range, { from: -0.5, to: 5 });
  assert.equal(range.to - 2, 3); // latest real candle index, followed by two blank slots
});

test('default window uses about 65 candles and fits all when fewer exist', () => {
  assert.deepEqual(viewport.visibleWindow(4), { start: 0, count: 4 });
  assert.deepEqual(viewport.visibleWindow(50), { start: 0, count: 50 });
  assert.deepEqual(viewport.visibleWindow(130), { start: 65, count: 65 });
});

test('new live data follows latest while preserving manual historical pan', () => {
  const following = viewport.appendedRange({ from: 35, to: 100 }, 100, 103);
  assert.deepEqual(following, { from: 39, to: 104 }); // same visible span plus right padding
  const panned = viewport.appendedRange({ from: 10, to: 55 }, 100, 103);
  assert.deepEqual(panned, { from: 10, to: 55 });
  assert.deepEqual(viewport.appendedRange(null, 0, 3), viewport.initialRange(3));
});

test('sparse actual candle timestamps do not create synthetic candles', () => {
  const observed = [100, 101, 106, 115];
  const range = viewport.visibleWindow(observed.length);
  assert.equal(range.count, observed.length);
  assert.deepEqual(observed.slice(range.start), [100, 101, 106, 115]);
});

test('spacing adapts to few candles and chart resizing', () => {
  const few = viewport.candleSpacing(1000, 4);
  const normal = viewport.candleSpacing(1000, 65);
  const narrow = viewport.candleSpacing(500, 65);
  const wide = viewport.candleSpacing(1400, 65);
  assert.ok(few > normal);
  assert.ok(wide > narrow);
  assert.ok(few <= 48 && narrow >= 5);
});

test('historical replay wheel zoom and drag pan remain inside available data', () => {
  assert.deepEqual(viewport.zoomWindow(130, 65, 65, 50, 0.5), { start: 73, count: 50 });
  assert.deepEqual(viewport.panWindow(130, 73, 50, -500), { start: 0, count: 50 });
  assert.deepEqual(viewport.panWindow(130, 73, 50, 500), { start: 80, count: 50 });
});

test('SMA uses actual closes and resets at genuine gaps and New York session boundaries', () => {
  const day = [
    { time: Date.parse('2026-09-24T13:30:00Z') / 1000, close: 10 },
    { time: Date.parse('2026-09-24T13:35:00Z') / 1000, close: 11 },
    { time: Date.parse('2026-09-24T13:40:00Z') / 1000, close: 12 },
    // Missing 13:45 is preserved and resets the 3-candle warmup.
    { time: Date.parse('2026-09-24T13:50:00Z') / 1000, close: 20 },
    { time: Date.parse('2026-09-24T13:55:00Z') / 1000, close: 21 },
    { time: Date.parse('2026-09-24T14:00:00Z') / 1000, close: 22 },
    // NY session change resets again.
    { time: Date.parse('2026-09-25T13:30:00Z') / 1000, close: 30 },
  ];
  const sma = viewport.simpleMovingAverage(day, 3);
  assert.deepEqual(sma, [
    { time: day[2].time, value: 11 },
    { time: day[5].time, value: 21 },
  ]);
});

test('indicator OFF returns no overlay points and latest live selection hides settlement', () => {
  const candles = Array.from({ length: 20 }, (_, i) => ({ time: 1790266200 + i * 300, close: 50 + i }));
  assert.equal(viewport.indicatorData(candles, 20, true).length, 1);
  assert.deepEqual(viewport.indicatorData(candles, 20, false), []);
  const latestLive = { time: candles[19].time, live: true };
  assert.equal(viewport.shouldShowSettledOutcome(latestLive, candles[19].time, true, false), false);
  assert.equal(viewport.shouldShowSettledOutcome(latestLive, candles[19].time, true, true), true);
  assert.equal(viewport.shouldShowSettledOutcome(latestLive, candles[19].time, false, false), true);
  assert.equal(viewport.shouldShowSettledOutcome({ time: candles[18].time, live: true }, candles[19].time, true, false), true);
  assert.equal(viewport.shouldShowSettledOutcome({ time: candles[0].time, live: false }, candles[19].time, true, false), true);
});

test('SMA values match an independent trailing-20 calculation and are viewport independent', () => {
  const candles = Array.from({ length: 28 }, (_, i) => ({ time: 1790266200 + i * 300, close: 100 + i / 10 }));
  const expected = [];
  for (let i = 19; i < candles.length; i++) {
    expected.push({ time: candles[i].time, value: candles.slice(i - 19, i + 1).reduce((sum, row) => sum + row.close, 0) / 20 });
  }
  const calculated = viewport.simpleMovingAverage(candles, 20);
  assert.deepEqual(calculated, expected);
  const full = viewport.indicatorData(candles, 20, true);
  viewport.visiblePriceRange(candles, { from: 0, to: 7 });
  viewport.visiblePriceRange(candles, { from: 20, to: 27 });
  assert.deepEqual(viewport.indicatorData(candles, 20, true), full);
});

test('SMA rejects null, undefined, empty, non-finite and non-positive closes', () => {
  const start = Date.parse('2026-09-24T13:30:00Z') / 1000;
  for (const invalidClose of [null, undefined, '', '   ', NaN, Infinity, -Infinity, 'NaN', 'Infinity', 0, -1]) {
    const candles = Array.from({ length: 19 }, (_, i) => ({ time: start + i * 300, close: 100 }));
    candles.push({ time: start + 19 * 300, close: invalidClose });
    assert.deepEqual(viewport.simpleMovingAverage(candles, 20), [], `invalid close ${String(invalidClose)} must not become an SMA input`);
  }
});

test('SMA requires exactly 20 valid closes and null in a 20-candle window resets warmup', () => {
  const start = Date.parse('2026-09-24T13:30:00Z') / 1000;
  const exactly20 = Array.from({ length: 20 }, (_, i) => ({ time: start + i * 300, close: 100 + i }));
  assert.deepEqual(viewport.simpleMovingAverage(exactly20, 20), [
    { time: exactly20[19].time, value: 109.5 },
  ]);

  const withNull = Array.from({ length: 40 }, (_, i) => ({
    time: start + i * 300,
    close: i === 19 ? null : 200 + i,
  }));
  const afterReset = viewport.simpleMovingAverage(withNull, 20);
  assert.deepEqual(afterReset, [
    { time: withNull[39].time, value: Array.from({ length: 20 }, (_, i) => 220 + i).reduce((a, b) => a + b, 0) / 20 },
  ]);
});

test('invalid close breaks the plotted SMA line and requires 20 new valid closes', () => {
  const start = Date.parse('2026-09-24T13:30:00Z') / 1000;
  const candles = Array.from({ length: 41 }, (_, i) => ({
    time: start + i * 300,
    close: i === 20 ? undefined : 100 + i,
  }));
  const line = viewport.indicatorData(candles, 20, true);
  assert.deepEqual(line, [
    { time: candles[19].time, value: 109.5 },
    { time: candles[21].time },
    { time: candles[40].time, value: 130.5 },
  ]);
  assert.equal(viewport.indicatorAvailability(candles, candles[20].time, 20).available, false);
  assert.equal(viewport.indicatorAvailability(candles, candles[40].time, 20).available, true);
});

test('SMA line data inserts whitespace at a missing-bucket reset and reports warmup reason', () => {
  const candles = [];
  for (let i = 0; i < 20; i++) candles.push({ time: 1790266200 + i * 300, close: 100 + i });
  for (let i = 0; i < 20; i++) candles.push({ time: 1790266200 + (i + 21) * 300, close: 200 + i });
  const series = viewport.indicatorData(candles, 20, true);
  assert.equal(series.length, 3);
  assert.equal(series[0].value, 109.5);
  assert.deepEqual(series[1], { time: candles[20].time });
  assert.equal(series[2].value, 209.5);
  assert.match(viewport.indicatorAvailability(candles, candles[22].time, 20).reason, /missing market data.*3\/20/i);
  assert.deepEqual(viewport.indicatorData(candles, 20, false), []);
});

test('SMA availability distinguishes new-session warmup from a data-gap reset', () => {
  const candles = Array.from({ length: 5 }, (_, i) => ({ time: Date.parse('2026-11-12T14:30:00Z') / 1000 + i * 300, close: 10 + i }))
    .concat(Array.from({ length: 4 }, (_, i) => ({ time: Date.parse('2026-11-13T13:30:00Z') / 1000 + i * 300, close: 20 + i })));
  assert.match(viewport.indicatorAvailability(candles, candles[6].time, 20).reason, /new session.*2\/20/i);
  const gap = Array.from({ length: 3 }, (_, i) => ({ time: Date.parse('2026-11-13T14:30:00Z') / 1000 + i * 300, close: 30 + i }))
    .concat(Array.from({ length: 4 }, (_, i) => ({ time: Date.parse('2026-11-13T14:50:00Z') / 1000 + i * 300, close: 40 + i })));
  assert.match(viewport.indicatorAvailability(gap, gap.at(-1).time, 20).reason, /missing market data.*4\/20/i);
});

test('price autoscale uses only the candles in the visible range', () => {
  const candles = Array.from({ length: 130 }, (_, i) => ({
    time: 1790266200 + i * 300,
    low: i < 65 ? 273.5 : 308.5,
    high: i < 65 ? 274.5 : 309.5,
    close: i < 65 ? 274 : 309,
  }));
  const latest = viewport.visiblePriceRange(candles, viewport.initialRange(candles.length));
  assert.ok(latest.minValue > 300, `latest autoscale min ${latest.minValue} should exclude distant $274 history`);
  assert.ok(latest.maxValue > 309.5 && latest.maxValue < 311);
  const historical = viewport.visiblePriceRange(candles, { from: 0, to: 64 });
  assert.ok(historical.minValue < 273.5 && historical.maxValue < 280);
  const zoomedOut = viewport.visiblePriceRange(candles, { from: 0, to: 129 });
  assert.ok(zoomedOut.minValue < 273.5 && zoomedOut.maxValue > 309.5);
});

test('visible price autoscale follows a live append without changing a manual range', () => {
  const candles = Array.from({ length: 70 }, (_, i) => ({ time: 1790266200 + i * 300, low: 308, high: 310 }));
  const pannedRange = { from: 0, to: 40 };
  const prior = viewport.visiblePriceRange(candles, pannedRange);
  candles.push({ time: candles.at(-1).time + 300, low: 319, high: 321, live: true });
  const stillPanned = viewport.visiblePriceRange(candles, pannedRange);
  assert.deepEqual(stillPanned, prior);
  const latest = viewport.visiblePriceRange(candles, viewport.initialRange(candles.length));
  assert.ok(latest.minValue < 308 && latest.maxValue > 321);
});
