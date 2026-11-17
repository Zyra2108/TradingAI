const test = require('node:test');
const assert = require('node:assert/strict');
require('../live_app/static/alpaca_levels.js');
const levels = globalThis.ECLMarketLevels;

function sessionCandles(count = 24) {
  const start = Date.parse('2026-11-13T14:30:00Z') / 1000;
  return Array.from({ length: count }, (_, i) => {
    const close = i === count - 1 ? 102 : (i % 6 < 3 ? 101 : 103);
    const low = i % 4 === 0 ? 99.9 : close - 0.2;
    const high = i % 4 === 2 ? 104.1 : close + 0.2;
    return { time: start + i * 300, open: close - 0.05, high, low, close, volume: 100 + i };
  });
}

test('support and resistance zones use repeated actual high/low touches', () => {
  const candles = sessionCandles();
  const result = levels.calculateZones(candles, candles.at(-1).time);
  assert.ok(result.sample_count >= 20);
  assert.ok(result.support.length > 0);
  assert.ok(result.resistance.length > 0);
  assert.ok(result.support.every(zone => zone.high < candles.at(-1).close && zone.touches >= 2));
  assert.ok(result.resistance.every(zone => zone.low > candles.at(-1).close && zone.touches >= 2));
});

test('future candles cannot change zones for an earlier selected timestamp', () => {
  const candles = sessionCandles();
  const selected = candles[18].time;
  const before = levels.calculateZones(candles.slice(0, 19), selected);
  const withFutureExtreme = candles.slice(0, 19).concat({
    time: selected + 300, open: 102, high: 800, low: 1, close: 400,
  });
  assert.deepEqual(levels.calculateZones(withFutureExtreme, selected), before);
});

test('a missing five-minute candle or NY session boundary resets the zone sample', () => {
  const candles = sessionCandles();
  const first = candles.slice(0, 19);
  const latest = candles.at(-1);
  const afterGap = first.concat({ ...latest, time: first.at(-1).time + 600 });
  assert.equal(levels.calculateZones(afterGap, afterGap.at(-1).time).sample_count, 1);
  const nextSession = sessionCandles(1)[0];
  nextSession.time = Date.parse('2026-11-16T14:30:00Z') / 1000;
  const acrossSessions = first.concat(nextSession);
  assert.equal(levels.calculateZones(acrossSessions, nextSession.time).sample_count, 1);
});

test('no zones are shown without a valid selected completed candle', () => {
  const candles = sessionCandles();
  assert.equal(levels.calculateZones(candles, candles.at(-1).time + 300).sample_count, 0);
  assert.deepEqual(levels.calculateZones([], 0).support, []);
});
