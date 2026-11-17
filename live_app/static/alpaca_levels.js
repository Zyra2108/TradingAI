(function (root) {
  'use strict';
  const TZ = 'America/New_York';

  function sessionDate(epochSeconds) {
    const parts = new Intl.DateTimeFormat('en-CA', {
      timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit'
    }).formatToParts(new Date(Number(epochSeconds) * 1000));
    const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
    return `${values.year}-${values.month}-${values.day}`;
  }

  // Trailing 5-minute high/low touch clusters. Only rows at or before the
  // selected candle are considered; gaps and NY sessions start a new window.
  function calculateZones(candles, selectedTime, options = {}) {
    const lookback = options.lookback || 100;
    const bins = options.bins || 30;
    const minTouches = options.minTouches || 2;
    const selected = Number(selectedTime);
    const eligible = (candles || []).filter(row => Number(row.time) <= selected)
      .sort((a, b) => Number(a.time) - Number(b.time));
    if (!eligible.length || Number(eligible.at(-1).time) !== selected) return { support: [], resistance: [], sample_count: 0, method: 'trailing high/low touch clusters' };

    const end = eligible.at(-1);
    const date = sessionDate(end.time);
    const segment = [];
    let previousTime = null;
    for (let i = eligible.length - 1; i >= 0 && segment.length < lookback; i--) {
      const row = eligible[i];
      if (sessionDate(row.time) !== date || (previousTime !== null && previousTime - Number(row.time) !== 300)) break;
      const values = [row.open, row.high, row.low, row.close].map(Number);
      if (!values.every(Number.isFinite) || values.some(value => value <= 0) || Number(row.high) < Math.max(Number(row.open), Number(row.close), Number(row.low)) || Number(row.low) > Math.min(Number(row.open), Number(row.close), Number(row.high))) break;
      segment.push(row);
      previousTime = Number(row.time);
    }
    segment.reverse();
    if (segment.length < 8) return { support: [], resistance: [], sample_count: segment.length, method: 'trailing high/low touch clusters' };

    const min = Math.min(...segment.map(row => Number(row.low)));
    const max = Math.max(...segment.map(row => Number(row.high)));
    const width = Math.max((max - min) / bins, Number(end.close) * 0.00025);
    const lowTouches = new Map();
    const highTouches = new Map();
    const binFor = value => Math.floor((Number(value) - min) / width);
    for (const row of segment) {
      const lowBin = binFor(row.low), highBin = binFor(row.high);
      lowTouches.set(lowBin, (lowTouches.get(lowBin) || 0) + 1);
      highTouches.set(highBin, (highTouches.get(highBin) || 0) + 1);
    }
    const close = Number(end.close);
    function build(touches, kind) {
      const candidates = [...touches.entries()].map(([bin, count]) => ({
        low: min + bin * width, high: min + (bin + 1) * width,
        center: min + (bin + 0.5) * width, touches: count, bin,
      })).filter(zone => zone.touches >= minTouches &&
        (kind === 'support' ? zone.high < close : zone.low > close));
      candidates.sort((a, b) => b.touches - a.touches || Math.abs(a.center - close) - Math.abs(b.center - close));
      const chosen = [];
      for (const zone of candidates) {
        if (chosen.every(other => Math.abs(other.bin - zone.bin) > 1)) chosen.push(zone);
        if (chosen.length === 2) break;
      }
      return chosen.sort((a, b) => a.center - b.center).map(({ bin, ...zone }) => zone);
    }
    return {
      support: build(lowTouches, 'support'), resistance: build(highTouches, 'resistance'),
      sample_count: segment.length,
      method: `5-minute high/low touch clusters · ${segment.length}-candle continuous session window`,
    };
  }

  root.ECLMarketLevels = { calculateZones, sessionDate };
})(typeof window === 'undefined' ? globalThis : window);
