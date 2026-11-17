/* Shared viewport math for the actual live and historical ECL candles. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ECLChartViewport = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  const DEFAULT_VISIBLE_CANDLES = 65;
  const RIGHT_BREATHING_ROOM = 2;

  function initialRange(count, desired = DEFAULT_VISIBLE_CANDLES) {
    const n = Math.max(0, Math.floor(Number(count) || 0));
    if (!n) return null;
    const visible = Math.min(n, Math.max(1, Math.floor(desired)));
    const to = n - 1 + RIGHT_BREATHING_ROOM;
    return { from: Math.max(-0.5, n - visible - 0.5), to };
  }

  function appendedRange(range, oldCount, newCount, desired = DEFAULT_VISIBLE_CANDLES, slack = 2) {
    const oldN = Math.max(0, Math.floor(Number(oldCount) || 0));
    const newN = Math.max(0, Math.floor(Number(newCount) || 0));
    if (!newN) return null;
    if (!range || oldN < 1) return initialRange(newN, desired);
    const span = Math.max(1, range.to - range.from);
    const wasFollowing = range.to >= oldN - 1 - slack;
    if (wasFollowing) {
      const to = newN - 1 + RIGHT_BREATHING_ROOM;
      return { from: to - span, to };
    }
    // Appended bars must not move a chart the user has panned into history.
    return { from: range.from, to: range.to };
  }

  function candleSpacing(plotWidth, count, desired = DEFAULT_VISIBLE_CANDLES) {
    const n = Math.max(1, Math.min(Math.floor(Number(count) || 1), Math.floor(desired)));
    const width = Math.max(1, Number(plotWidth) || 1);
    return Math.max(5, Math.min(48, width / (n + 1.75)));
  }

  function visibleWindow(count, desired = DEFAULT_VISIBLE_CANDLES) {
    const n = Math.max(0, Math.floor(Number(count) || 0));
    const visible = Math.min(n, Math.max(1, Math.floor(desired)));
    return { start: Math.max(0, n - visible), count: visible };
  }

  function zoomWindow(total, start, visible, nextVisible, anchorFraction) {
    const n = Math.max(0, Math.floor(Number(total) || 0));
    if (!n) return { start: 0, count: 0 };
    const oldCount = Math.max(1, Math.min(n, Math.floor(Number(visible) || 1)));
    const count = Math.max(1, Math.min(n, Math.floor(Number(nextVisible) || 1)));
    const fraction = Math.max(0, Math.min(1, Number(anchorFraction) || 0));
    const anchor = Math.max(0, Number(start) || 0) + fraction * oldCount;
    return { start: Math.max(0, Math.min(n - count, Math.round(anchor - fraction * count))), count };
  }

  function panWindow(total, start, visible, shift) {
    const n = Math.max(0, Math.floor(Number(total) || 0));
    const count = Math.max(0, Math.min(n, Math.floor(Number(visible) || 0)));
    return { start: Math.max(0, Math.min(n - count, Math.round((Number(start) || 0) + (Number(shift) || 0)))), count };
  }

  function sessionDateKey(epochSeconds) {
    const parts = new Intl.DateTimeFormat('en-CA', {
      timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit'
    }).formatToParts(new Date(Number(epochSeconds) * 1000));
    const part = type => parts.find(item => item.type === type)?.value || '';
    return `${part('year')}-${part('month')}-${part('day')}`;
  }

  function validPositiveNumber(value) {
    if (value === null || value === undefined) return null;
    if (typeof value !== 'number' && typeof value !== 'string') return null;
    if (typeof value === 'string' && value.trim() === '') return null;
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number : null;
  }

  function validCandleTime(value) {
    if (value === null || value === undefined) return null;
    if (typeof value !== 'number' && typeof value !== 'string') return null;
    if (typeof value === 'string' && value.trim() === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  // A trailing SMA over actual candles only. A new NYSE session or any missing
  // five-minute candle resets warmup; no whitespace candle is synthesized.
  function simpleMovingAverage(candles, period = 20) {
    const n = Math.max(1, Math.floor(Number(period) || 1));
    const result = [];
    let segment = [];
    let previousTime = null;
    let previousDay = null;
    for (const candle of candles) {
      const time = validCandleTime(candle?.time);
      const close = validPositiveNumber(candle?.close);
      if (time === null || close === null) {
        // Invalid source observations break continuity; they are never turned
        // into zeroes or skipped over as if the close had been observed.
        segment = [];
        previousTime = null;
        previousDay = null;
        continue;
      }
      const day = sessionDateKey(time);
      if (previousTime !== null && (time - previousTime !== 300 || day !== previousDay)) segment = [];
      segment.push(close);
      if (segment.length > n) segment.shift();
      if (segment.length === n) {
        result.push({ time, value: segment.reduce((sum, value) => sum + value, 0) / n });
      }
      previousTime = time;
      previousDay = day;
    }
    return result;
  }

  function indicatorData(candles, period = 20, enabled = true) {
    if (!enabled) return [];
    const n = Math.max(1, Math.floor(Number(period) || 1));
    const values = new Map(simpleMovingAverage(candles, n).map(row => [row.time, row.value]));
    const segmentByTime = new Map();
    let segment = 0, previousTime = null, previousDay = null;
    for (const candle of candles) {
      const time = validCandleTime(candle?.time);
      const close = validPositiveNumber(candle?.close);
      if (time === null || close === null) {
        segment++;
        previousTime = null;
        previousDay = null;
        continue;
      }
      const day = sessionDateKey(time);
      const beginsSegment = previousTime === null || time - previousTime !== 300 || day !== previousDay;
      if (previousTime !== null && beginsSegment) segment++;
      const previousSegment = previousTime === null ? null : segmentByTime.get(previousTime);
      segmentByTime.set(time, { id: segment, start: beginsSegment ? time : previousSegment.start });
      previousTime = time;
      previousDay = day;
    }
    const result = [];
    let previousValueSegment = null;
    for (const candle of candles) {
      const time = Number(candle.time);
      if (!values.has(time)) continue;
      const currentSegment = segmentByTime.get(time);
      if (previousValueSegment !== null && currentSegment.id !== previousValueSegment) {
        // Lightweight Charts treats a time-only item as a whitespace point,
        // breaking the line instead of drawing through a real data gap.
        result.push({ time: currentSegment.start });
      }
      result.push({ time, value: values.get(time) });
      previousValueSegment = currentSegment.id;
    }
    return result;
  }

  function indicatorAvailability(candles, time, period = 20) {
    const target = Number(time);
    const index = candles.findIndex(candle => validCandleTime(candle?.time) === target);
    if (index < 0) return { available: false, count: 0, reason: 'Candle is not in the available timeline.' };
    if (validPositiveNumber(candles[index]?.close) === null) {
      return { available: false, count: 0, reason: 'Invalid or missing close; waiting for 20 consecutive valid completed candles.' };
    }
    const n = Math.max(1, Math.floor(Number(period) || 1));
    let count = 1;
    while (index - count >= 0) {
      const current = candles[index - count + 1], previous = candles[index - count];
      if (validPositiveNumber(previous?.close) === null
          || validPositiveNumber(current?.close) === null
          || validCandleTime(current?.time) - validCandleTime(previous?.time) !== 300
          || sessionDateKey(current.time) !== sessionDateKey(previous.time)) break;
      count++;
    }
    if (count >= n) return { available: true, count, reason: '' };
    const firstIndex = index - count;
    if (firstIndex < 0 || sessionDateKey(candles[firstIndex].time) !== sessionDateKey(candles[index].time)) {
      return { available: false, count, reason: `New session warm-up: ${count}/${n} consecutive completed candles.` };
    }
    return { available: false, count, reason: `Reset after missing market data: ${count}/${n} consecutive completed candles.` };
  }

  function shouldShowSettledOutcome(candle, latestLiveTime, marketIsOpen, predictionSettled) {
    const latestForwardSignal = Boolean(candle) && candle.live === true
      && Number(candle.time) === Number(latestLiveTime) && marketIsOpen && !predictionSettled;
    return Boolean(candle) && !latestForwardSignal;
  }

  function visiblePriceRange(candles, logicalRange, indicatorByTime = new Map(), includeIndicator = true, paddingRatio = 0.08) {
    if (!candles?.length) return null;
    const from = logicalRange && Number.isFinite(logicalRange.from)
      ? Math.max(0, Math.ceil(logicalRange.from)) : Math.max(0, candles.length - DEFAULT_VISIBLE_CANDLES);
    const toExclusive = logicalRange && Number.isFinite(logicalRange.to)
      ? Math.min(candles.length, Math.floor(logicalRange.to) + 1) : candles.length;
    if (toExclusive <= from) return null;
    let minValue = Infinity, maxValue = -Infinity;
    for (let i = from; i < toExclusive; i++) {
      const candle = candles[i];
      const low = Number(candle.low), high = Number(candle.high);
      if (Number.isFinite(low)) minValue = Math.min(minValue, low);
      if (Number.isFinite(high)) maxValue = Math.max(maxValue, high);
      if (includeIndicator && indicatorByTime?.has(Number(candle.time))) {
        const value = Number(indicatorByTime.get(Number(candle.time)));
        if (Number.isFinite(value)) { minValue = Math.min(minValue, value); maxValue = Math.max(maxValue, value); }
      }
    }
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) return null;
    const span = maxValue - minValue;
    const padding = Math.max(span * Math.max(0, Number(paddingRatio) || 0), Math.abs((minValue + maxValue) / 2) * 0.001);
    return { minValue: minValue - padding, maxValue: maxValue + padding };
  }

  return { DEFAULT_VISIBLE_CANDLES, RIGHT_BREATHING_ROOM, initialRange, appendedRange,
    candleSpacing, visibleWindow, zoomWindow, panWindow, simpleMovingAverage,
    indicatorData, indicatorAvailability, shouldShowSettledOutcome, visiblePriceRange };
});
