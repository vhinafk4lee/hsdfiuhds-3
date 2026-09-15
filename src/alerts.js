/**
 * Decides whether a crossing is worth sending.
 *
 * A pump crosses the threshold minute after minute, and each of those minutes
 * is a distinct candle, so without a hold one move floods the channel. The hold
 * is per token rather than per pool: the same token trading in two pools is
 * still the same call.
 */
export function createAlertGate({ cooldownMs, now = () => Date.now() }) {
  const seenCandles = new Set();
  const lastAlertAt = new Map();

  function tokenKey(pool) {
    return pool.baseAddress ?? pool.address;
  }

  return {
    /** Lets a caller drop a token before paying for its confirmation request. */
    isHeld(pool) {
      const previous = lastAlertAt.get(tokenKey(pool));
      return previous !== undefined && now() - previous < cooldownMs;
    },

    allow(pool, candle) {
      const candleKey = `${pool.address}:${candle.timestamp}`;
      if (seenCandles.has(candleKey)) return false;

      const at = now();
      const previous = lastAlertAt.get(tokenKey(pool));
      if (previous !== undefined && at - previous < cooldownMs) {
        seenCandles.add(candleKey);
        return false;
      }

      seenCandles.add(candleKey);
      lastAlertAt.set(tokenKey(pool), at);

      if (seenCandles.size > 5000) {
        for (const old of [...seenCandles].slice(0, 1000)) seenCandles.delete(old);
      }
      for (const [key, time] of lastAlertAt) {
        if (at - time >= cooldownMs) lastAlertAt.delete(key);
      }

      return true;
    },
  };
}
