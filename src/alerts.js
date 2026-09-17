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

    shouldSend(pool, candle) {
      if (seenCandles.has(`${pool.address}:${candle.timestamp}`)) return false;
      return !this.isHeld(pool);
    },

    /**
     * Call only after the message is actually delivered: recording a send that
     * failed would start the hold and bury the alert until the pump is over.
     */
    record(pool, candle) {
      const at = now();
      seenCandles.add(`${pool.address}:${candle.timestamp}`);
      lastAlertAt.set(tokenKey(pool), at);

      if (seenCandles.size > 5000) {
        for (const old of [...seenCandles].slice(0, 1000)) seenCandles.delete(old);
      }
      for (const [key, time] of lastAlertAt) {
        if (at - time >= cooldownMs) lastAlertAt.delete(key);
      }
    },
  };
}
