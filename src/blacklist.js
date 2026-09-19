export const normalize = (value) => String(value ?? '').toLowerCase().replace(/\s+/g, '');

export function parseBlacklist(raw) {
  return new Set(
    (raw ?? '')
      .split(',')
      .map(normalize)
      .filter(Boolean),
  );
}

/** An entry matches a pair ("USDG / WETH"), a base token symbol, or either address. */
export function isBlacklisted(pool, blacklist) {
  if (blacklist.size === 0) return false;

  return [pool.name, pool.baseSymbol, pool.baseAddress, pool.address]
    .map(normalize)
    .some((field) => field && blacklist.has(field));
}
