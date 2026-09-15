const normalize = (value) => String(value ?? '').toLowerCase().replace(/\s+/g, '');

export function parseBlacklist(raw) {
  return (raw ?? '')
    .split(',')
    .map(normalize)
    .filter(Boolean);
}

/** An entry matches a pair ("USDG / WETH"), a base token symbol, or either address. */
export function isBlacklisted(pool, blacklist) {
  if (blacklist.length === 0) return false;

  const fields = [pool.name, pool.baseSymbol, pool.baseAddress, pool.address]
    .map(normalize)
    .filter(Boolean);
  return blacklist.some((entry) => fields.includes(entry));
}
