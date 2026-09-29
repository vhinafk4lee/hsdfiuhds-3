export const normalize = (value) => String(value ?? '').toLowerCase().replace(/\s+/g, '');

export function parseBlacklist(raw) {
  return new Set(
    (raw ?? '')
      .split(',')
      .map(normalize)
      .filter(Boolean),
  );
}

/**
 * Uniswap prints its fee tier in the pool name ("SHIT / WETH 0.05%"). The
 * cheapest tiers make churning volume nearly free, so they are where faked
 * volume shows up — worth excluding wholesale.
 */
export function hasSkippedFeeTier(pool, feeTiers) {
  if (feeTiers.size === 0) return false;

  const name = normalize(pool.name);
  return [...feeTiers].some((tier) => name.endsWith(tier));
}

/**
 * Which venue a token trades on says a lot about it: the churned volume lives
 * on one exchange while the real launches sit on another. An empty allowlist
 * means every venue is allowed.
 */
export function isAllowedDex(pool, allowed) {
  if (allowed.size === 0) return true;

  return allowed.has(normalize(pool.dex));
}

/** An entry matches a pair ("USDG / WETH"), a base token symbol, or either address. */
export function isBlacklisted(pool, blacklist) {
  if (blacklist.size === 0) return false;

  return [pool.name, pool.baseSymbol, pool.baseAddress, pool.address]
    .map(normalize)
    .some((field) => field && blacklist.has(field));
}
