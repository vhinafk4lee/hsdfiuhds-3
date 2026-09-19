import { normalize } from './blacklist.js';

const ADDRESS = /^0x[0-9a-f]{40}$/i;

/**
 * Robinhood Chain carries hundreds of tokenised stocks, and they clear the
 * threshold constantly without being the kind of token these alerts are for.
 * Listing them by hand never ends, so take the issuer's own registry.
 *
 * The response shape is not documented, so walk it for anything that looks like
 * a ticker or a contract address rather than depending on a particular layout.
 */
function collect(node, found, depth = 0) {
  if (depth > 8 || node === null || typeof node !== 'object') return found;

  if (Array.isArray(node)) {
    for (const item of node) collect(item, found, depth + 1);
    return found;
  }

  for (const [key, value] of Object.entries(node)) {
    if (typeof value === 'string') {
      const lowerKey = key.toLowerCase();
      if (lowerKey === 'symbol' || lowerKey === 'ticker') {
        // A ticker is short; anything longer is a name that could collide with
        // a memecoin.
        if (value.length >= 1 && value.length <= 8) found.add(normalize(value));
      } else if (lowerKey.includes('address') || lowerKey.includes('contract')) {
        if (ADDRESS.test(value)) found.add(normalize(value));
      }
    } else {
      collect(value, found, depth + 1);
    }
  }

  return found;
}

export async function fetchStockTokens(url) {
  const res = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error(`stock registry ${url} -> HTTP ${res.status}`);

  return collect(await res.json(), new Set());
}
