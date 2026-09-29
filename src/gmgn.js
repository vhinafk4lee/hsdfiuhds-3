import crypto from 'node:crypto';

const HOST = 'https://openapi.gmgn.ai';

/**
 * Read-only GMGN OpenAPI request. Market and token routes need only the key:
 * a fresh timestamp (checked to ±5s) and a one-off client_id (replays within
 * 7s are rejected) go in the query. Signing is for swap and order routes only,
 * which this bot never calls.
 */
async function get(apiKey, path, query) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null) continue;
    // Arrays travel as repeated keys (platforms=a&platforms=b).
    for (const item of Array.isArray(value) ? value : [value]) params.append(key, String(item));
  }
  params.set('timestamp', String(Math.floor(Date.now() / 1000)));
  params.set('client_id', crypto.randomUUID());

  const res = await fetch(`${HOST}${path}?${params}`, {
    headers: { 'X-APIKEY': apiKey, 'Content-Type': 'application/json' },
  });
  const text = await res.text();

  let json;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error(`GMGN ${path} -> HTTP ${res.status}: ${text.slice(0, 200)}`);
  }
  if (json.code !== 0) {
    throw new Error(
      `GMGN ${path} -> HTTP ${res.status} code=${json.code} ${json.error ?? ''} ${json.message ?? ''}`.trim(),
    );
  }
  return json.data;
}

/**
 * Tokens ranked by traded USD over the last `interval`. With interval=1m and
 * min_volume at the threshold this is the whole detection funnel in one call.
 */
export async function fetchRankData(apiKey, { chain, interval = '1m', limit = 20, minVolume, filters } = {}) {
  return get(apiKey, '/v1/market/rank', {
    chain,
    interval,
    order_by: 'volume',
    direction: 'desc',
    limit,
    min_volume: minVolume,
    // Omitted, the server applies the chain's defaults (on EVM: not_honeypot,
    // verified, renounced) — passing any list replaces them.
    filters: filters?.length ? filters : undefined,
  });
}

export async function fetchRank(apiKey, options) {
  return (await fetchRankData(apiKey, options))?.rank ?? [];
}
