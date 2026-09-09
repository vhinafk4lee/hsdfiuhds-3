const BASE = 'https://api.geckoterminal.com/api/v2';
const HEADERS = { Accept: 'application/json;version=20230302' };

async function get(path) {
  const res = await fetch(`${BASE}${path}`, { headers: HEADERS });
  if (!res.ok) throw new Error(`GeckoTerminal ${path} -> HTTP ${res.status}`);
  return res.json();
}

function tokenSymbols(included) {
  const byId = new Map();
  for (const item of included ?? []) {
    if (item?.type === 'token') byId.set(item.id, item.attributes ?? {});
  }
  return byId;
}

/**
 * Stage 1 of the funnel: every pool on the network with its rolling 5m volume.
 * A pool that traded the threshold within one minute necessarily shows at least
 * that much in its 5m window, so this list can never miss a candidate.
 */
export async function fetchPools(network, maxPages) {
  const pools = [];

  for (let page = 1; page <= maxPages; page++) {
    const body = await get(`/networks/${network}/pools?page=${page}`);
    const items = body?.data ?? [];
    if (items.length === 0) break;

    const tokens = tokenSymbols(body.included);

    for (const item of items) {
      const a = item?.attributes ?? {};
      const baseId = item?.relationships?.base_token?.data?.id;
      const quoteId = item?.relationships?.quote_token?.data?.id;

      pools.push({
        address: a.address,
        name: a.name,
        baseSymbol: tokens.get(baseId)?.symbol ?? null,
        baseAddress: tokens.get(baseId)?.address ?? null,
        quoteSymbol: tokens.get(quoteId)?.symbol ?? null,
        priceUsd: Number(a.base_token_price_usd) || null,
        liquidityUsd: Number(a.reserve_in_usd) || 0,
        volume5m: Number(a.volume_usd?.m5) || 0,
        volume1h: Number(a.volume_usd?.h1) || 0,
        volume24h: Number(a.volume_usd?.h24) || 0,
      });
    }

    if (!body?.links?.next) break;
  }

  return pools;
}

/**
 * Stage 2: exact per-candle volume in USD, newest candle first.
 */
export async function fetchCandles(network, poolAddress, windowMinutes, limit) {
  const timeframe = windowMinutes >= 60 ? 'hour' : 'minute';
  const aggregate = timeframe === 'hour' ? windowMinutes / 60 : windowMinutes;
  const body = await get(
    `/networks/${network}/pools/${poolAddress}/ohlcv/${timeframe}?aggregate=${aggregate}&limit=${limit}`,
  );

  const list = body?.data?.attributes?.ohlcv_list ?? [];
  return list.map(([timestamp, , , , close, volume]) => ({
    timestamp: Number(timestamp),
    close: Number(close),
    volumeUsd: Number(volume) || 0,
  }));
}
