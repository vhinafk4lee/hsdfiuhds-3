const BASE = 'https://api.geckoterminal.com/api/v2';
const HEADERS = { Accept: 'application/json;version=20230302' };
const PAGE_SIZE = 20;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function get(path, { retries = 1 } = {}) {
  const res = await fetch(`${BASE}${path}`, { headers: HEADERS });

  if (res.status === 429 && retries > 0) {
    const retryAfter = Number(res.headers.get('retry-after'));
    await sleep(Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : 5000);
    return get(path, { retries: retries - 1 });
  }

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

function parsePools(body) {
  const tokens = tokenSymbols(body?.included);

  return (body?.data ?? []).map((item) => {
    const a = item?.attributes ?? {};
    const baseId = item?.relationships?.base_token?.data?.id;
    const quoteId = item?.relationships?.quote_token?.data?.id;

    return {
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
    };
  });
}

/**
 * Pools moving right now, ranked over a short window rather than by 24h volume.
 * This is where a fresh token spikes: it can be trading hard this minute while
 * sitting far down the 24h ranking, out of reach of the paged scan.
 */
export async function fetchTrendingPools(network, duration = '5m') {
  const body = await get(
    `/networks/${network}/trending_pools?duration=${duration}&include=base_token,quote_token`,
    { retries: 0 },
  );

  return parsePools(body).filter((p) => p.address);
}

/**
 * Stage 1 of the funnel: pools with their rolling 5m volume. A pool that traded
 * the threshold within one minute necessarily shows at least that much in its
 * 5m window, so this list can never miss a candidate among the pools it covers.
 */
export async function fetchPoolPage(network, page) {
  // Without the include the response carries no token objects, leaving every
  // alert without a symbol or contract address.
  const body = await get(
    `/networks/${network}/pools?page=${page}&sort=h24_volume_usd_desc&include=base_token,quote_token`,
    { retries: 0 },
  );

  const pools = parsePools(body);
  const items = body?.data ?? [];
  const volumes = pools.map((p) => p.volume24h);
  // Pools shift between page requests as volumes update, so order only holds
  // within a page — checking it across pages produced false negatives.
  const sorted = volumes.every((v, i) => i === 0 || v <= volumes[i - 1]);

  return {
    pools: pools.filter((p) => p.address),
    sorted,
    lastVolume24h: volumes.at(-1) ?? 0,
    isLastPage: items.length < PAGE_SIZE,
  };
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
