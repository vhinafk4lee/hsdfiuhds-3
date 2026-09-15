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

/**
 * Stage 1 of the funnel: every pool on the network with its rolling 5m volume.
 * A pool that traded the threshold within one minute necessarily shows at least
 * that much in its 5m window, so this list can never miss a candidate.
 */
export async function fetchPools(network, maxPages, thresholdUsd) {
  const pools = [];
  const seen = new Set();
  let descending = true;
  let previous = Infinity;

  for (let page = 1; page <= maxPages; page++) {
    if (page > 1) await sleep(1500);

    // Without the include the response carries no token objects, leaving every
    // alert without a symbol or contract address.
    const body = await get(
      `/networks/${network}/pools?page=${page}&sort=h24_volume_usd_desc&include=base_token,quote_token`,
    );
    const items = body?.data ?? [];
    if (items.length === 0) break;

    const tokens = tokenSymbols(body.included);

    for (const item of items) {
      const a = item?.attributes ?? {};
      const baseId = item?.relationships?.base_token?.data?.id;
      const quoteId = item?.relationships?.quote_token?.data?.id;

      if (!a.address || seen.has(a.address)) continue;
      seen.add(a.address);

      const volume24h = Number(a.volume_usd?.h24) || 0;
      if (volume24h > previous) descending = false;
      previous = volume24h;

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
        volume24h,
      });
    }

    if (process.env.DEBUG_SCAN === '1') {
      const v = items.map((i) => Number(i?.attributes?.volume_usd?.h24) || 0);
      console.log(
        `scan page=${page} items=${items.length} first=${Math.round(v[0])} ` +
          `last=${Math.round(v.at(-1))} descending=${descending}`,
      );
    }

    // A window that trades the threshold sits inside the last 24h, so a pool
    // below it in 24h volume cannot hold one. Sorted descending, everything
    // after this point is below it too — but only stop if the data really came
    // back in that order, otherwise keep paging rather than trust the sort.
    if (descending && previous < thresholdUsd) break;

    // links.next is not always present, and trusting it truncated the scan to
    // the first page. A short page is the reliable end-of-list signal.
    if (items.length < PAGE_SIZE) break;
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
