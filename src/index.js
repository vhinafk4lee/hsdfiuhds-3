import { loadConfig } from './config.js';
import { fetchCandles } from './geckoterminal.js';
import { sendMessage, formatAlert, formatAge } from './telegram.js';
import { isBlacklisted, hasSkippedFeeTier, isAllowedDex } from './blacklist.js';
import { createScanner } from './scanner.js';
import { createAlertGate } from './alerts.js';
import { fetchStockTokens } from './stocktokens.js';
import { fetchRank, fetchRankData } from './gmgn.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let gmgnRawLogged = false;

/**
 * Logs GMGN's 1m ranking beside our own scan, so the two can be compared on the
 * same minutes before GMGN is trusted with alerts. Its field names are taken
 * from its docs, which contradict themselves in places — the first response is
 * dumped raw once so the real shape can be checked.
 */
async function shadowGmgn(config, blacklist) {
  let rank;
  try {
    rank = await fetchRank(config.gmgnApiKey, {
      chain: config.gmgnChain,
      interval: '1m',
      limit: 20,
      filters: config.gmgnFilters,
    });
  } catch (error) {
    console.error(`gmgn: ${error.message}`);
    return;
  }

  if (!gmgnRawLogged) {
    gmgnRawLogged = true;
    console.log(`gmgn raw (${rank.length} items): ${JSON.stringify(rank.slice(0, 2)).slice(0, 4000)}`);
  }

  const pct = (value) => (value === undefined || value === null || value === '' ? '?' : Math.round(value * 100));
  const describe = (item) => {
    const excluded = isBlacklisted({ baseSymbol: item.symbol, baseAddress: item.address }, blacklist);
    return (
      `${excluded ? '*' : ''}${item.symbol}=${Math.round(item.volume)}` +
      `(liq=${Math.round(item.liquidity ?? 0)} rug=${pct(item.rug_ratio)} wash=${item.is_wash_trading ? 1 : 0} ` +
      `bundler=${pct(item.bundler_rate)} ex=${item.exchange ?? item.launchpad_platform ?? '?'} ` +
      `age=${formatAge(item.creation_timestamp ? item.creation_timestamp * 1000 : null) ?? '?'})`
    );
  };

  console.log(`gmgn1m ${rank.slice(0, 5).map(describe).join(' ')}`);
  for (const item of rank.filter((i) => Number(i.volume) >= config.thresholdUsd)) {
    console.log(`gmgn-hit ${describe(item)} ${item.address}`);
  }
}

/**
 * An empty ranking can mean a quiet minute or filters that exclude the whole
 * chain, and the logs cannot tell those apart. Asking a few variants side by
 * side at startup shows which it is.
 */
async function probeGmgn(config) {
  const variants = [
    ['1m', []],
    ['5m', []],
    ['24h', []],
    ['1m', ['not_honeypot']],
    ['24h', ['not_honeypot']],
  ];
  for (const [interval, filters] of variants) {
    const label = `${interval} filters=${filters.join('+') || 'default'}`;
    try {
      const data = await fetchRankData(config.gmgnApiKey, {
        chain: config.gmgnChain,
        interval,
        limit: 10,
        filters,
      });
      const rank = data?.rank ?? [];
      console.log(
        `gmgn probe ${label}: keys=${Object.keys(data ?? {}).join(',') || 'none'} items=${rank.length} ` +
          rank
            .slice(0, 5)
            .map((i) => `${i.symbol}=${Math.round(i.volume ?? 0)}`)
            .join(' '),
      );
    } catch (error) {
      console.error(`gmgn probe ${label}: ${error.message}`);
    }
    await sleep(1500);
  }
}

async function runCycle(config, scanner, gate, blacklist) {
  const { pools, pages, trending } = await scanner.scan();

  // A candle of `windowMinutes` that crossed the threshold is always contained in
  // the wider rolling window below, so filtering on it cannot drop a real hit.
  const prefilter = config.windowMinutes <= 5 ? 'volume5m' : 'volume1h';
  const excluded = (pool) =>
    isBlacklisted(pool, blacklist) || hasSkippedFeeTier(pool, config.skipFeeTiers);

  const watched = [];
  const skipped = [];
  // Counted apart so the cost of each filter is visible: the fee-tier and venue
  // rules can quietly remove most of the real pools, not just the churned ones.
  let byTier = 0;
  let byDex = 0;
  for (const pool of pools) {
    if (isBlacklisted(pool, blacklist)) skipped.push(pool.baseSymbol ?? pool.name);
    else if (hasSkippedFeeTier(pool, config.skipFeeTiers)) byTier++;
    else if (!isAllowedDex(pool, config.dexAllowlist)) byDex++;
    else watched.push(pool);
  }
  // A token on hold cannot produce a message, so confirming it would spend a
  // request the API barely has — and those rejections were costing other
  // tokens their alerts.
  const crossing = watched.filter((pool) => pool[prefilter] >= config.thresholdUsd);
  const held = crossing.filter((pool) => gate.isHeld(pool));
  const candidates = crossing
    .filter((pool) => !gate.isHeld(pool))
    .sort((a, b) => b[prefilter] - a[prefilter])
    .slice(0, config.maxCandidates);

  console.log(
    `[${new Date().toISOString()}] trending=${trending} pages=${pages.join(',')} ` +
      `pools=${pools.length} skipped=${skipped.length}${skipped.length ? `(${[...new Set(skipped)].join(',')})` : ''} ` +
      `bytier=${byTier} bydex=${byDex} candidates=${candidates.length} held=${held.length}`,
  );

  // The venue ids have to be read off the live network before an allowlist can
  // be set from them, so report what this scan actually saw.
  const venues = new Map();
  for (const pool of pools) venues.set(pool.dex ?? 'unknown', (venues.get(pool.dex ?? 'unknown') ?? 0) + 1);
  console.log(
    'dex ' +
      [...venues]
        .sort((a, b) => b[1] - a[1])
        .map(([dex, count]) => `${dex}=${count}`)
        .join(' '),
  );

  // Questions about a miss always arrive after the fact, so record what the
  // scan actually saw: without this there is no way to tell later whether a
  // token was below the threshold, blacklisted, or never in view at all.
  const label = (pool) =>
    `${excluded(pool) ? '*' : ''}${pool.baseSymbol ?? pool.address}`;
  console.log(
    'top5m ' +
      [...pools]
        .sort((a, b) => b.volume5m - a.volume5m)
        .slice(0, 5)
        .map((p) => `${label(p)}=${Math.round(p.volume5m)}`)
        .join(' '),
  );

  for (const trace of (process.env.DEBUG_TOKEN ?? '')
    .split(',')
    .map((t) => t.trim().toLowerCase())
    .filter(Boolean)) {
    const seen = pools.filter(
      (p) => p.baseAddress?.toLowerCase() === trace || p.address?.toLowerCase() === trace,
    );
    console.log(
      seen.length === 0
        ? `trace ${trace}: not in this scan`
        : seen
            .map(
              (p) =>
                `trace ${label(p)} (${trace.slice(0, 10)}): 5m=${Math.round(p.volume5m)} ` +
                `1h=${Math.round(p.volume1h)} 24h=${Math.round(p.volume24h)}`,
            )
            .join(' | '),
    );
  }

  if (config.gmgnApiKey) await shadowGmgn(config, blacklist);

  for (const pool of candidates) {
    let candles;
    try {
      candles = await fetchCandles(config.network, pool.address, config.windowMinutes, 3);
    } catch (error) {
      // The spike stays in the 5m window for several cycles, so a failed
      // confirmation here is retried rather than lost.
      console.error(`confirm ${pool.baseSymbol ?? pool.address} failed: ${error.message}`);
      continue;
    }

    for (const candle of candles) {
      if (candle.volumeUsd < config.thresholdUsd) continue;

      if (!gate.shouldSend(pool, candle)) continue;

      if (config.silent) {
        console.log(`WOULD ALERT ${pool.baseSymbol ?? pool.address} ${candle.volumeUsd}`);
        continue;
      }

      try {
        await sendMessage(
          config.botToken,
          config.chatId,
          formatAlert({
            pool,
            candle,
            windowMinutes: config.windowMinutes,
            network: config.network,
          }),
        );
      } catch (error) {
        // Delivery can fail for reasons that have nothing to do with this pool
        // (a renamed channel, Telegram being down), so keep the rest of the
        // cycle running and leave the alert unrecorded for a later retry.
        console.error(`send ${pool.baseSymbol ?? pool.address} failed: ${error.message}`);
        continue;
      }

      gate.record(pool, candle);
      console.log(
        `alert ${pool.baseSymbol ?? pool.address} ${candle.volumeUsd} ` +
          `age=${formatAge(pool.createdAt) ?? 'unknown'}`,
      );
    }

    await sleep(300);
  }
}

async function main() {
  const config = loadConfig();

  const scanner = createScanner({
    network: config.network,
    hotPages: config.hotPages,
    rotatingPages: config.rotatingPages,
    maxPages: config.maxPoolPages,
    thresholdUsd: config.thresholdUsd,
    useTrending: config.useTrending,
  });

  const gate = createAlertGate({ cooldownMs: config.alertCooldownMinutes * 60_000 });

  const coverageSeconds = scanner.coverageCycles() * config.pollIntervalSeconds;
  console.log(
    `watching ${config.network}: >= $${config.thresholdUsd} per ${config.windowMinutes}m window, ` +
      `polling every ${config.pollIntervalSeconds}s, every page revisited within ${coverageSeconds}s`,
  );

  // The 5m volume is what makes a spike detectable, so a page left unscanned
  // for longer than that can hide one.
  if (coverageSeconds > 300) {
    console.warn(`WARNING: full coverage takes ${coverageSeconds}s, longer than the 300s window`);
  }

  if (config.silent) {
    console.log('SILENT mode: scanning and logging only, nothing is sent to Telegram');
  } else if (config.startupMessage) {
    await sendMessage(
      config.botToken,
      config.chatId,
      `✅ Bot started. Watching <b>${config.network}</b>: alerting on $${config.thresholdUsd.toLocaleString('en-US')}+ volume in ${config.windowMinutes} min.`,
    );
  }

  if (config.gmgnApiKey) await probeGmgn(config);

  let stockTokens = new Set();
  let refreshStockTokensAt = 0;

  for (;;) {
    // New stock tokens keep being issued, and a failed fetch must not leave the
    // category unfiltered for the life of the process — so retry it, sooner
    // after a failure than after a success.
    if (config.excludeStockTokens && Date.now() >= refreshStockTokensAt) {
      try {
        stockTokens = await fetchStockTokens(config.stockRegistryUrl);
        refreshStockTokensAt = Date.now() + 12 * 60 * 60_000;
        console.log(`stock tokens: excluding ${stockTokens.size} entries from the registry`);
      } catch (error) {
        refreshStockTokensAt = Date.now() + 10 * 60_000;
        console.error(`stock tokens: ${error.message}`);
      }
    }

    try {
      await runCycle(config, scanner, gate, new Set([...config.blacklist, ...stockTokens]));
    } catch (error) {
      console.error('cycle failed:', error.message);
    }
    await sleep(config.pollIntervalSeconds * 1000);
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
