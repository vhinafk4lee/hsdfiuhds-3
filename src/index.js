import { loadConfig } from './config.js';
import { fetchCandles } from './geckoterminal.js';
import { sendMessage, formatAlert, formatAge, formatGmgnAlert, assessRisk } from './telegram.js';
import { isBlacklisted, hasSkippedFeeTier, isAllowedDex } from './blacklist.js';
import { createScanner } from './scanner.js';
import { createAlertGate } from './alerts.js';
import { fetchStockTokens } from './stocktokens.js';
import { fetchRank } from './gmgn.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Stock tokens carry the issuer's suffix in their name even before the registry lists them. */
const isStockName = (name) => /robinhood token$/i.test(String(name ?? '').trim());

/**
 * One GMGN request replaces the whole GeckoTerminal funnel: its ranking is
 * already per token (all pools summed), in USD, over the last minute, and
 * carries GMGN's own rug and wash-trading scores for the message.
 */
async function runGmgnCycle(config, gate, blacklist) {
  const rank = await fetchRank(config.gmgnApiKey, {
    chain: config.gmgnChain,
    interval: '1m',
    // Filtered here rather than with min_volume, so the log shows the live top
    // even in a quiet minute and an empty result can only mean an empty market.
    limit: 50,
    filters: config.gmgnFilters,
  });

  const hits = [];
  const skipped = [];
  for (const token of rank) {
    if (Number(token.volume) < config.thresholdUsd) continue;
    // Shaped like a pool so the blacklist and the per-token hold apply unchanged.
    const asPool = { address: token.address, baseAddress: token.address, baseSymbol: token.symbol };
    if (isBlacklisted(asPool, blacklist) || isStockName(token.name)) skipped.push(token.symbol);
    else hits.push({ token, asPool });
  }

  console.log(
    `[${new Date().toISOString()}] gmgn items=${rank.length} hits=${hits.length} ` +
      `skipped=${skipped.length}${skipped.length ? `(${[...new Set(skipped)].join(',')})` : ''} ` +
      `top=${rank
        .slice(0, 3)
        .map((t) => `${t.symbol}=${Math.round(t.volume)}`)
        .join(' ')}`,
  );

  for (const { token, asPool } of hits) {
    const risk = assessRisk(token);
    const summary =
      `${token.symbol} ${Math.round(token.volume)} ${risk.label}` +
      `${risk.reasons.length ? `(${risk.reasons.join(',')})` : ''} ` +
      `age=${formatAge(token.creation_timestamp ? token.creation_timestamp * 1000 : null) ?? 'unknown'} ${token.address}`;

    if (gate.isHeld(asPool)) continue;

    // A high-risk verdict means rugged, faked or unsellable — the calls these
    // alerts kept getting wrong — so they stay out of the channel by default.
    if (risk.icon === '🔴' && !config.sendHighRisk) {
      console.log(`drop ${summary}`);
      continue;
    }

    if (config.silent) {
      console.log(`WOULD ALERT ${summary}`);
      continue;
    }

    try {
      await sendMessage(
        config.botToken,
        config.chatId,
        formatGmgnAlert({ token, windowMinutes: 1, chain: config.gmgnChain }),
      );
    } catch (error) {
      console.error(`send ${token.symbol} failed: ${error.message}`);
      continue;
    }

    // No candle here: the hold alone keeps one pump to one message.
    gate.record(asPool, { timestamp: Date.now() });
    console.log(`alert ${summary}`);
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

  const useGmgn = Boolean(config.gmgnApiKey);
  const pollSeconds = useGmgn ? config.gmgnPollSeconds : config.pollIntervalSeconds;

  if (useGmgn) {
    console.log(
      `watching ${config.gmgnChain} via GMGN: >= $${config.thresholdUsd} per 1m, polling every ${pollSeconds}s, ` +
        `high-risk tokens ${config.sendHighRisk ? 'sent' : 'dropped'}`,
    );
  } else {
    const coverageSeconds = scanner.coverageCycles() * pollSeconds;
    console.log(
      `watching ${config.network}: >= $${config.thresholdUsd} per ${config.windowMinutes}m window, ` +
        `polling every ${pollSeconds}s, every page revisited within ${coverageSeconds}s`,
    );

    // The 5m volume is what makes a spike detectable, so a page left unscanned
    // for longer than that can hide one.
    if (coverageSeconds > 300) {
      console.warn(`WARNING: full coverage takes ${coverageSeconds}s, longer than the 300s window`);
    }
  }

  if (config.silent) {
    console.log('SILENT mode: scanning and logging only, nothing is sent to Telegram');
  } else if (config.startupMessage) {
    await sendMessage(
      config.botToken,
      config.chatId,
      `✅ Bot started. Watching <b>${useGmgn ? config.gmgnChain : config.network}</b>: alerting on $${config.thresholdUsd.toLocaleString('en-US')}+ volume in ${useGmgn ? 1 : config.windowMinutes} min.`,
    );
  }

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

    const blacklist = new Set([...config.blacklist, ...stockTokens]);
    try {
      if (useGmgn) await runGmgnCycle(config, gate, blacklist);
      else await runCycle(config, scanner, gate, blacklist);
    } catch (error) {
      console.error('cycle failed:', error.message);
    }
    await sleep(pollSeconds * 1000);
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
