import { loadConfig } from './config.js';
import { fetchCandles } from './geckoterminal.js';
import { sendMessage, formatAlert, formatAge } from './telegram.js';
import { isBlacklisted } from './blacklist.js';
import { createScanner } from './scanner.js';
import { createAlertGate } from './alerts.js';
import { fetchStockTokens } from './stocktokens.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function runCycle(config, scanner, gate, blacklist) {
  const { pools, pages, trending } = await scanner.scan();

  // A candle of `windowMinutes` that crossed the threshold is always contained in
  // the wider rolling window below, so filtering on it cannot drop a real hit.
  const prefilter = config.windowMinutes <= 5 ? 'volume5m' : 'volume1h';
  const watched = [];
  const skipped = [];
  for (const pool of pools) {
    if (isBlacklisted(pool, blacklist)) skipped.push(pool.baseSymbol ?? pool.name);
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
      `candidates=${candidates.length} held=${held.length}`,
  );

  // Questions about a miss always arrive after the fact, so record what the
  // scan actually saw: without this there is no way to tell later whether a
  // token was below the threshold, blacklisted, or never in view at all.
  const label = (pool) =>
    `${isBlacklisted(pool, blacklist) ? '*' : ''}${pool.baseSymbol ?? pool.address}`;
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
