import { loadConfig } from './config.js';
import { fetchCandles } from './geckoterminal.js';
import { sendMessage, formatAlert } from './telegram.js';
import { isBlacklisted } from './blacklist.js';
import { createScanner } from './scanner.js';
import { createAlertGate } from './alerts.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function runCycle(config, scanner, gate) {
  const { pools, pages, trending } = await scanner.scan();

  // A candle of `windowMinutes` that crossed the threshold is always contained in
  // the wider rolling window below, so filtering on it cannot drop a real hit.
  const prefilter = config.windowMinutes <= 5 ? 'volume5m' : 'volume1h';
  const watched = pools.filter((pool) => !isBlacklisted(pool, config.blacklist));
  const candidates = watched
    .filter((pool) => pool[prefilter] >= config.thresholdUsd)
    .sort((a, b) => b[prefilter] - a[prefilter])
    .slice(0, config.maxCandidates);

  console.log(
    `[${new Date().toISOString()}] trending=${trending} pages=${pages.join(',')} ` +
      `pools=${pools.length} skipped=${pools.length - watched.length} ` +
      `candidates=${candidates.length}`,
  );

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

      if (!gate.allow(pool, candle)) continue;

      await sendMessage(
        config.botToken,
        config.chatId,
        formatAlert({ pool, candle, windowMinutes: config.windowMinutes, network: config.network }),
      );
      console.log(`alert ${pool.baseSymbol ?? pool.address} ${candle.volumeUsd}`);
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

  await sendMessage(
    config.botToken,
    config.chatId,
    `✅ Bot started. Watching <b>${config.network}</b>: alerting on $${config.thresholdUsd.toLocaleString('en-US')}+ volume in ${config.windowMinutes} min.`,
  );

  for (;;) {
    try {
      await runCycle(config, scanner, gate);
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
