import { loadConfig } from './config.js';
import { fetchPools, fetchCandles } from './geckoterminal.js';
import { sendMessage, formatAlert } from './telegram.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const alerted = new Set();

function remember(key) {
  alerted.add(key);
  if (alerted.size > 5000) {
    for (const old of [...alerted].slice(0, 1000)) alerted.delete(old);
  }
}

async function runCycle(config) {
  const pools = await fetchPools(config.network, config.maxPoolPages);

  // A candle of `windowMinutes` that crossed the threshold is always contained in
  // the wider rolling window below, so filtering on it cannot drop a real hit.
  const prefilter = config.windowMinutes <= 5 ? 'volume5m' : 'volume1h';
  const candidates = pools
    .filter((pool) => pool[prefilter] >= config.thresholdUsd)
    .sort((a, b) => b[prefilter] - a[prefilter])
    .slice(0, config.maxCandidates);

  console.log(
    `[${new Date().toISOString()}] pools=${pools.length} candidates=${candidates.length}`,
  );

  for (const pool of candidates) {
    const candles = await fetchCandles(config.network, pool.address, config.windowMinutes, 3);

    for (const candle of candles) {
      if (candle.volumeUsd < config.thresholdUsd) continue;

      const key = `${pool.address}:${candle.timestamp}`;
      if (alerted.has(key)) continue;

      await sendMessage(
        config.botToken,
        config.chatId,
        formatAlert({ pool, candle, windowMinutes: config.windowMinutes, network: config.network }),
      );
      remember(key);
      console.log(`alert ${pool.baseSymbol ?? pool.address} ${candle.volumeUsd}`);
    }

    await sleep(300);
  }
}

async function main() {
  const config = loadConfig();

  console.log(
    `watching ${config.network}: >= $${config.thresholdUsd} per ${config.windowMinutes}m window, ` +
      `polling every ${config.pollIntervalSeconds}s`,
  );

  await sendMessage(
    config.botToken,
    config.chatId,
    `✅ Бот запущен. Слежу за сетью <b>${config.network}</b>: алерт при объёме от $${config.thresholdUsd.toLocaleString('en-US')} за ${config.windowMinutes} мин.`,
  );

  for (;;) {
    try {
      await runCycle(config);
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
