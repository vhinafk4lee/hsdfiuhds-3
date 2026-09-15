import { parseBlacklist } from './blacklist.js';

function num(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) throw new Error(`${name} must be a number, got: ${raw}`);
  return n;
}

export function loadConfig() {
  const botToken = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;

  if (!botToken) throw new Error('TELEGRAM_BOT_TOKEN is required (get one from @BotFather)');
  if (!chatId) throw new Error('TELEGRAM_CHAT_ID is required (the channel/chat to post alerts to)');

  const windowMinutes = num('WINDOW_MINUTES', 1);
  if (![1, 5, 15].includes(windowMinutes)) {
    throw new Error(`WINDOW_MINUTES must be 1, 5 or 15, got: ${windowMinutes}`);
  }

  return {
    botToken,
    chatId,
    network: process.env.NETWORK || 'robinhood',
    thresholdUsd: num('VOLUME_THRESHOLD_USD', 300000),
    windowMinutes,
    pollIntervalSeconds: num('POLL_INTERVAL_SECONDS', 60),
    // 20 pools per page. The public API only tolerates a handful of requests a
    // minute from a cloud IP, so a cycle scans the hot pages plus a rotating
    // slice of the rest rather than all of them.
    maxPoolPages: num('MAX_POOL_PAGES', 8),
    hotPages: num('HOT_PAGES', 2),
    rotatingPages: num('ROTATING_PAGES', 2),
    useTrending: process.env.USE_TRENDING !== '0',
    maxCandidates: num('MAX_CANDIDATES_PER_CYCLE', 10),
    blacklist: parseBlacklist(process.env.BLACKLIST),
  };
}
