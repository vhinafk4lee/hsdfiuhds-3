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

  const withCoinGecko = Boolean(process.env.COINGECKO_API_KEY);

  return {
    botToken,
    chatId,
    network: process.env.NETWORK || 'robinhood',
    thresholdUsd: num('VOLUME_THRESHOLD_USD', 200000),
    alertCooldownMinutes: num('ALERT_COOLDOWN_MINUTES', 30),
    windowMinutes,
    pollIntervalSeconds: num('POLL_INTERVAL_SECONDS', 60),
    // A CoinGecko key serves the same data at ~30 requests a minute instead of
    // the few the keyless API allows, so the scan defaults below go deeper.
    coingeckoApiKey: process.env.COINGECKO_API_KEY || null,
    coingeckoPlan: process.env.COINGECKO_PLAN === 'pro' ? 'pro' : 'demo',
    // 20 pools per page. The public API throttles a cloud IP down to a couple
    // of requests a minute, so a cycle scans the hot pages plus a rotating
    // slice of the rest rather than all of them.
    maxPoolPages: num('MAX_POOL_PAGES', withCoinGecko ? 10 : 4),
    hotPages: num('HOT_PAGES', withCoinGecko ? 2 : 1),
    rotatingPages: num('ROTATING_PAGES', withCoinGecko ? 2 : 1),
    useTrending: process.env.USE_TRENDING !== '0',
    // Investigate without posting: everything runs and is logged, nothing
    // reaches Telegram.
    silent: process.env.SILENT === '1',
    // Every redeploy restarts the process, so the startup notice is one message
    // per deploy in the channel.
    startupMessage: process.env.STARTUP_MESSAGE !== '0',
    // Each candidate costs a confirmation request, and the API tolerates only a
    // few per minute; the rest are picked up next cycle, while the 5m evidence
    // still stands.
    maxCandidates: num('MAX_CANDIDATES_PER_CYCLE', withCoinGecko ? 5 : 2),
    blacklist: parseBlacklist(process.env.BLACKLIST),
    // Fee tiers to ignore entirely, matched at the end of the pool name.
    skipFeeTiers: parseBlacklist(process.env.SKIP_FEE_TIERS),
    // Venues allowed to raise an alert. Empty means all of them; the cycle log
    // reports which venues the scan actually saw, so this can be set from
    // observed ids rather than guessed.
    dexAllowlist: parseBlacklist(process.env.DEX_ALLOWLIST),
    // Robinhood Chain carries hundreds of tokenised stocks; they are not what
    // these alerts are for, and listing them by hand never ends.
    excludeStockTokens: process.env.EXCLUDE_STOCK_TOKENS !== '0',
    stockRegistryUrl: process.env.STOCK_REGISTRY_URL || 'https://api.robinhood.com/rhj/assets',
    // With a GMGN key the bot takes its 1m volume ranking instead of scanning
    // GeckoTerminal; without one it falls back to the GeckoTerminal funnel.
    gmgnApiKey: process.env.GMGN_API_KEY || null,
    gmgnChain: process.env.GMGN_CHAIN || 'robinhood',
    gmgnFilters: (process.env.GMGN_FILTERS ?? '')
      .split(',')
      .map((f) => f.trim())
      .filter(Boolean),
    // GMGN's 1m figure moves within the minute, so sampling it more often than
    // once a minute keeps a spike from falling between two looks. Its rate
    // limit allows far more than this.
    gmgnPollSeconds: num('GMGN_POLL_SECONDS', 20),
    // Tokens GMGN rates high risk (rug, wash trading, honeypot, sell tax) are
    // logged but not posted unless this is 1.
    sendHighRisk: process.env.SEND_HIGH_RISK === '1',
  };
}
