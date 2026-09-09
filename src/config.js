function int(name, fallback) {
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

  return {
    botToken,
    chatId,
    network: process.env.NETWORK || 'robinhood',
    volumeThresholdUsd: int('VOLUME_THRESHOLD_USD', 300000),
    pollIntervalSeconds: int('POLL_INTERVAL_SECONDS', 60),
    maxPools: int('MAX_POOLS', 25),
    universeRefreshMinutes: int('UNIVERSE_REFRESH_MINUTES', 5),
    alertCooldownMinutes: int('ALERT_COOLDOWN_MINUTES', 10),
  };
}
