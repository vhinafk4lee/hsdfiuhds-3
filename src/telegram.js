export async function sendMessage(botToken, chatId, text) {
  const res = await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chat_id: chatId,
      text,
      parse_mode: 'HTML',
      disable_web_page_preview: true,
    }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Telegram sendMessage -> HTTP ${res.status}: ${body}`);
  }
}

const usd = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
});

/** How long the pool has existed — a token minutes old reads very differently. */
export function formatAge(createdAt, now = Date.now()) {
  if (!createdAt) return null;

  const minutes = Math.floor((now - createdAt) / 60_000);
  if (minutes < 0) return null;
  if (minutes < 60) return `${minutes} min`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m`;

  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
}

const compactUsd = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  notation: 'compact',
  maximumFractionDigits: 2,
});

const percent = (ratio) => `${(Number(ratio) * 100).toFixed(1)}%`;

/**
 * GMGN's own risk scores, reduced to one verdict. Red is for what makes a token
 * untradeable or its volume fake; yellow is for concentration that can be
 * dumped on buyers.
 */
export function assessRisk(token) {
  const red = [];
  const yellow = [];
  if (Number(token.rug_ratio) > 0.3) red.push('rug risk');
  if (token.is_wash_trading) red.push('wash trading');
  if (Number(token.is_honeypot) === 1) red.push('honeypot');
  if (Number(token.sell_tax) > 0.1) red.push('sell tax');
  if (Number(token.bundler_rate) > 0.3) yellow.push('bundlers');
  if (Number(token.top_10_holder_rate) > 0.5) yellow.push('top 10 holders');
  if (Number(token.dev_team_hold_rate) > 0.1) yellow.push('dev holds');
  if (Number(token.rat_trader_amount_rate) > 0.3) yellow.push('insiders');

  if (red.length) return { icon: '🔴', label: 'High risk', reasons: red };
  if (yellow.length) return { icon: '🟡', label: 'Caution', reasons: yellow };
  return { icon: '🟢', label: 'Low risk', reasons: [] };
}

/** Alert built from a GMGN ranking item: the volume plus GMGN's risk scores. */
export function formatGmgnAlert({ token, windowMinutes, chain, now = Date.now() }) {
  const symbol = escapeHtml(token.symbol ?? token.address);
  const age = formatAge(token.creation_timestamp ? token.creation_timestamp * 1000 : null, now);
  const change = Number(token.price_change_percent1m);
  const venue = token.launchpad_platform || token.launchpad;
  const flag = (bad) => (bad ? '⚠️' : '✅');

  return [
    `🚨 <b>${symbol}</b> — ${usd.format(token.volume)} in ${windowMinutes} min`,
    '',
    [age ? `Age: ${age}` : null, venue ? escapeHtml(venue) : null].filter(Boolean).join(' · '),
    `Price: $${Number(token.price).toPrecision(4)}${Number.isFinite(change) ? ` (${change >= 0 ? '+' : ''}${change.toFixed(2)}% 1m)` : ''}`,
    `MCap: ${compactUsd.format(token.market_cap ?? 0)} · Liq: ${compactUsd.format(token.liquidity ?? 0)}`,
    `Buys/Sells: ${token.buys ?? '?'} / ${token.sells ?? '?'} · Holders: ${Number(token.holder_count ?? 0).toLocaleString('en-US')}`,
    '',
    `${flag(Number(token.rug_ratio) > 0.3)} Rug risk: ${percent(token.rug_ratio ?? 0)}`,
    `${flag(token.is_wash_trading)} Wash trading: ${token.is_wash_trading ? 'yes' : 'no'}`,
    `${flag(Number(token.bundler_rate) > 0.3)} Bundlers: ${percent(token.bundler_rate ?? 0)}`,
    `${flag(Number(token.top_10_holder_rate) > 0.5)} Top 10 holders: ${percent(token.top_10_holder_rate ?? 0)}`,
    `${flag(Number(token.dev_team_hold_rate) > 0.1)} Dev holds: ${percent(token.dev_team_hold_rate ?? 0)}`,
    `${flag(Number(token.is_honeypot) === 1 || Number(token.sell_tax) > 0.1)} Honeypot: ${Number(token.is_honeypot) === 1 ? 'yes' : 'no'} · Tax: ${percent(token.buy_tax || 0)} / ${percent(token.sell_tax || 0)}`,
    '',
    `<code>${escapeHtml(token.address)}</code>`,
    '',
    `<a href="https://gmgn.ai/${chain}/token/${token.address}">GMGN</a>` +
      (token.twitter_username ? ` · <a href="${escapeHtml(token.twitter_username)}">X</a>` : ''),
  ].join('\n');
}

export function formatAlert({ pool, candle, windowMinutes, network }) {
  const symbol = escapeHtml(pool.baseSymbol ?? pool.name ?? pool.address);
  const price = pool.priceUsd ? `$${pool.priceUsd.toPrecision(4)}` : 'n/a';
  const chartUrl = `https://www.geckoterminal.com/${network}/pools/${pool.address}`;
  const age = formatAge(pool.createdAt);

  return [
    `🚨 <b>${symbol}</b> — ${usd.format(candle.volumeUsd)} in ${windowMinutes} min`,
    '',
    `Pair: ${escapeHtml(pool.name ?? '')}`,
    age ? `Age: ${age}` : null,
    pool.dex ? `Dex: ${escapeHtml(pool.dex)}` : null,
    `Price: ${price}`,
    `Liquidity: ${usd.format(pool.liquidityUsd)}`,
    `24h volume: ${usd.format(pool.volume24h)}`,
    // On its own line the address is a clean tap target: tapping a <code> span
    // copies exactly its contents, so nothing else comes along with it.
    pool.baseAddress ? `\n<code>${escapeHtml(pool.baseAddress)}</code>` : null,
    '',
    `<a href="${chartUrl}">Chart</a>`,
  ]
    .filter((line) => line !== null)
    .join('\n');
}
