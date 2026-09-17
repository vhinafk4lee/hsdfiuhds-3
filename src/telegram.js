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

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
}

export function formatAlert({ pool, candle, windowMinutes, network }) {
  const symbol = escapeHtml(pool.baseSymbol ?? pool.name ?? pool.address);
  const price = pool.priceUsd ? `$${pool.priceUsd.toPrecision(4)}` : 'n/a';
  const chartUrl = `https://www.geckoterminal.com/${network}/pools/${pool.address}`;

  return [
    `🚨 <b>${symbol}</b> — ${usd.format(candle.volumeUsd)} in ${windowMinutes} min`,
    '',
    `Pair: ${escapeHtml(pool.name ?? '')}`,
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
