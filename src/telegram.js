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
    `🚨 <b>${symbol}</b> — ${usd.format(candle.volumeUsd)} за ${windowMinutes} мин`,
    '',
    `Пара: ${escapeHtml(pool.name ?? '')}`,
    `Цена: ${price}`,
    `Ликвидность: ${usd.format(pool.liquidityUsd)}`,
    `Объём 24ч: ${usd.format(pool.volume24h)}`,
    pool.baseAddress ? `Контракт: <code>${escapeHtml(pool.baseAddress)}</code>` : null,
    '',
    `<a href="${chartUrl}">График</a>`,
  ]
    .filter((line) => line !== null)
    .join('\n');
}
