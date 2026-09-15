import { fetchPoolPage } from './geckoterminal.js';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Spreads the scan across cycles to stay inside the API rate limit.
 *
 * The evidence a spike leaves — a 5m volume at or above the threshold — lasts
 * five minutes, so every page only has to be revisited faster than that, not
 * every cycle. Hot pages are scanned every time; the rest take turns.
 */
export function createScanner({
  network,
  hotPages,
  rotatingPages,
  maxPages,
  thresholdUsd,
  pageDelayMs = 1500,
}) {
  let cursor = 0;
  let lastKnownPage = maxPages;

  function tailPages() {
    const tail = [];
    for (let page = hotPages + 1; page <= lastKnownPage; page++) tail.push(page);
    return tail;
  }

  function pagesForCycle() {
    const pages = [];
    for (let page = 1; page <= Math.min(hotPages, lastKnownPage); page++) pages.push(page);

    const tail = tailPages();
    if (tail.length > 0) {
      const take = Math.min(rotatingPages, tail.length);
      for (let i = 0; i < take; i++) pages.push(tail[(cursor + i) % tail.length]);
      cursor = (cursor + take) % tail.length;
    }

    return pages;
  }

  return {
    /** Cycles needed to revisit every page once. */
    coverageCycles() {
      const tail = tailPages().length;
      return tail === 0 ? 1 : Math.ceil(tail / rotatingPages);
    },

    async scan() {
      const pools = [];
      const scanned = [];

      for (const page of pagesForCycle()) {
        if (scanned.length > 0) await sleep(pageDelayMs);

        const { pools: pagePools, sorted, lastVolume24h } = await fetchPoolPage(network, page);
        scanned.push(page);

        if (pagePools.length === 0) {
          lastKnownPage = Math.max(hotPages, page - 1);
          break;
        }

        pools.push(...pagePools);

        // A window that trades the threshold sits inside the last 24h, so a
        // pool below it in 24h volume cannot hold one; on a descending page
        // every pool after this one is lower still.
        if (sorted && lastVolume24h < thresholdUsd) break;
      }

      return { pools, pages: scanned };
    },
  };
}
