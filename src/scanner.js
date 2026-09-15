import { fetchPoolPage, fetchTrendingPools } from './geckoterminal.js';

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
  useTrending = true,
  cooldownCycles = 5,
  pageDelayMs = 1500,
}) {
  let cursor = 0;
  let lastKnownPage = maxPages;
  let cooldown = 0;

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
      let trending = 0;

      if (useTrending) {
        try {
          const hot = await fetchTrendingPools(network);
          pools.push(...hot);
          trending = hot.length;
        } catch (error) {
          console.error(`trending failed: ${error.message}`);
        }
      }

      // The API throttles harder the more it is pushed, so back off the paged
      // scan for a while after a rejection and keep running on trending alone.
      if (cooldown > 0) {
        cooldown--;
      } else {
        for (const page of pagesForCycle()) {
          if (scanned.length > 0 || trending > 0) await sleep(pageDelayMs);

          let result;
          try {
            result = await fetchPoolPage(network, page);
          } catch (error) {
            console.error(`page ${page} failed: ${error.message}`);
            if (error.message.includes('429')) cooldown = cooldownCycles;
            break;
          }

          scanned.push(page);

          if (result.pools.length === 0) {
            lastKnownPage = Math.max(hotPages, page - 1);
            break;
          }

          pools.push(...result.pools);

          // A window that trades the threshold sits inside the last 24h, so a
          // pool below it in 24h volume cannot hold one; on a descending page
          // every pool after this one is lower still.
          if (result.sorted && result.lastVolume24h < thresholdUsd) break;
        }
      }

      const unique = new Map();
      for (const pool of pools) if (!unique.has(pool.address)) unique.set(pool.address, pool);

      return { pools: [...unique.values()], pages: scanned, trending };
    },
  };
}
