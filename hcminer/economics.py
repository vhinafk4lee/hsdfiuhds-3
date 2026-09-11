"""Rent-vs-reward math. Run this before you run the miner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Projection:
    """One mined cat in a running projection."""

    index: int
    hours: float
    entry_usd: float
    margin_usd: float
    cumulative_usd: float


@dataclass
class Economics:
    hashrate: float          # hashes per second, whole rig
    zero_bits: int           # difficulty: hash must have this many leading zero bits
    entry_price_eth: float   # msg.value per accepted solution
    rent_usd_hour: float
    eth_usd: float
    epoch_doubling: bool = True   # entry price doubles each epoch
    resale_usd: float = 0.0       # what a cat actually sells for
    epoch_cats: int = 0           # cats minted network-wide per epoch (0 = never doubles)
    network_share: float = 1.0    # fraction of network mints that are yours

    @property
    def expected_hashes(self) -> float:
        return float(1 << self.zero_bits)

    @property
    def seconds_per_cat(self) -> float:
        if self.hashrate <= 0:
            return float("inf")
        return self.expected_hashes / self.hashrate

    @property
    def cats_per_day(self) -> float:
        return 86400.0 / self.seconds_per_cat if self.seconds_per_cat else 0.0

    @property
    def rent_cost_per_cat_usd(self) -> float:
        return self.rent_usd_hour * self.seconds_per_cat / 3600.0

    @property
    def entry_cost_per_cat_usd(self) -> float:
        return self.entry_price_eth * self.eth_usd

    @property
    def total_cost_per_cat_usd(self) -> float:
        return self.rent_cost_per_cat_usd + self.entry_cost_per_cat_usd

    def report(self) -> str:
        h = self.hashrate
        unit, div = ("GH/s", 1e9) if h >= 1e9 else ("MH/s", 1e6)
        secs = self.seconds_per_cat
        if secs < 3600:
            eta = f"{secs / 60:.1f} min"
        elif secs < 86400:
            eta = f"{secs / 3600:.1f} h"
        else:
            eta = f"{secs / 86400:.2f} days"

        lines = [
            f"hashrate            {h / div:.2f} {unit}",
            f"difficulty          {self.zero_bits} zero bits  (~{self.expected_hashes:.3e} hashes per cat)",
            f"expected time/cat   {eta}",
            f"cats per day        {self.cats_per_day:.3f}",
            "",
            f"rent                ${self.rent_usd_hour:.3f}/h  =  ${self.rent_usd_hour * 24:.2f}/day",
            f"  rent per cat      ${self.rent_cost_per_cat_usd:,.2f}",
            f"  entry per cat     ${self.entry_cost_per_cat_usd:,.2f}  "
            f"({self.entry_price_eth} ETH @ ${self.eth_usd:,.0f})",
            f"  TOTAL per cat     ${self.total_cost_per_cat_usd:,.2f}",
            "",
            "A cat is profitable only if its resale value exceeds TOTAL per cat.",
        ]
        if self.epoch_doubling:
            lines.append(
                "Entry price doubles every epoch, so each subsequent cat costs at least "
                "twice the entry above."
            )
        return "\n".join(lines)


    # --- projection over many cats ------------------------------------------

    def project(self, max_cats: int = 500, max_hours: float = 24 * 30) -> List[Projection]:
        """Mine cats one after another until the entry price passes the resale price.

        The entry price doubles every `epoch_cats` mints *network-wide*, so when you
        are only part of the network you pay for other people's mints too: every cat
        you take is accompanied by roughly (1/network_share - 1) cats from everyone
        else, and the price climbs that much faster.
        """
        rows: List[Projection] = []
        entry_eth = self.entry_price_eth
        hours = 0.0
        cumulative = 0.0
        network_mints = 0.0
        share = max(min(self.network_share, 1.0), 1e-9)

        for index in range(1, max_cats + 1):
            entry_usd = entry_eth * self.eth_usd
            margin = self.resale_usd - entry_usd - self.rent_cost_per_cat_usd / share
            if margin <= 0:
                break

            hours += self.seconds_per_cat / 3600.0
            if hours > max_hours:
                break
            cumulative += margin
            rows.append(Projection(index, hours, entry_usd, margin, cumulative))

            # your mint, plus everyone else's during the same stretch
            network_mints += 1.0 / share
            if self.epoch_cats and network_mints >= self.epoch_cats:
                entry_eth *= 2
                network_mints -= self.epoch_cats

        return rows

    def projection_report(self, max_cats: int = 500) -> str:
        if self.resale_usd <= 0:
            return "(pass --resale-usd to project cumulative profit)"

        rows = self.project(max_cats=max_cats)
        if not rows:
            return ("First cat is already unprofitable: entry "
                    f"${self.entry_cost_per_cat_usd:,.0f} vs resale ${self.resale_usd:,.0f}.")

        lines = [f"resale price        ${self.resale_usd:,.0f} per cat",
                 f"your share of network mints  {self.network_share * 100:.0f}%", ""]
        lines.append("  cat    elapsed     entry      margin   cumulative")
        shown = [r for i, r in enumerate(rows) if i < 3 or i >= len(rows) - 3 or r.index % 25 == 0]
        last_index = 0
        for row in shown:
            if row.index != last_index + 1 and last_index:
                lines.append("   ...")
            lines.append(f"  {row.index:>4}  {row.hours:>8.1f}h  ${row.entry_usd:>8,.0f}  "
                         f"${row.margin_usd:>9,.0f}  ${row.cumulative_usd:>10,.0f}")
            last_index = row.index

        total = rows[-1]
        lines += [
            "",
            f"profitable cats     {len(rows)} before the entry price passes resale",
            f"time to mine them   {total.hours:.1f} h ({total.hours / 24:.1f} days)",
            f"total margin        ${total.cumulative_usd:,.0f}",
        ]
        if self.epoch_cats:
            lines.append(
                f"Entry doubles every {self.epoch_cats} network mints, so the profitable "
                "window is finite: faster hardware wins a larger share of it, it does not "
                "make it bigger."
            )
        return "\n".join(lines)
