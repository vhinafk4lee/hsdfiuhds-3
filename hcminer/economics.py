"""Rent-vs-reward math. Run this before you run the miner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Economics:
    hashrate: float          # hashes per second, whole rig
    zero_bits: int           # difficulty: hash must have this many leading zero bits
    entry_price_eth: float   # msg.value per accepted solution
    rent_usd_hour: float
    eth_usd: float
    epoch_doubling: bool = True   # entry price doubles each epoch

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
