"""Controller configuration (config.json)."""
import json
import os
from pathlib import Path

DEFAULTS = {
    "rpc_url": "https://mainnet.unichain.org",
    "send_rpc_urls": [],
    "private_key_file": "wallet.key",
    "address": "",
    "servers_file": "servers.txt",
    "ssh_key": "",
    "ssh_password": "",
    "remote_dir": "unicred",
    "priority_fee_gwei": 0.05,
    "max_fee_gwei": 2.0,
    "gas_limit": 350000,
    "max_total_spend_eth": 0.05,
    "max_price_eth": 0.006,
    "min_balance_eth": 0.001,
    "max_mints": 10,
    "simulate_before_send": False,
    "poll_interval": 0.25,
    "anchor_refresh_blocks": 150,
    "anchor_max_age_blocks": 240,
    "candidate_cache_shift": 2,
    "runtime_dir": "runtime",
    "worker_idle_timeout": 30,
}

ETH = 10 ** 18
GWEI = 10 ** 9


class Config(dict):
    """dict with attribute access and derived wei values."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    @property
    def base_dir(self):
        return Path(self["_base_dir"])

    def path(self, key):
        p = Path(os.path.expanduser(str(self[key])))
        return p if p.is_absolute() else self.base_dir / p

    @property
    def runtime(self):
        p = self.path("runtime_dir")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def wei(self, key):
        return int(round(float(self[key]) * ETH))

    def gwei(self, key):
        return int(round(float(self[key]) * GWEI))

    def public_view(self):
        """Config for display: no secrets."""
        hidden = {"ssh_password"}
        return {k: ("***" if k in hidden and v else v)
                for k, v in self.items() if not k.startswith("_")}


def load_config(path="config.json", overrides=None):
    path = Path(path)
    data = dict(DEFAULTS)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            user = json.load(fh)
        unknown = [k for k in user if k not in DEFAULTS and not k.startswith("_")]
        if unknown:
            raise SystemExit("config.json: неизвестные параметры: %s" % ", ".join(unknown))
        data.update({k: v for k, v in user.items() if not k.startswith("_")})
    elif str(path) != "config.json":
        raise SystemExit("нет файла конфига %s" % path)
    if overrides:
        data.update(overrides)
    data["_base_dir"] = str(path.resolve().parent)
    cfg = Config(data)
    if not isinstance(cfg.send_rpc_urls, list):
        raise SystemExit("config.json: send_rpc_urls должен быть списком")
    for key in ("max_total_spend_eth", "max_price_eth", "min_balance_eth", "priority_fee_gwei",
                "max_fee_gwei", "poll_interval"):
        if float(cfg[key]) < 0:
            raise SystemExit("config.json: %s < 0" % key)
    return cfg
