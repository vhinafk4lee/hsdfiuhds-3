# Setup

## Worker host

`scripts/bootstrap.sh` installs everything under `/opt/hashbroker`: the
repository in `repo/`, a virtualenv in `venv/`, and cupy matched to the driver.
It refuses to run without `nvidia-smi`.

Start the feed and one worker per GPU with `scripts/start-all.sh`
(`HASHBROKER_WALLET` must be set; `HASHBROKER_GPUS` overrides the GPU count).
Logs land in `/opt/hashbroker/logs`. `scripts/stop-all.sh` stops the workers and
the feed but never the signer.

A worker prints `SELF_TEST_OK` before mining: it hashes a known nonce on the GPU
and compares against `hashlib`. A GPU that fails it is excluded rather than
trusted.

## Controller

Run the signer on exactly one host — it owns the account nonce. It needs:

- `HASHBROKER_WALLET` and a 0600 key file in `HASHBROKER_PRIVATE_KEY_FILE`
- `HASHBROKER_SUBMIT_CAP_WEI`, the hard ceiling for value + gas per mint
- `--solutions`, a file or a directory of `solution*.json`

For a multi-host fleet, either run a signer per host with a wallet per host, or
sync `solution*.json` into the controller's directory (rsync over SSH) and keep
one signer. Do not point two signers at one wallet.

## systemd example

```ini
[Unit]
Description=Hash Broker signer
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/hashbroker
EnvironmentFile=/etc/hashbroker/config.env
ExecStart=/opt/hashbroker/venv/bin/python /opt/hashbroker/repo/scripts/signer.py --solutions /opt/hashbroker
Restart=on-failure
RestartSec=1
UMask=0077

[Install]
WantedBy=multi-user.target
```

## Re-deriving the ABI

If the contract is redeployed or a view changes, re-run

```bash
python3 scripts/collect_protocol.py --rpc "$RPC" --contract 0x... --wallet 0x... --out report.json
```

and update `scripts/protocol.json`. Nothing else in the miner hardcodes the ABI.
