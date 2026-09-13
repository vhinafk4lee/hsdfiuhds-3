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

## Rented hosts (vast.ai and similar)

The rental is a container reached over SSH on a non-standard port
(`ssh -p <port> root@<ip>`). Install and measure before committing to a long
rental:

```bash
curl -sO https://raw.githubusercontent.com/vhinafk4lee/hsdfiuhds-3/claude/sweet-rubin-w4jyk9/scripts/bootstrap.sh
bash bootstrap.sh
/opt/hashbroker/venv/bin/python /opt/hashbroker/repo/scripts/benchmark.py
```

The benchmark self-tests the kernel, sweeps launch shapes, reports GH/s and the
expected time per proof. Feed its suggestion back with
`fleet.py start --blocks N --threads N --iterations N`, or set
`HASHBROKER_BLOCKS` / `HASHBROKER_THREADS` / `HASHBROKER_ITERATIONS` on the host.
An RTX 5090 measured 10.3 GH/s at `--blocks 16384 --threads 256 --iterations 64`.

An RTX 50xx card is Blackwell (sm_120): it needs a cupy build whose NVRTC can
target it, which `bootstrap.sh` picks from the CUDA version `nvidia-smi`
reports. Override with `HASHBROKER_CUPY=cupy-cuda12x` if the automatic choice is
wrong for an image.

Container storage on these hosts is ephemeral. Keep the key file and the
runtime directory on the controller, never on a rented worker: a worker needs
no key at all.

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
