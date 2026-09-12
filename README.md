# Hash Broker Miner

Controller/worker miner for the [Hash Broker](https://www.hashbroker.fun/#miner)
proof-of-work mint on Robinhood Chain, built along the same lines as the
Hashcats miner: GPU workers search for proofs, a single controller signs and
broadcasts the mint transaction.

The one difference that matters: Hash Broker proofs are **SHA-256**, not
Keccak-256, so the CUDA kernel in `scripts/sha256_cuda.py` implements SHA-256.

## Status

The search core is complete and tested. The contract-specific constants are
not: `scripts/protocol.json` still needs the deployed contract address and a
verified view/mint ABI. Fill those in (or set `HASHBROKER_CONTRACT`) before
running against the live chain — every module reads them from that one file.

| Piece | State |
| --- | --- |
| SHA-256 CUDA kernel, verified against `hashlib` | done |
| Preimage layout, padding, nonce placement | done, layout pending confirmation |
| Job feed, candidate cache, shared job file | done |
| Signer, fleet control, dashboard | next |
| Contract address, ABI, mint calldata | **pending** |

## Layout

```
scripts/protocol.json   contract address, chain, ABI signatures, preimage layout
scripts/protocol.py     loads that file, derives 4-byte selectors
scripts/pow.py          reference proof: preimage -> SHA-256 -> target check
scripts/sha256_cuda.py  the CUDA kernel source
scripts/miner.py        GPU worker (cupy)
scripts/miner_cpu.py    CPU worker, for validating a deployment end to end
scripts/job_feed.py     publishes the on-chain job into job.json
scripts/job_state.py    chain reads and shared-job-file handling
scripts/candidate_cache.py  keeps the best proof per anchor until it is usable
tests/                  runs without a GPU
```

## Pinning down the ABI

`scripts/collect_protocol.py` gathers everything needed to fill in
`protocol.json`. It runs on any host that can reach a Robinhood Chain RPC and
needs nothing but the standard library:

```bash
python3 scripts/collect_protocol.py \
    --rpc "$HASHBROKER_RPC" \
    --tx 0xeeb4cf123542544d4d967f6df3afdb32dfa8f89a7dfba489e38e9f68bccfc75a \
    --wallet 0xYourWallet \
    --out hashbroker-report.json
```

It reads the mint transaction and receipt, pulls the deployed bytecode (through
an EIP-1967 proxy when there is one), walks the opcodes to recover the
dispatcher's 4-byte selectors, matches them against a dictionary of plausible
signatures, calls the read-only ones, and writes a single JSON report.

Keep the RPC URL itself in `config.env`: an endpoint with an API key in its path
is a credential and does not belong in this repository.

## Quick start

```bash
python3 -m pip install -r requirements.txt      # controller and worker
python3 -m pip install cupy-cuda12x             # worker only

cp scripts/config.example.env config.env        # fill in wallet and contract
set -a && . ./config.env && set +a

python3 scripts/protocol.py                     # print the resolved protocol
python3 scripts/job_feed.py --wallet "$HASHBROKER_WALLET" --once
python3 scripts/job_feed.py --wallet "$HASHBROKER_WALLET" --output /opt/hashbroker/job.json
python3 scripts/miner.py --wallet "$HASHBROKER_WALLET"
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

`tests/test_kernel_sha256.py` compiles the CUDA device functions as plain C and
compares them against `hashlib`, so the kernel can be verified on a host without
a GPU.

## Safety

No wallet keys, RPC credentials, or host lists belong in this repository. Keep
them in `config.env`, `rentals.json`, and a 0600 key file — all git-ignored.
