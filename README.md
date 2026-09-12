# Hash Broker Miner

GPU miner and signer for the [Hash Broker](https://www.hashbroker.fun/#miner)
proof-of-work mint on Robinhood Chain (chain id 4663), built along the same
lines as the Hashcats miner: workers search, one controller signs and
broadcasts.

## The proof

Recovered from the deployed contract and a mainnet mint, and pinned by
`tests/test_mainnet_proof.py`:

```
contract   0x4272D6f51771839F596082eF48fa84D35239Bab3   ("Hash Broker", HBRKR)
proof      sha256( miner[20] || nonce[uint256 big endian] || challenge[bytes32] )   84 bytes
wins when  the digest has at least currentDifficulty() leading zero bits
submit     mine(uint256 nonce, bytes32 challenge)  payable with mintPrice()
```

There is no anchor block and no per-wallet target: the challenge is global and
changes on every mint, so a proof is worthless the moment someone else mints.
The contract exposes `isValidProof(address,uint256,bytes32)`, which the signer
calls before spending anything.

| view | meaning |
| --- | --- |
| `challenge()` | the bytes32 every proof is bound to |
| `currentDifficulty()` | required leading zero bits (50 when this was written) |
| `mintPrice()` | transaction value for a mint (0.0001 ETH) |
| `totalSupply()` / `MAX_SUPPLY()` | minted so far / 4444 |
| `lastMintBlock()` | block of the last accepted proof |

## Layout

```
scripts/protocol.json   contract, chain, ABI signatures, preimage layout
scripts/protocol.py     loads that file, derives selectors, builds mine() calldata
scripts/pow.py          reference proof: preimage -> SHA-256 -> difficulty check
scripts/sha256_cuda.py  the CUDA kernel source
scripts/miner.py        GPU worker (cupy), one process per GPU
scripts/miner_cpu.py    CPU worker, for validating a deployment end to end
scripts/job_feed.py     publishes the on-chain job into job.json
scripts/job_state.py    chain reads, job file handling
scripts/candidate_cache.py  keeps the best proof for the live challenge
scripts/signer.py       verifies, signs, broadcasts, records
scripts/run_all.py      supervises feed + workers + signer on one host
scripts/status.py       read-only terminal view of the local miner
scripts/benchmark.py    measures a GPU's proof rate, no wallet needed
scripts/playground.py   the whole miner against a fake chain, for free
scripts/stub_chain.py   that fake chain
scripts/collect_protocol.py  re-derives the ABI from the chain
scripts/bootstrap.sh start-all.sh stop-all.sh   host setup and process control
tests/                  the whole pipeline, without a GPU or a chain
```

## Try it with no chain, no wallet, no GPU

```bash
python3 -m pip install -r requirements.txt
python3 scripts/playground.py --difficulty 20 --seconds 60
```

A stub node serves the contract's views at a difficulty a CPU solves in seconds
and accepts mints, advancing the challenge like the real contract. The feed,
worker and signer that run against it are the same processes that run on a rig.
`references/how-it-works.md` walks through what each of them does.

## Install

On each GPU host, as root:

```bash
curl -sO https://raw.githubusercontent.com/vhinafk4lee/hsdfiuhds-3/claude/sweet-rubin-w4jyk9/scripts/bootstrap.sh
bash bootstrap.sh
```

It installs the repository under `/opt/hashbroker`, creates a virtualenv, and
picks the CUDA 11 or 12 build of cupy from the driver version.

## Run

```bash
cp scripts/config.example.env config.env     # wallet, key file, cap
set -a && . ./config.env && set +a

python3 scripts/signer.py --check            # state, balance, cap: spends nothing
python3 scripts/run_all.py --wallet "$HASHBROKER_WALLET" --dry-run   # sign, never send
python3 scripts/run_all.py --wallet "$HASHBROKER_WALLET"             # live
python3 scripts/status.py                    # in another shell
```

`run_all.py` starts the feed, one worker per GPU, and the signer, and restarts
whatever dies. `start-all.sh` does the mining half only, for hosts where the
signer runs elsewhere.

Workers write `solution-gpuN.json`; the signer picks them up, re-verifies the
proof against the live challenge and difficulty, prices the mint, refuses
anything above `HASHBROKER_SUBMIT_CAP_WEI`, persists the signed transaction,
then broadcasts. Run the signer on one host only — it owns the account nonce.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

53 tests, no GPU and no network needed. They cover the mainnet proof, the CUDA
device functions (compiled as plain C and compared against `hashlib`), the RPC
reads against an in-process stub chain, every safety refusal in the signer, and
the full feed -> worker -> solution -> signed transaction pipeline.

## Safety

No wallet keys, RPC credentials with API keys in the URL, or host lists belong
in this repository. Keep them in `config.env` and a 0600 key file, both
git-ignored; the signer refuses a key file that is group or world readable.
