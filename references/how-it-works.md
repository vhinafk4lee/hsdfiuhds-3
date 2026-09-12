# How it works

A walkthrough of the whole miner, in the order the data moves. Everything here
can be watched live with `python3 scripts/playground.py`, which runs these exact
processes against a fake chain.

## 1. What the contract wants

```solidity
mine(uint256 nonce, bytes32 challenge) payable
```

The contract recomputes

```
sha256( msg.sender[20] || nonce[32, big endian] || challenge[32] )
```

and accepts the call when that digest has at least `currentDifficulty()` leading
zero bits, the challenge matches the live one, and the value equals
`mintPrice()`. Then it mints the next token and moves to a new challenge.

Three consequences shape the design:

- **The proof is bound to your address.** Nobody can steal a found nonce and
  mint with it, and a solution found for one wallet is useless to another.
- **It is a race with no partial credit.** The moment anyone mints, the
  challenge changes and every proof in flight for the old one is worthless.
- **There is no anchor window.** Hashcats binds proofs to a block; here the
  only clock is the challenge.

## 2. Preparing the search (`pow.py`)

SHA-256 hashes 64-byte blocks. The 84-byte preimage is padded to 128 bytes: a
`0x80` byte, zeros, and the bit length in the last eight bytes. `padded_words`
returns that as 32 big-endian 32-bit words, built with a **zero nonce**.

The nonce sits at byte 20 and is 32 bytes wide, so its low eight bytes are words
11 and 12 of the message. `nonce_word_indices` computes those indices from the
layout in `protocol.json` rather than hardcoding them: change the layout, and
both the CPU reference and the kernel follow.

Searching therefore means: take the fixed message, write two words, hash.

## 3. Searching on the GPU (`sha256_cuda.py`, `miner.py`)

Each thread gets a slice of the 32-bit counter space and runs `iterations`
hashes. The message and the target live in shared memory; only the two nonce
words differ per hash. A thread that finds a digest below the target does an
`atomicCAS` on a found flag so exactly one winner reports, and the rest of the
batch returns early.

The 64-bit search space is `stream << 32 | counter`: the counter walks a batch,
and the stream is re-randomized whenever the challenge changes, so restarting a
worker never repeats work another worker is doing.

Two safeguards sit around the kernel:

- **Self-test at startup.** The worker hashes a known nonce on the GPU and
  compares against `hashlib`. A card that fails is refused, not trusted.
- **CPU re-verification.** Every candidate the GPU reports is recomputed on the
  CPU before it is cached. A silent GPU error can waste a mint fee otherwise.

`tests/test_kernel_sha256.py` compiles the device functions as ordinary C and
compares them against `hashlib`, so the kernel is verified on machines with no
GPU at all.

## 4. Feeding the job (`job_feed.py`, `job_state.py`)

The feed reads challenge, difficulty, price, supply and last mint block in one
batched call **at a single block number**, so the pieces cannot disagree, and
writes `job.json` atomically (write to a temp file, rename). Workers watch the
file's mtime.

`accept_job` throws away a snapshot that moves backwards, and `read_shared_job`
refuses a file older than five seconds: mining against a dead challenge is worse
than not mining, since it burns the card and produces nothing.

## 5. Keeping the best proof (`candidate_cache.py`)

Within one challenge the cache keeps the best digest seen. If the difficulty
eases while the challenge stands, a proof that just missed becomes a winner with
no new work. When the challenge changes, the cache empties — there is nothing to
carry over.

Workers also mine against a slightly wider target (`search_target`, three bits of
slack) so a proof survives a small difficulty rise.

## 6. Spending (`signer.py`)

The signer is the only process with a key, and the only one that can lose money.
Every broadcast passes the same gate, in this order:

1. recompute the proof from the solution file — a file is data, not a promise;
2. confirm the challenge is still live and the digest still beats the target;
3. ask the contract's own `isValidProof(address,uint256,bytes32)`;
4. estimate gas, refuse an estimate above the gas cap;
5. refuse if value + gas exceeds `HASHBROKER_SUBMIT_CAP_WEI` or the balance;
6. write the signed transaction to the runtime directory **before** sending, so
   a crash cannot lose track of an in-flight account nonce;
7. broadcast to every endpoint at once and take the first acceptance.

Run exactly one signer per wallet: two signers share an account nonce and will
knock each other out.

## 7. Running it (`run_all.py`, `status.py`)

`run_all.py` starts the feed, one worker per GPU, and the signer as separate
processes, streams their output with a prefix, and restarts whatever dies with a
backoff. Separate processes mean a CUDA fault in one worker cannot take down the
signer holding the key.

`status.py` renders the job file, pending solutions, and the signer's event log.
It reads files only — safe to run anywhere.

## 8. Learning by breaking it

```bash
python3 scripts/playground.py --difficulty 20 --seconds 60
```

A stub node (`stub_chain.py`) serves the contract's views and accepts mints,
advancing the challenge exactly as the real one does. Things worth trying:

- raise `--difficulty` and watch proofs get rarer;
- edit the preimage layout in `protocol.json` and watch the golden test in
  `tests/test_mainnet_proof.py` fail — that test is what pins the miner to the
  real contract;
- break a rule in `signer.py` and watch `tests/test_signer.py` catch it;
- run `scripts/benchmark.py` on a GPU host to see what a card actually does.
