# Architecture

Workers and the controller have separate trust boundaries: a worker never holds
a key and never talks to the chain about money. It only reads a job file and
writes unsigned candidates.

## Job flow

`job_feed.py` reads the contract's anchor, mint price, previous work, per-wallet
target, minted count, and anchor window in a single batched RPC call, then
writes `job.json` atomically. The GPU worker watches that file's mtime and
reloads on change. A stale job file (older than five seconds) stops mining
rather than wasting hashes on dead work.

## Search

The preimage layout lives in `protocol.json`. `pow.py` builds the padded SHA-256
message with a zero nonce and reports which two 32-bit words carry the searched
nonce tail; the kernel rewrites exactly those words, so nothing in CUDA depends
on where the nonce sits. A found nonce is re-hashed on the CPU before it is
cached — a GPU that disagrees with `hashlib` is a hardware fault, not a
candidate.

Workers mine against a slightly wider target than the chain's (`search_target`)
and keep the best proof per `(prevWork, anchor)` pair, so a proof that just
misses stays usable if the target eases before the anchor expires.

## Submission invariants

- `prevWork` changes as other miners mint; stale candidates are worthless.
- `anchorBlock` is valid for a finite window; candidates near expiry are dropped.
- The account nonce and gas estimate are read from more than one RPC.
- A signed transaction is persisted before broadcast so a crash cannot lose
  track of an in-flight nonce.
