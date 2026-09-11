"""hcminer command line: discover -> solve-schema -> verify -> bench -> mine."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chain import Chain, MiningState
from .config import Config
from .economics import Economics
from .gpu import GpuMiner
from .keccak import keccak256, leading_zero_bits
from .rpc import Rpc
from .schema import Schema
from .tx import Limits, SpendGuardError, Submitter


def _rpc(cfg: Config) -> Rpc:
    return Rpc(cfg.require("chain.rpc_url"))


def _chain(cfg: Config, rpc: Rpc) -> Chain:
    return Chain(rpc, cfg.require("contract.address"), cfg)


# --------------------------------------------------------------------- commands

def cmd_state(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    rpc = _rpc(cfg)
    print(f"chain id  {rpc.chain_id()} (config says {cfg.get('chain.chain_id')})")
    state = _chain(cfg, rpc).read_state()
    print(state.describe())
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    """Pull the ABI and recent mints so the PoW scheme can be pinned down."""
    from .discover import abi_signature, describe_abi, fetch_abi, find_mint_txs

    cfg = Config.load(args.config)
    explorer = cfg.require("chain.explorer")
    address = args.address or cfg.require("contract.address")

    print(f"fetching ABI for {address} from {explorer} ...")
    abi_entries = fetch_abi(explorer, address)
    report = describe_abi(abi_entries)
    print(report.render())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "abi.json").write_text(json.dumps(abi_entries, indent=2))
    print(f"\nfull ABI written to {out_dir / 'abi.json'}")

    if not report.mint_candidates:
        print("no obvious mint function found; inspect abi.json by hand")
        return 1

    entry = report.mint_candidates[0]
    if args.mint_signature:
        match = [e for e in abi_entries
                 if e.get("type") == "function" and abi_signature(e) == args.mint_signature]
        if not match:
            print(f"no ABI function matches {args.mint_signature}")
            return 1
        entry = match[0]

    print(f"\nlooking for recent successful calls to {abi_signature(entry)} ...")
    txs = find_mint_txs(explorer, address, entry, limit=args.limit)
    if not txs:
        print("none found — try --mint-signature with another candidate above")
        return 1

    for tx in txs:
        print(f"  {tx['hash']}  from={tx['from']}  args={tx['args']}")
    (out_dir / "mints.json").write_text(json.dumps(txs, indent=2))
    print(f"\nwrote {out_dir / 'mints.json'}")
    print(
        "\nNext: build an observation file (miner, nonce, prev_work, anchor as they were\n"
        "at that block — read them with 'hcminer state' pinned to that block) and run\n"
        "  hcminer solve-schema --observations observations.json"
    )
    return 0


def cmd_solve_schema(args: argparse.Namespace) -> int:
    from .discover import Observation, solve_schema

    raw = json.loads(Path(args.observations).read_text())
    items = raw if isinstance(raw, list) else [raw]
    observations = [Observation(**item) for item in items]

    print(f"testing candidate layouts against {len(observations)} known-good mint(s) "
          f"at >= {args.min_zero_bits} zero bits ...")
    matches = solve_schema(observations, min_zero_bits=args.min_zero_bits)

    if not matches:
        print(
            "no layout reproduces the difficulty.\n"
            "Either a field value is wrong (check prev_work/anchor at the right block),\n"
            "or the contract hashes something extra — read the verified source in\n"
            "discover/abi.json and add the field to hcminer/discover.py:ENCODING_CHOICES."
        )
        return 1

    for match in matches:
        print(f"  MATCH  {match.schema}   zero_bits={match.zero_bits}  hash={match.digest}")
    if len(matches) > 1:
        print("\nmore than one layout fits; add a second observation to disambiguate")
    print("\nPaste into config:\n  [pow]\n  schema = " +
          json.dumps(matches[0].schema.specs()))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Check the configured schema really reproduces a past accepted solution."""
    from .discover import Observation

    cfg = Config.load(args.config)
    schema = Schema.parse(cfg.schema_specs())
    raw = json.loads(Path(args.observations).read_text())
    items = raw if isinstance(raw, list) else [raw]

    ok = True
    for item in items:
        obs = Observation(**item)
        digest = schema.hash(obs.values())
        bits = leading_zero_bits(digest)
        verdict = "OK " if bits >= args.min_zero_bits else "FAIL"
        if bits < args.min_zero_bits:
            ok = False
        print(f"  {verdict} nonce={obs.nonce} zero_bits={bits} hash=0x{digest.hex()}")

    print(f"\nschema: {schema}\npreimage length: "
          f"{len(schema.build(Observation(**items[0]).values()))} bytes, "
          f"GPU varies bytes {schema.vary_offset}..{schema.vary_offset + 7}")
    if not ok:
        print("\nDo NOT mine with this schema: it does not reproduce a real solution.")
        return 1
    print("\nSchema verified against real on-chain solutions.")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    """Read-only sanity pass over config, chain, wallet and GPU before spending."""
    from .preflight import Preflight

    cfg = Config.load(args.config)
    want_send = not bool(cfg.get("limits.dry_run", True))
    print(f"preflight for {cfg.path} "
          f"({'LIVE — transactions will be sent' if want_send else 'dry-run mode'})\n")
    return Preflight(cfg, want_send).run()


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config) if Path(args.config).exists() else None
    binary = args.binary or (cfg.get("miner.gpu_binary") if cfg else "src/cuda/hcminer-gpu")
    devices = args.devices or (cfg.get("miner.devices", "") if cfg else "")

    import subprocess

    cmd = [binary, "--bench", str(args.seconds)]
    if devices:
        cmd += ["--devices", devices]
    cmd += ["--threads", str(args.threads), "--inner", str(args.inner)]
    print("running", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rate = 0.0
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "status":
            print(f"  {event['hashrate'] / 1e9:.2f} GH/s")
        elif event.get("type") == "bench":
            rate = event["hashrate"]
        elif event.get("type") == "error":
            print("  error:", event["message"])
    if proc.stderr.strip():
        print(proc.stderr.strip()[-500:])
    if rate:
        print(f"\nsustained: {rate / 1e9:.2f} GH/s across the rig")
    return 0 if rate else 1


def cmd_econ(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config) if Path(args.config).exists() else None
    zero_bits = args.zero_bits
    entry = args.entry_eth
    if cfg and (zero_bits is None or entry is None) and not args.offline:
        try:
            state = _chain(cfg, _rpc(cfg)).read_state()
            zero_bits = zero_bits if zero_bits is not None else state.zero_bits
            entry = entry if entry is not None else state.price_wei / 1e18
            print(f"(live chain state: {zero_bits} zero bits, entry {entry:.6f} ETH)\n")
        except Exception as exc:
            print(f"(could not read chain state: {exc}; using CLI values)\n")
    zero_bits = zero_bits if zero_bits is not None else 42
    entry = entry if entry is not None else 0.01

    print(Economics(
        hashrate=args.hashrate,
        zero_bits=zero_bits,
        entry_price_eth=entry,
        rent_usd_hour=args.rent,
        eth_usd=args.eth_usd,
    ).report())
    return 0


def cmd_mine(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    schema = Schema.parse(cfg.schema_specs())
    limits = Limits.from_config(cfg)
    if args.dry_run:
        limits.dry_run = True

    if not cfg.get("pow.verified", False) and not args.force:
        print(
            "pow.verified is not set in the config.\n"
            "Run 'hcminer verify --observations ...' against a real past mint first, then set\n"
            "  [pow]\n  verified = true\n"
            "Mining with an unverified schema burns GPU time on hashes the contract will reject.\n"
            "Use --force to override."
        )
        return 1

    rpc = _rpc(cfg)
    chain = _chain(cfg, rpc)
    wallet = cfg.require("miner.wallet_address")

    key_env = cfg.get("miner.private_key_env", "HC_PRIVATE_KEY")
    private_key = os.environ.get(key_env)
    if not private_key and not limits.dry_run:
        print(f"set {key_env} in the environment, or run with --dry-run")
        return 1

    submitter = Submitter(rpc, cfg.require("contract.address"), cfg, private_key, limits)
    if submitter.address and submitter.address.lower() != wallet.lower():
        print(f"key holds {submitter.address} but miner.wallet_address is {wallet}")
        return 1

    gpu = GpuMiner(
        binary=cfg.get("miner.gpu_binary", "src/cuda/hcminer-gpu"),
        devices=str(cfg.get("miner.devices", "")),
        threads=int(cfg.get("miner.threads", 256)),
        blocks=int(cfg.get("miner.blocks", 0)),
        inner=int(cfg.get("miner.inner", 256)),
    )
    gpu.start()

    nonce_field = next(f for f in schema.fields if f.source == "nonce")
    prefix_bits = nonce_field.size * 8 - 64
    poll_seconds = float(cfg.get("miner.poll_seconds", 5))

    job_id = 0
    job_key: Optional[tuple] = None
    state: Optional[MiningState] = None
    session_prefix = 0
    last_state_check = 0.0
    mints = 0

    print(f"mining as {wallet}")
    print(f"schema: {schema}")
    print(f"limits: dry_run={limits.dry_run} max_mints={limits.max_mints} "
          f"max_spend_eth={limits.max_spend_eth}")

    try:
        while True:
            now = time.time()
            if now - last_state_check >= poll_seconds:
                last_state_check = now
                try:
                    fresh = chain.read_state()
                except Exception as exc:
                    print(f"chain read failed: {exc}")
                    time.sleep(poll_seconds)
                    continue

                key = (fresh.target, fresh.prev_work, fresh.anchor)
                if key != job_key:
                    state = fresh
                    job_key = key
                    job_id += 1
                    session_prefix = secrets.randbits(prefix_bits) if prefix_bits > 0 else 0
                    values = _job_values(wallet, session_prefix, state)
                    preimage = schema.build(values)
                    gpu.submit_job(job_id, preimage, schema.vary_offset, state.target)
                    print(f"[job {job_id}] block={state.block} bits={state.zero_bits} "
                          f"prev_work={state.prev_work[:14]}... anchor={state.anchor[:14]}... "
                          f"entry={state.price_wei / 1e18:.6f} ETH")

            for event in gpu.poll(timeout=0.5):
                kind = event.get("type")
                if kind == "status" and event.get("hashrate"):
                    print(f"    {event['hashrate'] / 1e9:.2f} GH/s   total={event['total']:.3g}",
                          end="\r", flush=True)
                elif kind == "error":
                    print(f"\nGPU error: {event['message']}")
                elif kind == "exit":
                    print("\nGPU process exited")
                    print(gpu.stderr_tail())
                    return 1
                elif kind == "solution":
                    if event.get("job") != job_id or state is None:
                        continue
                    full_nonce = (session_prefix << 64) | int(event["nonce"], 16)
                    values = _job_values(wallet, session_prefix, state)
                    values["nonce"] = full_nonce
                    digest = schema.hash(values)

                    if int.from_bytes(digest, "big") >= state.target:
                        print(f"\nGPU reported a solution the CPU cannot confirm "
                              f"(hash=0x{digest.hex()}); ignoring")
                        continue
                    print(f"\n[job {job_id}] solution: nonce={full_nonce} "
                          f"hash=0x{digest.hex()} ({leading_zero_bits(digest)} zero bits)")

                    values["hash"] = "0x" + digest.hex()
                    try:
                        result = submitter.submit(values, state.price_wei)
                    except SpendGuardError as exc:
                        print(f"stopped by spending guard: {exc}")
                        return 0
                    print("submit:", json.dumps(result, indent=2))

                    if result["status"] == "sent":
                        mints += 1
                        if limits.max_mints and mints >= limits.max_mints:
                            print(f"reached limits.max_mints={limits.max_mints}; done")
                            return 0
                    last_state_check = 0.0  # re-read chain, the tip has moved
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 0
    finally:
        gpu.stop()


def _job_values(wallet: str, session_prefix: int, state: MiningState) -> Dict[str, Any]:
    """Field values for the current job; the low 64 bits of the nonce stay zero
    because the GPU ORs its counter into exactly those bytes."""
    return {
        "miner": wallet,
        "nonce": session_prefix << 64,
        "prev_work": state.prev_work,
        "anchor": state.anchor,
    }


# ------------------------------------------------------------------------ entry

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hcminer", description=__doc__)
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("state", help="read target / prev work / anchor / price")
    p.set_defaults(func=cmd_state)

    p = sub.add_parser("discover", help="fetch ABI and recent mints from the explorer")
    p.add_argument("--address")
    p.add_argument("--mint-signature")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--out", default="discover")
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("solve-schema", help="recover the hashing layout from a past mint")
    p.add_argument("--observations", required=True)
    p.add_argument("--min-zero-bits", type=int, default=40)
    p.set_defaults(func=cmd_solve_schema)

    p = sub.add_parser("verify", help="check the configured schema against real solutions")
    p.add_argument("--observations", required=True)
    p.add_argument("--min-zero-bits", type=int, default=40)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("preflight", help="check config, chain, wallet and GPU before mining")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("bench", help="measure rig hashrate")
    p.add_argument("--binary")
    p.add_argument("--devices")
    p.add_argument("--seconds", type=float, default=15)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--inner", type=int, default=256)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("econ", help="profitability of renting GPUs for this difficulty")
    p.add_argument("--hashrate", type=float, required=True, help="hashes/second, whole rig")
    p.add_argument("--zero-bits", type=int)
    p.add_argument("--entry-eth", type=float)
    p.add_argument("--rent", type=float, default=1.928, help="USD per hour")
    p.add_argument("--eth-usd", type=float, default=3000.0)
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_econ)

    p = sub.add_parser("mine", help="run the miner")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="mine even without pow.verified")
    p.set_defaults(func=cmd_mine)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
