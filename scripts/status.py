#!/usr/bin/env python3
"""Read-only view of what the miner is doing on this host.

Reads the job file the feed publishes, the solution files workers leave, and
the signer's event log. It never talks to the chain and never holds a key, so
it is safe to leave running anywhere.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CLEAR = "\033[2J\033[H"


def read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def read_events(path: Path, limit: int) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def age(value) -> str:
    if not isinstance(value, (int, float)):
        return "?"
    seconds = max(0.0, time.time() - value)
    return f"{seconds:.1f}s ago"


def render(root: Path, runtime: Path, events_limit: int) -> str:
    lines = [f"hashbroker status  {time.strftime('%H:%M:%S')}", ""]
    job = read_json(root / "job.json")
    if job is None:
        lines.append("job        no job file yet (is the feed running?)")
    else:
        minted = job.get("minted")
        supply = job.get("maxSupply", "?")
        lines += [
            f"challenge  {job.get('challenge')}",
            f"difficulty {job.get('difficulty')}  target 2^{256 - int(job.get('difficulty', 0))}",
            f"minted     {minted}/{supply}",
            f"price      {int(job.get('priceWei', 0)) / 1e18:.6f} ETH",
            f"block      {job.get('blockNumber')}  (last mint {job.get('lastMintBlock', '?')})",
            f"fetched    {age(job.get('fetchedAt'))}",
        ]

    solutions = sorted(root.glob("solution*.json"))
    lines += ["", f"solutions  {len(solutions)} waiting"]
    for path in solutions[:5]:
        solution = read_json(path) or {}
        lines.append(f"  {path.name}  {str(solution.get('hash'))[:20]}...  "
                     f"challenge {str(solution.get('challenge'))[:12]}...  "
                     f"{age(solution.get('foundAt'))}")

    events = read_events(runtime / "events.jsonl", events_limit)
    lines += ["", f"events     {len(events)} shown"]
    for row in events:
        stamp = time.strftime("%H:%M:%S", time.localtime(row.get("at", 0)))
        detail = {key: value for key, value in row.items() if key not in ("at", "event")}
        lines.append(f"  {stamp}  {row.get('event'):<18} "
                     f"{json.dumps(detail, separators=(',', ':'), default=str)[:110]}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/opt/hashbroker")
    parser.add_argument("--runtime", default=None, help="signer runtime dir (default: --dir)")
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()

    root = Path(arguments.dir)
    runtime = Path(arguments.runtime) if arguments.runtime else root
    while True:
        frame = render(root, runtime, arguments.events)
        if arguments.once:
            print(frame)
            return
        print(CLEAR + frame, flush=True)
        time.sleep(arguments.interval)


if __name__ == "__main__":
    main()
