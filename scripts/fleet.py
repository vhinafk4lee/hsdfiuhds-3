#!/usr/bin/env python3
"""Controller for a fleet of rented GPU boxes.

Your machine holds the key and the wallet; the rented boxes only hash. The
controller installs the worker on each of them over SSH, streams the solutions
they find back over the same connection, and hands them to the local signer.

    python3 scripts/fleet.py keygen                 # make the SSH key, print the public half
    python3 scripts/fleet.py check                  # can we reach every box, how many GPUs
    python3 scripts/fleet.py deploy                 # install the worker everywhere
    python3 scripts/fleet.py start                  # start mining everywhere
    python3 scripts/fleet.py run                    # collect solutions + sign, locally
    python3 scripts/fleet.py status                 # what each box is doing
    python3 scripts/fleet.py stop                   # stop mining everywhere

Hosts live in rentals.json; copy scripts/rentals.example.json and fill it in.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from run_all import Service  # noqa: E402

DEFAULT_REMOTE_DIR = "/opt/hashbroker"
DEFAULT_BRANCH = os.environ.get("HASHBROKER_BRANCH", "claude/sweet-rubin-w4jyk9")
RAW_BASE = (f"https://raw.githubusercontent.com/vhinafk4lee/hsdfiuhds-3/{DEFAULT_BRANCH}/scripts")
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
)


@dataclass(frozen=True)
class Rental:
    name: str
    host: str
    port: int
    user: str
    gpus: int | None

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def load_rentals(path: Path) -> list[Rental]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"{path} not found: copy scripts/rentals.example.json and fill it in")
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"{path} must be a non-empty list of hosts")
    rentals = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise SystemExit(f"host {index} must be an object")
        name = str(item.get("name") or f"host{index}").strip()
        host = str(item.get("host") or "").strip()
        port = item.get("port", 22)
        gpus = item.get("gpus")
        if not host:
            raise SystemExit(f"host {index} needs a host address")
        if not isinstance(port, int) or not 0 < port <= 65535:
            raise SystemExit(f"host {name} needs a numeric port")
        if gpus is not None and (not isinstance(gpus, int) or gpus < 1):
            raise SystemExit(f"host {name} has an invalid gpu count")
        rentals.append(Rental(name, host, port, str(item.get("user") or "root").strip(), gpus))
    return rentals


def ssh_argv(rental: Rental, command: str, ssh: str, key: str | None) -> list[str]:
    argv = [ssh, "-p", str(rental.port), *SSH_OPTIONS]
    if key:
        argv += ["-i", key]
    argv += [rental.target, command]
    return argv


def run_ssh(rental: Rental, command: str, ssh: str, key: str | None,
            timeout: float = 900.0) -> subprocess.CompletedProcess:
    return subprocess.run(ssh_argv(rental, command, ssh, key),
                          capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------- subcommands

def keygen(arguments) -> None:
    path = Path(arguments.key).expanduser()
    public = path.with_suffix(path.suffix + ".pub")
    if path.exists():
        print(f"key already exists: {path}")
    else:
        if not shutil.which("ssh-keygen"):
            raise SystemExit(
                "ssh-keygen not found. Install OpenSSH (it ships with Windows 10+, macOS and "
                "every Linux) and run:\n"
                f"  ssh-keygen -t ed25519 -N \"\" -C hashbroker -f {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "hashbroker",
                        "-f", str(path)], check=True, capture_output=True, text=True)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        print(f"created {path}")
    print("\nPaste this public key into the rental provider (vast.ai: the key icon on the "
          "instance, \"Add/remove SSH keys\"):\n")
    print(public.read_text(encoding="utf-8").strip())
    print(f"\nThe private half stays here: {path}. Never copy it to a rented box.")


def check(arguments, rentals: list[Rental]) -> None:
    command = "echo HASHBROKER_OK; nvidia-smi -L 2>/dev/null | wc -l"
    for rental in rentals:
        try:
            result = run_ssh(rental, command, arguments.ssh, arguments.key, timeout=40)
        except subprocess.SubprocessError as exc:
            print(f"{rental.name:<12} unreachable ({type(exc).__name__})")
            continue
        if "HASHBROKER_OK" not in result.stdout:
            detail = (result.stderr or result.stdout).strip().splitlines()
            print(f"{rental.name:<12} unreachable: {detail[-1] if detail else 'no output'}")
            continue
        gpus = 0
        for line in result.stdout.splitlines():
            if line.strip().isdigit():
                gpus = int(line.strip())
        print(f"{rental.name:<12} ok  {rental.target}:{rental.port}  gpus={gpus}")


def deploy(arguments, rentals: list[Rental]) -> None:
    command = (
        f"set -e; mkdir -p {arguments.dir}; cd {arguments.dir}; "
        f"curl -sfSO {RAW_BASE}/bootstrap.sh && bash bootstrap.sh"
    )
    for rental in rentals:
        print(f"== {rental.name}: installing ==", flush=True)
        result = run_ssh(rental, command, arguments.ssh, arguments.key, timeout=1800)
        print((result.stdout or "").strip()[-1500:])
        if result.returncode != 0:
            print(f"{rental.name}: install failed\n{(result.stderr or '').strip()[-800:]}")


def start(arguments, rentals: list[Rental]) -> None:
    wallet = arguments.wallet
    for rental in rentals:
        gpu_option = f"export HASHBROKER_GPUS={rental.gpus}; " if rental.gpus else ""
        command = (
            f"cd {arguments.dir}; export HASHBROKER_WALLET={wallet}; {gpu_option}"
            f"curl -sfSO {RAW_BASE}/autopilot.sh && bash autopilot.sh --no-signer --skip-benchmark"
        )
        result = run_ssh(rental, command, arguments.ssh, arguments.key, timeout=1800)
        tail = (result.stdout or "").strip().splitlines()[-4:]
        print(f"== {rental.name} ==")
        print("\n".join(tail) if tail else (result.stderr or "").strip()[-400:])


def stop(arguments, rentals: list[Rental]) -> None:
    command = "pkill -f run_all.py; pkill -f miner.py; pkill -f job_feed.py; echo stopped"
    for rental in rentals:
        result = run_ssh(rental, command, arguments.ssh, arguments.key, timeout=60)
        print(f"{rental.name:<12} {(result.stdout or result.stderr).strip().splitlines()[-1:]}")


def status(arguments, rentals: list[Rental]) -> None:
    command = (
        f"cd {arguments.dir} 2>/dev/null || exit 1; "
        "pgrep -f miner.py >/dev/null && echo 'workers: running' || echo 'workers: DOWN'; "
        "nvidia-smi --query-gpu=index,utilization.gpu,temperature.gpu --format=csv,noheader "
        "2>/dev/null | tr '\\n' ' '; echo; "
        "grep -h RATE logs/miner.log 2>/dev/null | tail -n 2"
    )
    for rental in rentals:
        result = run_ssh(rental, command, arguments.ssh, arguments.key, timeout=60)
        print(f"== {rental.name} ({rental.target}) ==")
        print((result.stdout or result.stderr).strip() or "no output")


# ------------------------------------------------------------------ collector

def stream_command(remote_dir: str) -> str:
    """Claim, print and delete solution files on the worker, one JSON per line."""
    return (
        f"cd {remote_dir} 2>/dev/null || exit 1; "
        "while :; do "
        "for f in solution-*.json; do "
        "[ -e \"$f\" ] || continue; "
        "c=\"$f.claim\"; "
        "mv -- \"$f\" \"$c\" 2>/dev/null || continue; "
        "tr -d '\\r\\n' < \"$c\"; printf '\\n'; "
        "rm -f -- \"$c\"; "
        "done; sleep 0.2; done"
    )


class Collector(threading.Thread):
    """One long-lived SSH connection per host, streaming its solutions back."""

    def __init__(self, rental: Rental, arguments, sink: queue.Queue, stopping: threading.Event):
        super().__init__(daemon=True, name=f"collect-{rental.name}")
        self.rental = rental
        self.arguments = arguments
        self.sink = sink
        self.stopping = stopping
        self.process: subprocess.Popen | None = None

    def run(self) -> None:
        backoff = 1.0
        command = stream_command(self.arguments.dir)
        while not self.stopping.is_set():
            started = time.monotonic()
            try:
                self.process = subprocess.Popen(
                    ssh_argv(self.rental, command, self.arguments.ssh, self.arguments.key),
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                )
            except OSError as exc:
                print(f"[{self.rental.name}] cannot start ssh: {exc}", flush=True)
                return
            assert self.process.stdout is not None
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    solution = json.loads(line)
                except ValueError:
                    continue
                if isinstance(solution, dict):
                    self.sink.put((self.rental.name, solution))
            self.process.stdout.close()
            self.process = None
            if self.stopping.is_set():
                return
            if time.monotonic() - started > 60:
                backoff = 1.0
            print(f"[{self.rental.name}] connection dropped, retrying in {backoff:.0f}s", flush=True)
            self.stopping.wait(backoff)
            backoff = min(backoff * 2, 30.0)

    def stop(self) -> None:
        process = self.process
        if process is not None:
            process.terminate()


def write_solution(directory: Path, name: str, index: int, solution: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"solution-{name}-{index:06d}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(solution, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def collect(arguments, rentals: list[Rental], signer: Service | None = None) -> int:
    sink: queue.Queue = queue.Queue()
    stopping = threading.Event()
    collectors = [Collector(rental, arguments, sink, stopping) for rental in rentals]
    for collector in collectors:
        collector.start()
    print("COLLECTING " + ", ".join(rental.name for rental in rentals), flush=True)

    solutions = Path(arguments.solutions)
    deadline = time.monotonic() + arguments.run_for if arguments.run_for else None
    received = 0
    try:
        while True:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            if signer is not None:
                signer.supervise(now)
            try:
                name, solution = sink.get(timeout=0.25)
            except queue.Empty:
                continue
            received += 1
            path = write_solution(solutions, name, received, solution)
            print(f"SOLUTION from {name}: {solution.get('hash')} -> {path.name}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        for collector in collectors:
            collector.stop()
        if signer is not None:
            signer.stop()
    return received


def run(arguments, rentals: list[Rental]) -> None:
    """Collect from every box and sign locally: the controller's normal mode."""
    environment = {**os.environ, "PYTHONPATH": str(SCRIPTS), "PYTHONUNBUFFERED": "1"}
    command = [arguments.python, str(SCRIPTS / "signer.py"), "--solutions", arguments.solutions]
    if arguments.dry_run:
        command.append("--dry-run")
    signer = Service("signer", command, environment)
    collect(arguments, rentals, signer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=("keygen", "check", "deploy", "start", "stop", "status",
                                 "collect", "run"))
    parser.add_argument("--rentals", default=os.environ.get("HASHBROKER_RENTALS_FILE",
                                                            "rentals.json"))
    parser.add_argument("--key", default=os.environ.get("HASHBROKER_SSH_KEY",
                                                        str(Path.home() / ".hashbroker" / "id_ed25519")))
    parser.add_argument("--ssh", default=os.environ.get("HASHBROKER_SSH", "ssh"))
    parser.add_argument("--dir", default=DEFAULT_REMOTE_DIR, help="worker directory on the boxes")
    parser.add_argument("--solutions", default="./solutions",
                        help="local directory the signer watches")
    parser.add_argument("--wallet", default=os.environ.get("HASHBROKER_WALLET", ""))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="sign but never broadcast")
    parser.add_argument("--run-for", type=float, default=0.0, help="stop after N seconds")
    arguments = parser.parse_args()

    if arguments.command == "keygen":
        keygen(arguments)
        return

    rentals = load_rentals(Path(arguments.rentals))
    if arguments.command in ("start",) and not arguments.wallet.startswith("0x"):
        raise SystemExit("pass --wallet 0x... or set HASHBROKER_WALLET")

    handlers = {"check": check, "deploy": deploy, "start": start, "stop": stop,
                "status": status, "collect": collect, "run": run}
    handlers[arguments.command](arguments, rentals)


if __name__ == "__main__":
    main()
