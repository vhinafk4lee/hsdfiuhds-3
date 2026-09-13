#!/usr/bin/env python3
"""Runs the whole miner on one host: job feed, workers, and the signer.

Each piece stays a separate process, exactly as it runs in production, so a
crash in one never takes the others down. This supervisor starts them, streams
their output with a prefix, and restarts whatever dies with a growing backoff.

    python3 scripts/run_all.py --wallet 0x... --mode gpu            # sign and send
    python3 scripts/run_all.py --wallet 0x... --mode cpu --dry-run  # never sends
    python3 scripts/run_all.py --wallet 0x... --no-signer           # mine only
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
BACKOFF_START = 1.0
BACKOFF_CAP = 30.0
INSTANT_FAILURE_SECONDS = 5.0
INSTANT_FAILURES_BEFORE_GIVING_UP = 3


def count_gpus() -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        listing = subprocess.run(["nvidia-smi", "--list-gpus"],
                                 capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return 0
    return len([line for line in listing.stdout.splitlines() if line.strip()])


class Service:
    """One supervised child process, restarted with a growing backoff."""

    def __init__(self, name: str, command: list[str], environment: dict[str, str]):
        self.name = name
        self.command = command
        self.environment = environment
        self.process: subprocess.Popen | None = None
        self.backoff = BACKOFF_START
        self.restart_at = 0.0
        self.started_at = 0.0
        self.restarts = -1
        self.instant_failures = 0
        self.fatal = False

    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command, env=self.environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.started_at = time.monotonic()
        self.restarts += 1
        threading.Thread(target=self._stream, args=(self.process,), daemon=True).start()

    def _stream(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{self.name}] {line.rstrip()}", flush=True)
        process.stdout.close()

    def supervise(self, now: float) -> None:
        if self.fatal:
            return
        if self.process is None:
            if now >= self.restart_at:
                self.start()
            return
        code = self.process.poll()
        if code is None:
            if now - self.started_at > 60:
                self.backoff = BACKOFF_START  # it has been healthy for a while
            return
        self.process = None
        lifetime = now - self.started_at
        if code != 0 and lifetime < INSTANT_FAILURE_SECONDS:
            self.instant_failures += 1
        else:
            self.instant_failures = 0
        if self.instant_failures >= INSTANT_FAILURES_BEFORE_GIVING_UP:
            # It never got far enough to do any work, so restarting will not help:
            # this is a configuration error waiting for a person, not a crash.
            self.fatal = True
            print(f"[{self.name}] failed immediately {self.instant_failures} times; "
                  f"giving up — fix the error above and start again", flush=True)
            return
        delay = BACKOFF_START if code == 0 else self.backoff
        print(f"[{self.name}] exited with {code}, restarting in {delay:.0f}s", flush=True)
        self.restart_at = now + delay
        self.backoff = BACKOFF_START if code == 0 else min(self.backoff * 2, BACKOFF_CAP)

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None


def build_services(arguments, environment: dict[str, str]) -> list[Service]:
    python = arguments.python
    root = Path(arguments.dir)
    root.mkdir(parents=True, exist_ok=True)
    job_file = root / "job.json"

    services = [Service("feed", [
        python, str(SCRIPTS / "job_feed.py"),
        "--wallet", arguments.wallet, "--output", str(job_file),
        "--interval", str(arguments.interval),
    ], environment)]

    if arguments.mode == "gpu":
        gpus = arguments.gpus if arguments.gpus is not None else count_gpus()
        if gpus < 1:
            raise SystemExit("no GPUs found: pass --gpus, or use --mode cpu")
        for index in range(gpus):
            services.append(Service(f"gpu{index}", [
                python, str(SCRIPTS / "miner.py"),
                "--wallet", arguments.wallet, "--device", str(index),
                "--job-file", str(job_file),
                "--output", str(root / f"solution-gpu{index}.json"),
                "--keep-mining",
            ], environment))
    else:
        services.append(Service("cpu", [
            python, str(SCRIPTS / "miner_cpu.py"),
            "--wallet", arguments.wallet, "--job-file", str(job_file),
            "--output", str(root / "solution-cpu.json"), "--keep-mining",
        ], environment))

    if arguments.signer:
        command = [python, str(SCRIPTS / "signer.py"), "--solutions", str(root)]
        if arguments.dry_run:
            command.append("--dry-run")
        services.append(Service("signer", command, environment))
    return services


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet", default=os.environ.get("HASHBROKER_WALLET", ""))
    parser.add_argument("--dir", default="/opt/hashbroker")
    parser.add_argument("--mode", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="sign but never broadcast")
    parser.add_argument("--no-signer", dest="signer", action="store_false",
                        help="mine only; run the signer elsewhere")
    parser.add_argument("--run-for", type=float, default=0.0, help="stop after N seconds")
    arguments = parser.parse_args()

    wallet = arguments.wallet.strip()
    if not wallet.startswith("0x") or len(wallet) != 42:
        raise SystemExit("pass --wallet 0x... or set HASHBROKER_WALLET")
    arguments.wallet = wallet

    environment = {**os.environ, "PYTHONPATH": str(SCRIPTS), "PYTHONUNBUFFERED": "1"}
    services = build_services(arguments, environment)
    print("RUN " + " ".join(service.name for service in services), flush=True)

    stopping = threading.Event()

    def request_stop(*_signal_args):
        stopping.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    deadline = time.monotonic() + arguments.run_for if arguments.run_for else None
    try:
        while not stopping.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            for service in services:
                service.supervise(now)
            if all(service.fatal for service in services):
                print("every service gave up; nothing left to supervise", flush=True)
                break
            time.sleep(0.25)
    finally:
        print("STOPPING", flush=True)
        for service in services:
            service.stop()


if __name__ == "__main__":
    main()
