"""Drives the CUDA process: push jobs in, read hashrate and solutions out."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class Solution:
    job_id: int
    nonce_low: int
    hash_hex: str
    device: int


def resolve_binary(binary: str) -> str:
    """Accept a POSIX-style path on Windows and add .exe when that is what exists."""
    path = Path(binary)
    if path.exists():
        return str(path)
    if sys.platform == "win32":
        exe = path.with_suffix(".exe")
        if exe.exists():
            return str(exe)
    return str(path)


class GpuMiner:
    def __init__(self, binary: str, devices: str = "", threads: int = 256,
                 blocks: int = 0, inner: int = 256, streams: int = 4,
                 blocks_mult: int = 1, max_kernel_ms: float = 0.0,
                 extra_args: Optional[List[str]] = None):
        self.binary = resolve_binary(binary)
        args: List[str] = [self.binary]
        if devices:
            args += ["--devices", devices]
        args += [
            "--threads", str(threads),
            "--inner", str(inner),
            "--streams", str(streams),
            "--blocks-mult", str(blocks_mult),
        ]
        if blocks:
            args += ["--blocks", str(blocks)]
        if max_kernel_ms:
            args += ["--max-kernel-ms", str(max_kernel_ms)]
        args += list(extra_args or [])
        self.args = args
        self.proc: Optional[subprocess.Popen] = None
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.hashrate = 0.0
        self._reader: Optional[threading.Thread] = None

    def start(self) -> None:
        if not Path(self.binary).exists():
            how = (r"build it first:  scripts\build_windows.bat"
                   if sys.platform == "win32" else
                   "build it first:  make -C src/cuda")
            raise SystemExit(f"GPU binary not found: {self.binary}\n{how}")
        self.proc = subprocess.Popen(
            self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "status":
                self.hashrate = float(event.get("hashrate", 0))
            self.events.put(event)
        self.events.put({"type": "exit"})

    def submit_job(self, job_id: int, preimage: bytes, vary_offset: int,
                   target: int, nonce_start: int = 0) -> None:
        assert self.proc and self.proc.stdin
        msg = {
            "cmd": "job",
            "id": job_id,
            "preimage": preimage.hex(),
            "vary_offset": vary_offset,
            "target": f"{target:064x}",
            "nonce_start": hex(nonce_start),
        }
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def poll(self, timeout: float = 1.0) -> Iterator[dict]:
        try:
            while True:
                yield self.events.get(timeout=timeout)
                timeout = 0.01
        except queue.Empty:
            return

    def stop(self) -> None:
        if not self.proc:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def stderr_tail(self, lines: int = 10) -> str:
        if not self.proc or not self.proc.stderr:
            return ""
        try:
            return "".join(self.proc.stderr.readlines()[-lines:])
        except Exception:
            return ""
