#!/usr/bin/env python3
"""Creating and checking the private key file, on every platform.

POSIX permissions are enforced where they exist. Windows has no 0600, so the
check is skipped there rather than refusing to run: the file still lands in the
user's profile, which is what Windows access control protects.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

WINDOWS = os.name == "nt"


def write_private(path: Path, content: str) -> None:
    """Create a new key file readable only by this user. Never overwrite one."""
    if path.exists():
        raise SystemExit(f"{path} already exists; refusing to overwrite a key")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
    assert_private(path)


def assert_private(path: Path) -> None:
    if WINDOWS:
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise SystemExit(f"{path} is group or world readable (mode {mode:o}); "
                         f"run: chmod 600 {path}")


def describe(path: Path) -> str:
    if WINDOWS:
        return f"{path}  (keep it in your user profile; never copy it to a rented box)"
    return f"{path}  (mode 600, never print or copy it)"
