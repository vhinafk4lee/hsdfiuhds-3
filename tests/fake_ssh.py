#!/usr/bin/env python3
"""Stands in for ssh in the fleet tests.

It ignores every option and the destination, and runs the remote command here
instead, so the controller's real argv, streaming and claim protocol are
exercised without a network or a second machine.
"""
import subprocess
import sys

if __name__ == "__main__":
    raise SystemExit(subprocess.run(["sh", "-c", sys.argv[-1]]).returncode)
