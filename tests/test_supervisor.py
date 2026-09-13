#!/usr/bin/env python3
"""A service that cannot start must not be restarted forever."""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_all import INSTANT_FAILURES_BEFORE_GIVING_UP, Service  # noqa: E402


def service(command: list[str]) -> Service:
    return Service("probe", [sys.executable, "-c", *command], {})


class SupervisorTests(unittest.TestCase):
    def drain(self, item: Service, seconds: float = 20.0) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not item.fatal:
            item.supervise(time.monotonic())
            time.sleep(0.05)

    def test_gives_up_on_a_process_that_fails_instantly(self):
        item = service(["raise SystemExit(1)"])
        self.addCleanup(item.stop)
        self.drain(item)
        self.assertTrue(item.fatal, "a config error must not be retried forever")
        self.assertEqual(item.instant_failures, INSTANT_FAILURES_BEFORE_GIVING_UP)

    def test_keeps_a_healthy_process_running(self):
        item = service(["import time; time.sleep(30)"])
        self.addCleanup(item.stop)
        for _ in range(10):
            item.supervise(time.monotonic())
            time.sleep(0.05)
        self.assertFalse(item.fatal)
        self.assertEqual(item.restarts, 0)

    def test_a_clean_exit_restarts_without_backing_off(self):
        item = service(["pass"])
        self.addCleanup(item.stop)
        started = time.monotonic()
        while time.monotonic() - started < 4 and item.restarts < 2:
            item.supervise(time.monotonic())
            time.sleep(0.05)
        self.assertFalse(item.fatal, "finishing normally is not a failure")
        self.assertGreaterEqual(item.restarts, 1)


if __name__ == "__main__":
    unittest.main()
