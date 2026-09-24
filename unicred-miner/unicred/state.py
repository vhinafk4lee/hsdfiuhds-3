"""runtime/: state.json (persistent limits), log.txt, txs.csv, event feed."""
import csv
import json
import os
import threading
import time
from collections import deque

TX_FIELDS = ["time", "hash", "status", "token_id", "nonce", "anchor_block", "challenge",
             "price_wei", "value_wei", "gas_used", "fee_wei", "result"]


class Runtime(object):
    def __init__(self, directory):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.log_path = self.dir / "log.txt"
        self.txs_path = self.dir / "txs.csv"
        self.lock = threading.RLock()
        self.events = deque(maxlen=300)   # (time, level, msg, seq)
        self.event_seq = 0
        self.state = {"spent_wei": 0, "mints_ok": 0, "mints_revert": 0, "txs_sent": 0,
                      "pending": {}}
        if self.state_path.exists():
            try:
                with self.state_path.open("r", encoding="utf-8") as fh:
                    self.state.update(json.load(fh))
            except ValueError:
                raise SystemExit("%s повреждён; исправьте или удалите его" % self.state_path)

    def save(self):
        with self.lock:
            tmp = self.state_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self.state, fh, indent=2, sort_keys=True)
            os.replace(str(tmp), str(self.state_path))

    def event(self, level, msg):
        """level: ok / warn / err / info / log (log = только в файл)."""
        now = time.time()
        with self.lock:
            if level != "log":
                self.event_seq += 1
                self.events.append((now, level, msg, self.event_seq))
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write("%s [%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                                           level, msg))

    def tx_row(self, row):
        with self.lock:
            new = not self.txs_path.exists()
            with self.txs_path.open("a", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=TX_FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                row = dict(row)
                row.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
                w.writerow(row)
