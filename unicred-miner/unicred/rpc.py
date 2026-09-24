"""Minimal JSON-RPC client: batch with automatic fallback to sequential calls."""
import itertools
import threading

import requests


class RpcError(Exception):
    def __init__(self, message, code=None, data=None):
        Exception.__init__(self, message)
        self.code = code
        self.data = data

    @property
    def revert_data(self):
        """0x... revert payload if the error carries one."""
        d = self.data
        if isinstance(d, dict):
            d = d.get("data") or d.get("result")
        if isinstance(d, str) and d.startswith("0x") and len(d) >= 10:
            return d
        msg = str(self)
        idx = msg.find("0x")
        if "revert" in msg.lower() and idx >= 0:
            tail = msg[idx:].split()[0].strip(",.;:'\"")
            if len(tail) >= 10:
                return tail
        return None


class Rpc(object):
    def __init__(self, url, timeout=4.0):
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.batch_ok = True
        self.last_latency = None

    def _post(self, payload, timeout=None):
        r = self.session.post(self.url, json=payload, timeout=timeout or self.timeout)
        self.last_latency = r.elapsed.total_seconds()
        if r.status_code == 429:
            raise RpcError("HTTP 429 rate limited", code=429)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _result(item):
        if "error" in item and item["error"] is not None:
            err = item["error"]
            if isinstance(err, dict):
                return RpcError(err.get("message", "rpc error"), err.get("code"), err.get("data"))
            return RpcError(str(err))
        return item.get("result")

    def call(self, method, params=None, timeout=None):
        with self._lock:
            rid = next(self._ids)
        resp = self._post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or []}, timeout)
        if isinstance(resp, list):
            resp = resp[0]
        res = self._result(resp)
        if isinstance(res, RpcError):
            raise res
        return res

    def batch(self, calls, timeout=None):
        """calls: [(method, params)] -> list of results; failed entries are RpcError objects."""
        if not calls:
            return []
        if self.batch_ok:
            with self._lock:
                ids = [next(self._ids) for _ in calls]
            payload = [{"jsonrpc": "2.0", "id": i, "method": m, "params": p}
                       for i, (m, p) in zip(ids, calls)]
            try:
                resp = self._post(payload, timeout)
            except RpcError:
                raise
            except (requests.RequestException, ValueError):
                raise
            if isinstance(resp, list) and len(resp) == len(calls):
                by_id = {item.get("id"): item for item in resp if isinstance(item, dict)}
                if all(i in by_id for i in ids):
                    return [self._result(by_id[i]) for i in ids]
            # RPC does not support batch requests: switch to sequential calls
            self.batch_ok = False
        out = []
        for method, params in calls:
            try:
                out.append(self.call(method, params, timeout))
            except RpcError as exc:
                out.append(exc)
        return out
