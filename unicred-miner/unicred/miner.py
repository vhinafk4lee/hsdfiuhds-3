"""Controller core: network polling, jobs, candidates, signing, limits, receipts."""
import queue
import threading
import time
from collections import OrderedDict, deque

from . import pow as P
from .chain import parse_mint_log
from .config import ETH
from .servers import WorkerConn
from .state import Runtime

NONCE_ERRORS = ("nonce too low", "nonce too high", "already known", "replacement transaction",
                "invalid nonce", "known transaction")


class Job(object):
    def __init__(self, job_id, challenge, anchor_block, anchor_hash, target, search_target, sender):
        self.id = job_id
        self.challenge = challenge
        self.anchor_block = anchor_block
        self.anchor_hash = anchor_hash
        self.target = target
        self.search_target = search_target
        self.sender = P.addr20(sender)
        self.prefix = P.job_prefix(anchor_hash, challenge)
        self.created = time.time()


class Candidate(object):
    def __init__(self, job, nonce, digest_int, server):
        self.job = job
        self.nonce = nonce
        self.digest = digest_int
        self.server = server
        self.at = time.time()


class Miner(object):
    def __init__(self, cfg, chain, address, wallet=None, specs=(), dry_run=False):
        self.cfg = cfg
        self.chain = chain
        self.address = address
        self.wallet = wallet
        self.dry_run = dry_run
        self.rt = Runtime(cfg.runtime)
        self.lock = threading.RLock()
        self.running = False
        self.started = time.time()

        self.net = None                  # last poll result
        self.net_at = 0.0
        self.rpc_errors = 0
        self.rpc_last_error = ""
        self.price = None
        self.price_minted = None
        self.global_target = None
        self.last_mint_block = None
        self.balance = None
        self.retired = deque(maxlen=64)  # challenges already replaced

        self.job = None
        self.jobs = OrderedDict()
        self.job_seq = 0
        self.halt = None                 # reason mining is paused

        self.sent = OrderedDict()        # challenge -> tx hash / "dry-run"
        self.cache = {}                  # challenge -> best near-candidate
        self.cand_q = queue.Queue()
        self.tx_nonce = None
        self.send_lock = threading.Lock()  # tx nonce: signing/sending vs resync
        self.stats = {"found": 0, "stale": 0, "cached": 0, "bad": 0, "dry": 0, "sent": 0}
        self.feed = deque(maxlen=64)     # network mints
        self.logs_from = None
        self.seen_logs = set()

        self.conns = [WorkerConn(s, cfg, self.on_found, self.rt.event, self.on_ready) for s in specs]

    # ------------------------------------------------------------------ utils
    def event(self, level, msg):
        self.rt.event(level, msg)

    @property
    def state(self):
        return self.rt.state

    def reserved_wei(self):
        return sum(int(p.get("max_cost", 0)) for p in self.state["pending"].values())

    def fees(self):
        base = int(self.net["base_fee"]) if self.net else 0
        cap = self.cfg.gwei("max_fee_gwei")
        prio = min(self.cfg.gwei("priority_fee_gwei"), cap)
        max_fee = min(cap, 2 * base + prio)
        return max(max_fee, prio), prio

    def mint_cost_max(self):
        max_fee, _ = self.fees()
        return (self.price or 0) + self.cfg.gas_limit * max_fee

    # ---------------------------------------------------------------- metrics
    def my_hashrate(self):
        return sum(c.hashrate() for c in self.conns)

    def mint_interval(self):
        blocks = sorted({m["block"] for m in self.feed})[-21:]
        if len(blocks) < 2:
            return None
        return (blocks[-1] - blocks[0]) / float(len(blocks) - 1)  # 1 block = 1 s on Unichain

    def hashes_per_mint(self):
        if not self.net or not self.net["target"]:
            return None
        return float(2 ** 256) / self.net["target"]

    def net_hashrate(self):
        hpm, iv = self.hashes_per_mint(), self.mint_interval()
        if not hpm or not iv:
            return None
        return hpm / iv

    def expected_per_hour(self):
        hpm = self.hashes_per_mint()
        return self.my_hashrate() * 3600.0 / hpm if hpm else None

    def share(self):
        net = self.net_hashrate()
        mine = self.my_hashrate()
        if not net:
            return None
        return min(1.0, mine / max(net, mine)) if mine else 0.0

    # --------------------------------------------------------------- lifecycle
    def start(self):
        self.running = True
        if self.state["pending"]:
            self.event("info", "незавершённые транзакции из прошлого запуска: %d" % len(self.state["pending"]))
        if not self.dry_run:
            self.tx_nonce = self.chain.tx_count(self.address, "pending")
        for target in (self._poll_loop, self._slow_loop, self._submit_loop, self._receipt_loop,
                       self._servers_watch_loop):
            threading.Thread(target=target, daemon=True, name=target.__name__).start()
        for c in self.conns:
            c.start()
        self.event("info", "старт: %d серверов, %s" % (
            len(self.conns), "DRY-RUN (транзакции не отправляются)" if self.dry_run else "боевой режим"))

    # ------------------------------------------------------- servers.txt hot reload
    def _servers_watch_loop(self):
        path = self.cfg.path("servers_file")
        try:
            last = path.stat().st_mtime
        except OSError:
            last = None
        while self.running:
            time.sleep(3.0)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime != last:
                last = mtime
                try:
                    self.reload_servers()
                except BaseException as exc:  # load_servers raises SystemExit on a bad line
                    self.event("warn", "servers.txt не прочитан: %s" % exc)

    def reload_servers(self):
        """Start connections for new lines in servers.txt, stop removed ones. Mining continues."""
        from .servers import load_servers
        specs = load_servers(self.cfg.path("servers_file"))
        wanted = {s.name: s for s in specs}
        with self.lock:
            current = {c.spec.name: c for c in self.conns}
            added = [s for n, s in wanted.items() if n not in current]
            removed = [c for n, c in current.items() if n not in wanted]
            for c in removed:
                c.stop()
            new = [WorkerConn(s, self.cfg, self.on_found, self.rt.event, self.on_ready) for s in added]
            self.conns = [c for c in self.conns if c not in removed] + new
        for c in new:
            c.start()
        if added or removed:
            self.event("info", "servers.txt: добавлено %d, удалено %d (майнинг не прерывался)" % (
                len(added), len(removed)))
        return added, removed

    def stop(self):
        self.running = False
        for c in self.conns:
            c.stop()
        self.rt.save()
        self.event("info", "остановлено")

    # ------------------------------------------------------------------ polling
    def _poll_loop(self):
        interval = float(self.cfg.poll_interval)
        while self.running:
            t0 = time.time()
            try:
                self.poll_once()
            except Exception as exc:
                self.rpc_errors += 1
                self.rpc_last_error = str(exc)[:200]
                if self.rpc_errors % 20 == 1:
                    self.event("warn", "RPC: %s" % self.rpc_last_error)
                time.sleep(min(2.0, interval * 4))
            time.sleep(max(0.0, interval - (time.time() - t0)))

    def poll_once(self):
        s = self.chain.poll()
        with self.lock:
            prev = self.net
            if prev and s["head"] < prev["head"]:
                return  # lagging RPC node
            if s["challenge"] in self.retired:
                return  # lagging RPC node returned an old challenge
            if prev and prev["challenge"] != s["challenge"]:
                self.retired.append(prev["challenge"])
            self.net, self.net_at = s, time.time()
        if self.price is None or s["minted"] != self.price_minted:
            self.price = self.chain.next_price(s["minted"])
            self.price_minted = s["minted"]
        self._update_job(s)

    def halt_reason(self):
        s = self.net
        if s and s["minted"] >= P.MAX_SUPPLY:
            return "sold out (%d/%d)" % (s["minted"], P.MAX_SUPPLY)
        if self.dry_run:
            return None
        st = self.state
        pending = len(st["pending"])
        if self.cfg.max_mints and st["mints_ok"] + pending >= int(self.cfg.max_mints):
            return "достигнут max_mints = %s" % self.cfg.max_mints
        if self.price is not None and self.price > self.cfg.wei("max_price_eth"):
            return "цена %.4f ETH > max_price_eth" % (self.price / ETH)
        cost = self.mint_cost_max()
        if st["spent_wei"] + self.reserved_wei() + cost > self.cfg.wei("max_total_spend_eth"):
            return "исчерпан max_total_spend_eth (потрачено %.5f ETH)" % (st["spent_wei"] / ETH)
        if self.balance is not None and self.balance - self.reserved_wei() - cost < self.cfg.wei("min_balance_eth"):
            return "баланс ниже min_balance_eth + цена минта"
        return None

    def _update_job(self, s):
        with self.lock:
            halt = self.halt_reason()
            if halt:
                if halt != self.halt:
                    self.event("warn", "майнинг на паузе: %s" % halt)
                    self.halt = halt
                    self.job = None
                    self._broadcast(None)
                return
            if self.halt:
                self.event("ok", "майнинг возобновлён")
                self.halt = None
            job = self.job
            refresh = int(self.cfg.anchor_refresh_blocks)
            need_anchor = job is None or s["head"] - job.anchor_block > refresh
            changed = job is None or job.challenge != s["challenge"]
            grown = job is not None and not changed and s["target"] > job.target
            if not (changed or grown or need_anchor):
                self._check_cache(s)
                return
            if need_anchor:
                anchor_block, anchor_hash = s["head"] - 1, s["parent_hash"]
            else:
                anchor_block, anchor_hash = job.anchor_block, job.anchor_hash
            self.job_seq += 1
            search = min(P.MAX_UINT256, s["target"] << int(self.cfg.candidate_cache_shift))
            new = Job("%x" % self.job_seq, s["challenge"], anchor_block, anchor_hash, s["target"],
                      search, self.address)
            self.job = new
            self.jobs[new.id] = new
            while len(self.jobs) > 256:
                self.jobs.popitem(last=False)
            why = "challenge" if changed else ("target" if grown else "anchor")
            self.event("log", "job %s (%s): challenge %s.. anchor %d target %s" % (
                new.id, why, s["challenge"][:18], anchor_block, P.log2_str(s["target"])))
            self._broadcast(new)
            self._check_cache(s)

    def _broadcast(self, job):
        for c in self.conns:
            if c.ready:
                c.send_job(job)

    def on_ready(self, conn):
        with self.lock:
            conn.send_job(self.job)

    def _check_cache(self, s):
        c = self.cache.get(s["challenge"])
        if c and c.digest < s["target"] and s["challenge"] not in self.sent:
            self.event("info", "кэшированный кандидат стал валидным (target вырос)")
            del self.cache[s["challenge"]]
            self.cand_q.put(c)

    # --------------------------------------------------------------- candidates
    def on_found(self, conn, job_id, nonce, digest_hex):
        self.stats["found"] += 1
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            self.event("log", "FOUND для неизвестного задания %s от %s" % (job_id, conn.spec.name))
            return
        self.cand_q.put((job, nonce, digest_hex, conn.spec.name))

    def _submit_loop(self):
        while self.running:
            try:
                item = self.cand_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if isinstance(item, Candidate):
                    self.submit(item)
                else:
                    self.handle_candidate(*item)
            except Exception as exc:
                self.event("err", "ошибка обработки кандидата: %s" % exc)

    def handle_candidate(self, job, nonce, digest_hex, server):
        s = self.net
        # 1. stale?
        if s is None or job.challenge != s["challenge"]:
            self.stats["stale"] += 1
            self.event("log", "кандидат от %s устарел (challenge сменился)" % server)
            return
        # 2. re-check digest locally
        d = P.digest(job.anchor_hash, job.challenge, self.address, nonce)
        dv = int.from_bytes(d, "big")
        if d.hex() != digest_hex.lower().replace("0x", "") or dv >= job.search_target:
            self.stats["bad"] += 1
            self.event("err", "неверный кандидат от %s (digest не совпал)" % server)
            return
        cand = Candidate(job, nonce, dv, server)
        target = max(job.target, s["target"])
        if dv >= target:
            with self.lock:
                best = self.cache.get(job.challenge)
                if best is None or dv < best.digest:
                    self.cache[job.challenge] = cand
                    for ch in list(self.cache):
                        if ch != s["challenge"]:
                            del self.cache[ch]
            self.stats["cached"] += 1
            self.event("log", "почти-кандидат от %s: digest %s > target, в кэше" % (server, P.log2_str(dv)))
            return
        self.submit(cand)

    def submit(self, cand):
        job = cand.job
        s = self.net
        with self.lock:
            if job.challenge != s["challenge"]:
                self.stats["stale"] += 1
                return
            if job.challenge in self.sent:
                return  # 4. one send per challenge
            if s["head"] - job.anchor_block > int(self.cfg.anchor_max_age_blocks):
                self.event("warn", "кандидат отброшен: anchor старше %s блоков" % self.cfg.anchor_max_age_blocks)
                return
            halt = self.halt_reason()
            if halt:
                self.event("warn", "кандидат найден, но отправка заблокирована: %s" % halt)
                return
            self.sent[job.challenge] = "pending"
            while len(self.sent) > 512:
                self.sent.popitem(last=False)
        price = self.price
        max_fee, prio = self.fees()
        max_cost = price + self.cfg.gas_limit * max_fee
        found_msg = "НАЙДЕН nonce (%s, digest %s < target %s)" % (
            cand.server, P.log2_str(cand.digest), P.log2_str(max(job.target, s["target"])))
        if self.cfg.simulate_before_send:
            res = self.chain.simulate_mint(job.anchor_block, cand.nonce, price, price, self.address)
            if res is not None:
                self.event("warn", "%s, но симуляция: %s" % (found_msg, res[1]))
                return
        if self.dry_run:
            self.stats["dry"] += 1
            self.sent[job.challenge] = "dry-run"
            self.event("ok", "DRY-RUN %s: tx НЕ отправлена" % found_msg)
            self.rt.tx_row({"hash": "", "status": "dry-run", "nonce": "0x%064x" % cand.nonce,
                            "anchor_block": job.anchor_block, "challenge": job.challenge,
                            "price_wei": price, "value_wei": price, "result": "dry-run"})
            return
        with self.send_lock:
            self._sign_and_send(cand, price, max_fee, prio, max_cost, found_msg)

    def _sign_and_send(self, cand, price, max_fee, prio, max_cost, found_msg, retry=True):
        job = cand.job
        raw, tx_hash, tx = self.wallet.sign_mint(self.tx_nonce, job.anchor_block, cand.nonce, price,
                                                 int(self.cfg.gas_limit), max_fee, prio)
        t0 = time.time()
        try:
            self.chain.send_raw(raw)
        except Exception as exc:
            msg = str(exc)
            if retry and any(e in msg.lower() for e in NONCE_ERRORS):
                self.event("warn", "nonce рассинхронизирован (%s), пересинхронизация" % msg[:80])
                self.tx_nonce = self.chain.tx_count(self.address, "pending")
                return self._sign_and_send(cand, price, max_fee, prio, max_cost, found_msg, retry=False)
            self.sent[job.challenge] = "error"
            self.event("err", "%s, но отправка не удалась: %s" % (found_msg, msg[:160]))
            self.rt.tx_row({"hash": tx_hash, "status": "send-error", "nonce": "0x%064x" % cand.nonce,
                            "anchor_block": job.anchor_block, "challenge": job.challenge,
                            "value_wei": price, "result": msg[:200]})
            try:
                self.tx_nonce = self.chain.tx_count(self.address, "pending")
            except Exception:
                pass
            return
        self.sent[job.challenge] = tx_hash
        self.stats["sent"] += 1
        with self.rt.lock:
            self.state["pending"][tx_hash] = {
                "sent_at": time.time(), "tx_nonce": self.tx_nonce, "max_cost": max_cost,
                "value": price, "nonce": "0x%064x" % cand.nonce, "anchor_block": job.anchor_block,
                "challenge": job.challenge}
            self.state["txs_sent"] += 1
            self.rt.save()
        self.tx_nonce += 1
        self.event("ok", "%s -> tx %s (%.0f мс)" % (found_msg, tx_hash[:18], (time.time() - t0) * 1000))

    # ----------------------------------------------------------------- receipts
    def _receipt_loop(self):
        while self.running:
            time.sleep(1.0)
            for tx_hash in list(self.state["pending"]):
                try:
                    self.check_receipt(tx_hash)
                except Exception as exc:
                    self.event("log", "receipt %s: %s" % (tx_hash[:18], exc))

    def check_receipt(self, tx_hash):
        info = self.state["pending"][tx_hash]
        r = self.chain.receipt(tx_hash)
        if r is None:
            if time.time() - info["sent_at"] > 90:
                self.event("warn", "tx %s не попала в блок за 90 с, пересинхронизация nonce" % tx_hash[:18])
                with self.rt.lock:
                    del self.state["pending"][tx_hash]
                    self.rt.save()
                self.rt.tx_row({"hash": tx_hash, "status": "dropped", "nonce": info["nonce"],
                                "anchor_block": info["anchor_block"], "challenge": info["challenge"],
                                "value_wei": info["value"], "result": "no receipt"})
                with self.send_lock:
                    self.tx_nonce = self.chain.tx_count(self.address, "pending")
            return
        gas_used = int(r.get("gasUsed", "0x0"), 16)
        eff = int(r.get("effectiveGasPrice", "0x0"), 16)
        fee = gas_used * eff + int(r.get("l1Fee", "0x0") or "0x0", 16)
        ok = int(r.get("status", "0x0"), 16) == 1
        token_id, price_paid, result = "", 0, ""
        if ok:
            price_paid = info["value"]  # upper bound if the Mint log is missing
            for lg in r.get("logs", []):
                if lg.get("topics") and lg["topics"][0].lower() == P.MINT_TOPIC:
                    m = parse_mint_log(dict(lg, blockNumber=r["blockNumber"]))
                    token_id, price_paid = m["token_id"], m["price"]
            result = "mint #%s" % token_id
        else:
            result = "revert"
            try:
                others = self.chain.mint_logs(0, block_hash=r["blockHash"])
                if others:
                    result = "revert: гонка проиграна (минт #%d в том же блоке)" % others[0]["token_id"]
                else:
                    result = "revert (вероятно, challenge сменился до включения)"
            except Exception:
                pass
        with self.rt.lock:
            st = self.state
            st["spent_wei"] += fee + price_paid
            st["mints_ok" if ok else "mints_revert"] += 1
            del st["pending"][tx_hash]
            self.rt.save()
        self.rt.tx_row({"hash": tx_hash, "status": "ok" if ok else "revert", "token_id": token_id,
                        "nonce": info["nonce"], "anchor_block": info["anchor_block"],
                        "challenge": info["challenge"], "price_wei": price_paid,
                        "value_wei": info["value"], "gas_used": gas_used, "fee_wei": fee,
                        "result": result})
        self.event("ok" if ok else "err", "tx %s: %s, газ %.2e ETH" % (tx_hash[:18], result, fee / ETH))

    # ----------------------------------------------------------------- slow loop
    def _slow_loop(self):
        while self.running:
            try:
                self.slow_once()
            except Exception as exc:
                self.event("log", "slow poll: %s" % exc)
            time.sleep(2.0)

    def slow_once(self):
        sp = self.chain.slow_poll()
        self.global_target = sp.get("global_target", self.global_target)
        self.last_mint_block = sp.get("last_mint_block", self.last_mint_block)
        self.balance = sp.get("balance", self.balance)
        if not self.net:
            return
        head = self.net["head"]
        start = self.logs_from if self.logs_from is not None else max(0, head - 900)
        if start > head:
            return
        start = max(start, head - 2000)
        logs = self.chain.mint_logs(start, head)
        self.logs_from = head + 1
        for m in logs:
            key = (m["token_id"], m["tx"])
            if key in self.seen_logs:
                continue
            self.seen_logs.add(key)
            self.feed.append(m)
