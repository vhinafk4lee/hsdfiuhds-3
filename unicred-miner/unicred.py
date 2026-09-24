#!/usr/bin/env python3
"""UNICRED (Unichain) GPU miner — controller.

  python unicred.py check                  онлайн-проверки: сеть, digest, revert, баланс, цена, target
  python unicred.py servers                подключение к серверам, nvidia-smi, selftest, bench
  python unicred.py run --dry-run          майнинг без отправки транзакций
  python unicred.py run                    боевой режим (спросит подтверждение; --yes чтобы не спрашивать)
"""
import argparse
import math
import os
import random
import sys
import threading
import time

from unicred import pow as P
from unicred.chain import Chain
from unicred.config import ETH, load_config
from unicred.dashboard import enable_ansi, fmt_hr, run_dashboard, run_plain
from unicred.servers import (WORKER_FILES, LocalTransport, ensure_python, load_servers,
                             make_transport, run_worker_once)

OK = "\x1b[32mOK\x1b[0m"
FAIL = "\x1b[31mFAIL\x1b[0m"
WARN = "\x1b[33mWARN\x1b[0m"


def load_identity(cfg, need_key):
    """-> (wallet or None, address). The key is loaded only from the local file."""
    from unicred.signer import Wallet
    key_path = cfg.path("private_key_file")
    wallet = None
    if key_path.exists():
        wallet = Wallet(key_path)
    elif need_key:
        raise SystemExit("нет файла ключа %s (private_key_file в config.json)" % key_path)
    address = wallet.address if wallet else (cfg.address or "")
    if wallet and cfg.address and cfg.address.lower() != wallet.address.lower():
        raise SystemExit("address в config.json не совпадает с адресом ключа %s" % wallet.address)
    return wallet, address


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def cmd_check(cfg, args):
    enable_ansi()
    results = []

    def res(ok, name, detail=""):
        results.append(ok)
        print("[%s] %s %s" % (OK if ok else FAIL, name, detail))

    ok_tv = "0x" + P.digest(P.TV["blockhash"], P.TV["challenge"], P.TV["miner"], P.TV["nonce"]).hex() == P.TV["digest"]
    res(ok_tv, "локальный digest = тест-вектор (минт #636)")

    wallet, address = load_identity(cfg, need_key=False)
    if wallet:
        print("[%s] ключ загружен из %s, адрес %s" % (OK, cfg.private_key_file, address))
    elif address:
        print("[%s] ключа нет, адрес из config.json: %s (годится только для --dry-run)" % (WARN, address))
    else:
        print("[%s] нет ни ключа, ни address — проверки кошелька пропущены" % WARN)

    chain = Chain(cfg.rpc_url, address or P.TV["miner"], send_urls=cfg.send_rpc_urls)
    print("RPC: %s" % cfg.rpc_url)
    try:
        cid = chain.chain_id()
    except Exception as exc:
        res(False, "RPC недоступен", str(exc)[:200])
        return 1
    res(cid == P.CHAIN_ID, "chainId = %d" % cid, "(ожидается 130)")
    try:
        chain.poll()
        print("[%s] batch JSON-RPC: %s" % (OK, "поддерживается" if chain.rpc.batch_ok else
                                          "не поддерживается, используются последовательные запросы"))
    except Exception as exc:
        res(False, "опрос состояния", str(exc)[:200])

    # digest: test vector + 3 random inputs vs contract view 0x3a39a703
    try:
        onchain = chain.contract_digest(P.TV["blockhash"], P.TV["challenge"], P.TV["miner"], P.TV["nonce"])
        res(onchain == P.TV["digest"], "контракт digest(тест-вектор) = %s" % onchain[:26] + "…")
        for i in range(3):
            bh, ch = os.urandom(32), os.urandom(32)
            miner, nonce = "0x" + os.urandom(20).hex(), random.getrandbits(256)
            local = "0x" + P.digest(bh, ch, miner, nonce).hex()
            remote = chain.contract_digest(bh, ch, miner, nonce)
            res(local == remote, "случайный вход %d: локальный digest = контракт" % (i + 1), local[:18] + "…")
    except Exception as exc:
        res(False, "вызов 0x3a39a703 (digest)", str(exc)[:200])

    try:
        s = chain.poll()
        minted = s["minted"]
        price = chain.next_price(minted)
        p0, p1 = chain.price(minted), chain.price(minted + 1)
        gt = chain.global_target()
        last = chain.last_mint_block()
        hpm = 2 ** 256 / s["target"] if s["target"] else float("inf")
        print("    блок %d, minted %d/%d, последний минт в блоке %d (%d бл. назад)" % (
            s["head"], minted, P.MAX_SUPPLY, last, s["head"] - last))
        print("    цена: price(%d) = %.6f ETH, price(%d) = %.6f ETH -> платим %.6f ETH (излишек возвращается)" % (
            minted, p0 / ETH, minted + 1, p1 / ETH, price / ETH))
        print("    target(мой) = %s, global = %s, baseFee = %.4f gwei" % (
            P.log2_str(s["target"]), P.log2_str(gt), s["base_fee"] / 1e9))
        print("    ожидается хешей на минт: 2^%.2f = %.3e" % (math.log2(hpm), hpm))
        for hr in (1e9, 10e9, 100e9, 1e12):
            print("      при %-10s ~%.2f минтов/ч" % (fmt_hr(hr), hr * 3600 / hpm))
        res(price <= cfg.wei("max_price_eth"), "цена <= max_price_eth (%s)" % cfg.max_price_eth)
    except Exception as exc:
        res(False, "чтение состояния контракта", str(exc)[:200])
        price, s = None, None

    # mint() with a bad nonce must revert with PoW error 0x7ca55c77
    if s is not None:
        sender = address or P.TV["miner"]
        got = None
        for attempt in range(6):
            head = chain.block_number()
            got = chain.simulate_mint(head - 1, 0, price, price, sender)
            if got and got[0] == "440f8b45":  # a mint in this very block, retry
                time.sleep(1.0)
                continue
            break
        res(bool(got) and got[0] == "7ca55c77", "eth_call mint(плохой nonce) -> revert 0x7ca55c77",
            "получено: %s" % (("0x%s %s" % got) if got else "успех (?!)"))

    if address:
        bal = chain.balance(address)
        need = (price or 0) + cfg.gas_limit * cfg.gwei("max_fee_gwei") + cfg.wei("min_balance_eth")
        print("    баланс %s: %.6f ETH" % (address, bal / ETH))
        if not bal >= need:
            print("[%s] баланса мало для минта: нужно >= %.6f ETH (цена + газ + min_balance_eth)" % (WARN, need / ETH))
        nonce = chain.tx_count(address, "pending")
        print("    nonce кошелька (pending): %d" % nonce)

    try:
        specs = load_servers(cfg.path("servers_file"))
        print("    серверов в %s: %d" % (cfg.servers_file, len(specs)))
    except SystemExit as exc:
        print("[%s] %s" % (WARN, exc))

    ok = all(results)
    print("\nИТОГ: %s" % ("все проверки пройдены" if ok else "есть ошибки"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# servers
# ---------------------------------------------------------------------------
def probe_server(spec, cfg, bench_seconds, out, printer):
    row = {"name": spec.name, "gpus": "", "driver": "", "selftest": "--", "bench": None,
           "per_gpu": "", "note": ""}
    out.append(row)
    tr = make_transport(spec, cfg)
    try:
        printer(spec.name, "подключение…")
        tr.connect()
        tr.upload(WORKER_FILES, cfg.remote_dir)
        if not isinstance(tr, LocalTransport):
            code, text = tr.run("nvidia-smi --query-gpu=index,name,memory.total,driver_version "
                                "--format=csv,noheader", timeout=30)
            lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
            if code != 0:
                row["note"] = "nvidia-smi: " + text.strip()[:80]
            else:
                names = sorted(set(l.split(",")[1].strip() for l in lines))
                row["gpus"] = "%d× %s" % (len(lines), " / ".join(names))
                row["driver"] = lines[0].split(",")[3].strip() if lines else ""
                for l in lines:
                    printer(spec.name, "GPU " + l)
            if not ensure_python(tr):
                row["note"] = "на сервере нет python3 (apt-get не помог)"
                return
        printer(spec.name, "selftest (первый запуск: JIT-компиляция PTX, до минуты)…")
        code, text = run_worker_once(tr, cfg, ["--selftest"], timeout=600)
        for l in text.strip().splitlines():
            printer(spec.name, l)
        row["selftest"] = "OK" if code == 0 and "SELFTEST PASS" in text else "FAIL"
        if row["selftest"] != "OK":
            row["note"] = (text.strip().splitlines() or ["?"])[-1][:80]
            return
        printer(spec.name, "bench %d с…" % bench_seconds)
        code, text = run_worker_once(tr, cfg, ["--bench", str(bench_seconds)], timeout=bench_seconds + 300)
        for l in text.strip().splitlines():
            if l.startswith("BENCH "):
                parts = l.split()
                row["bench"] = float(parts[1])
                row["per_gpu"] = " ".join(fmt_hr(float(x)) for x in parts[2].split(",")) if len(parts) > 2 else ""
            else:
                printer(spec.name, l)
        if row["bench"] is None:
            row["note"] = "bench не выдал результат"
    except Exception as exc:
        row["note"] = str(exc).split("\n")[0][:100]
    finally:
        tr.close()


def cmd_servers(cfg, args):
    enable_ansi()
    specs = load_servers(cfg.path("servers_file"))
    if not specs:
        print("servers.txt пуст")
        return 1
    lock = threading.Lock()

    def printer(name, msg):
        with lock:
            print("[%s] %s" % (name, msg), flush=True)

    rows = []
    threads = [threading.Thread(target=probe_server, args=(s, cfg, args.bench_seconds, rows, printer))
               for s in specs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    target = None
    try:
        _, address = load_identity(cfg, need_key=False)
        chain = Chain(cfg.rpc_url, address or P.TV["miner"])
        target = chain.poll()["target"]
    except Exception:
        pass
    print()
    print("%-26s %-28s %-8s %-8s %12s  %s" % ("СЕРВЕР", "GPU", "ДРАЙВЕР", "SELFTEST", "ХЕШРЕЙТ", "ПРИМЕЧАНИЕ"))
    total = 0.0
    order = {s.name: i for i, s in enumerate(specs)}
    for r in sorted(rows, key=lambda r: order.get(r["name"], 0)):
        total += r["bench"] or 0
        st = OK if r["selftest"] == "OK" else (FAIL if r["selftest"] == "FAIL" else "--")
        print("%-26s %-28s %-8s %-17s %12s  %s" % (r["name"][:26], r["gpus"][:28], r["driver"][:8], st,
                                                   fmt_hr(r["bench"]), r["note"] or r["per_gpu"]))
    print("ИТОГО: %s" % fmt_hr(total), end="")
    if target:
        print(", при текущем target %s ожидается ~%.2f минтов/ч" % (P.log2_str(target), total * 3600 * target / 2 ** 256))
    else:
        print()
    return 0 if rows and all(r["selftest"] == "OK" for r in rows) else 1


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run(cfg, args):
    enable_ansi()
    from unicred.miner import Miner
    wallet, address = load_identity(cfg, need_key=not args.dry_run)
    if not address:
        raise SystemExit("нужен private_key_file (или address для --dry-run)")
    specs = load_servers(cfg.path("servers_file"))
    chain = Chain(cfg.rpc_url, address, send_urls=cfg.send_rpc_urls)
    cid = chain.chain_id()
    if cid != P.CHAIN_ID:
        raise SystemExit("chainId RPC = %d, ожидается 130" % cid)
    s = chain.poll()
    price = chain.next_price(s["minted"])
    balance = chain.balance(address)

    m = Miner(cfg, chain, address, wallet=wallet, specs=specs, dry_run=args.dry_run)
    st = m.state
    print("=" * 72)
    print("UNICRED miner — %s" % ("DRY-RUN: транзакции НЕ отправляются" if args.dry_run else "БОЕВОЙ РЕЖИМ"))
    print("=" * 72)
    print("Конфиг (без секретов):")
    for k, v in sorted(cfg.public_view().items()):
        print("  %-24s %s" % (k, v))
    print("Кошелёк:   %s   (ключ: %s)" % (address, "загружен из файла" if wallet else "нет, только адрес"))
    print("Баланс:    %.6f ETH" % (balance / ETH))
    print("Сеть:      блок %d, minted %d/%d, цена %.6f ETH, target %s" % (
        s["head"], s["minted"], P.MAX_SUPPLY, price / ETH, P.log2_str(s["target"])))
    print("Лимиты:    потратить всего <= %s ETH (уже потрачено %.6f, state.json), цена <= %s ETH," % (
        cfg.max_total_spend_eth, st["spent_wei"] / ETH, cfg.max_price_eth))
    print("           неснижаемый остаток %s ETH, минтов <= %s (уже %d)" % (
        cfg.min_balance_eth, cfg.max_mints, st["mints_ok"]))
    print("Газ:       limit %d, priority %s gwei, maxFee <= %s gwei" % (
        cfg.gas_limit, cfg.priority_fee_gwei, cfg.max_fee_gwei))
    print("Серверы:   %d (%s)" % (len(specs), ", ".join(sp.name for sp in specs[:6]) + ("…" if len(specs) > 6 else "")))
    halt = m.halt_reason()
    if halt:
        print("\x1b[33mВНИМАНИЕ: сейчас майнинг будет на паузе: %s\x1b[0m" % halt)
    if not args.yes:
        try:
            ans = input("Запустить? Введите yes: ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("yes", "y", "да"):
            print("отменено")
            return 1
    m.balance = balance
    m.start()
    try:
        if args.no_dashboard:
            run_plain(m)
        else:
            run_dashboard(m)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nостановка: воркеры получают EOF и завершаются…")
        m.stop()
        st = m.state
        print("минтов ok %d, revert %d, в пути %d, потрачено %.6f ETH. Лог: %s" % (
            st["mints_ok"], st["mints_revert"], len(st["pending"]), st["spent_wei"] / ETH,
            cfg.runtime / "log.txt"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="UNICRED GPU miner controller")
    ap.add_argument("--config", default="config.json", help="путь к config.json")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("check", help="онлайн-проверки")
    sp = sub.add_parser("servers", help="проверка серверов: nvidia-smi, selftest, bench")
    sp.add_argument("--bench-seconds", type=int, default=10)
    rp = sub.add_parser("run", help="майнинг")
    rp.add_argument("--dry-run", action="store_true", help="не отправлять транзакции")
    rp.add_argument("--yes", action="store_true", help="не спрашивать подтверждение")
    rp.add_argument("--no-dashboard", action="store_true", help="построчный лог вместо дашборда")
    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 1
    cfg = load_config(args.config)
    return {"check": cmd_check, "servers": cmd_servers, "run": cmd_run}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
