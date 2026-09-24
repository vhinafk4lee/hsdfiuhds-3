"""Console dashboard (ANSI, refreshed once per second; works in Windows Terminal / cmd / PowerShell)."""
import math
import os
import sys
import time

from . import pow as P
from .config import ETH

GREEN, YELLOW, RED, CYAN, DIM, BOLD, RESET = ("\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[36m",
                                              "\x1b[2m", "\x1b[1m", "\x1b[0m")
LEVEL = {"ok": (GREEN, "+"), "warn": (YELLOW, "!"), "err": (RED, "x"), "info": (CYAN, "i")}


def enable_ansi():
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k32.GetConsoleMode(h, ctypes.byref(mode)):
                k32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            os.system("")
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


def fmt_hr(h):
    if h is None:
        return "--"
    for unit, div in (("TH/s", 1e12), ("GH/s", 1e9), ("MH/s", 1e6), ("kH/s", 1e3)):
        if h >= div:
            return "%.2f %s" % (h / div, unit)
    return "%.0f H/s" % h


def fmt_eth(wei, digits=5):
    return "--" if wei is None else ("%." + str(digits) + "f") % (wei / float(ETH))


def fmt_dur(sec):
    sec = int(sec)
    return "%02d:%02d:%02d" % (sec // 3600, sec % 3600 // 60, sec % 60)


def short(addr):
    return addr[:6] + "…" + addr[-4:] if addr else "--"


def render(m, width=100):
    now = time.time()
    s = m.net
    line = DIM + "-" * width + RESET
    out = []
    mode = (YELLOW + "DRY-RUN" + RESET) if m.dry_run else (GREEN + "БОЕВОЙ" + RESET)
    out.append("%sUNICRED miner%s  %s  режим %s  работает %s   Ctrl+C = стоп" % (
        BOLD, RESET, time.strftime("%Y-%m-%d %H:%M:%S"), mode, fmt_dur(now - m.started)))
    out.append(line)
    if s:
        age = now - m.net_at
        rpc_ms = m.chain.rpc.last_latency
        out.append("СЕТЬ     блок %d   minted %d/%d   цена %s ETH   RPC %s%s" % (
            s["head"], s["minted"], P.MAX_SUPPLY, fmt_eth(m.price, 4),
            "%d мс" % (rpc_ms * 1000) if rpc_ms else "--",
            (RED + "  данные устарели %.0f с" % age + RESET) if age > 5 else ""))
        hpm = m.hashes_per_mint()
        iv = m.mint_interval()
        out.append("         мой target %s   global %s   ~2^%.1f хешей/минт   интервал %s   сеть ~%s" % (
            P.log2_str(s["target"]), P.log2_str(m.global_target) if m.global_target else "--",
            math.log2(hpm) if hpm else 0, "%.1f с" % iv if iv else "--", fmt_hr(m.net_hashrate())))
    else:
        out.append("СЕТЬ     нет данных от RPC %s" % (m.rpc_last_error[:70] if m.rpc_last_error else ""))
    share = m.share()
    eph = m.expected_per_hour()
    job = m.job
    if m.halt:
        jtxt = RED + "ПАУЗА: " + m.halt + RESET
    elif job:
        jtxt = "задание #%s (%.1f с), anchor -%d бл." % (
            job.id, now - job.created, (s["head"] - job.anchor_block) if s else 0)
    else:
        jtxt = "ожидание данных сети"
    out.append("МАЙНИНГ  хешрейт %s%s%s   доля %s   ожидается %s минтов/ч   %s" % (
        BOLD, fmt_hr(m.my_hashrate()), RESET, "%.1f%%" % (share * 100) if share is not None else "--",
        "%.2f" % eph if eph is not None else "--", jtxt))
    st = m.state
    out.append("КОШЕЛЁК  %s   баланс %s ETH   потрачено %s / %s ETH   минты ok %s%d%s / revert %d / в пути %d" % (
        short(m.address), fmt_eth(m.balance, 4), fmt_eth(st["spent_wei"]), m.cfg.max_total_spend_eth,
        GREEN, st["mints_ok"], RESET, st["mints_revert"], len(st["pending"])))
    out.append("         лимиты: цена <= %s ETH, остаток >= %s ETH, минтов <= %s   находки %d (устар. %d, кэш %d%s)" % (
        m.cfg.max_price_eth, m.cfg.min_balance_eth, m.cfg.max_mints, m.stats["found"], m.stats["stale"],
        m.stats["cached"], ", dry-run %d" % m.stats["dry"] if m.dry_run else ""))
    out.append(line)
    out.append(BOLD + "%-24s %-22s %-22s %12s %8s %6s" % ("СЕРВЕР", "СТАТУС", "GPU", "ХЕШРЕЙТ", "ПИНГ", "НАШЁЛ")
               + RESET)
    for c in m.conns:
        gpus = ""
        if c.gpus:
            names = sorted(set(g.replace("NVIDIA ", "").replace("GeForce ", "") for g in c.gpus))
            gpus = "%d× %s" % (len(c.gpus), names[0] if len(names) == 1 else "mixed")
        color = GREEN if c.ready else (RED if c.status == "ошибка" else YELLOW)
        out.append("%-24s %s%-22s%s %-22s %12s %8s %6d" % (
            c.spec.name[:24], color, c.status[:22], RESET, gpus[:22], fmt_hr(c.hashrate()),
            "%.0f мс" % c.ping_ms if c.ping_ms is not None else "--", c.found))
        if c.ready and c.hr_gpu and len(c.hr_gpu) > 1:
            out.append(DIM + "    " + "  ".join("gpu%d %s" % (i, fmt_hr(h)) for i, h in enumerate(c.hr_gpu)) + RESET)
        if not c.ready and c.last_error:
            out.append(DIM + "    " + c.last_error[:width - 4] + RESET)
    if not m.conns:
        out.append(DIM + "    нет серверов в servers.txt" + RESET)
    out.append(line)
    out.append(BOLD + "СОБЫТИЯ" + RESET)
    for t, level, msg, _ in list(m.rt.events)[-12:]:
        color, mark = LEVEL.get(level, ("", " "))
        out.append("%s %s%s %s%s" % (time.strftime("%H:%M:%S", time.localtime(t)), color, mark,
                                    msg[:width - 12], RESET))
    return out


def run_dashboard(m, refresh=1.0):
    enable_ansi()
    sys.stdout.write("\x1b[2J\x1b[H")
    while True:
        frame = render(m)
        sys.stdout.write("\x1b[H" + "\n".join(l + "\x1b[K" for l in frame) + "\n\x1b[J")
        sys.stdout.flush()
        time.sleep(refresh)


def run_plain(m, refresh=1.0):
    """No-dashboard mode: print events as lines + a status line every 30 s."""
    shown = 0
    last_status = 0
    while True:
        for t, level, msg, seq in list(m.rt.events):
            if seq > shown:
                shown = seq
                print("%s [%s] %s" % (time.strftime("%H:%M:%S", time.localtime(t)), level, msg), flush=True)
        if time.time() - last_status > 30:
            last_status = time.time()
            eph = m.expected_per_hour()
            print("%s [stat] %s, ожидается %s минтов/ч, minted %s" % (
                time.strftime("%H:%M:%S"), fmt_hr(m.my_hashrate()), "%.2f" % eph if eph else "--",
                m.net["minted"] if m.net else "--"), flush=True)
        time.sleep(refresh)
