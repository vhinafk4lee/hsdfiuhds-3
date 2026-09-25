"""bigshort_checker — массовая проверка eligibility кошельков для аирдропа $short.

Запуск:
    python checker.py wallets.txt   # CLI: печать в консоль + results.csv
    python checker.py --bot         # Telegram-бот (aiogram 3)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import os
import re
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Awaitable, Callable, Iterable, Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE", "https://airdrop.bigshort.xyz")
AIRDROP_ID = "airdrop-20260920-v1"
SITE_URL = "https://airdrop.bigshort.xyz/"
CLAIM_URL = os.getenv("CLAIM_URL", "https://airdrop.bigshort.xyz/?ref=EVM-5150")
TOKEN_SYMBOL = "SHORT"

REQUEST_TIMEOUT = 30
MAX_RETRIES = max(0, int(os.getenv("MAX_RETRIES", "3")))  # ретраи на 429 (помимо первой попытки)
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.0"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Origin": SITE_URL.rstrip("/"),
    "Referer": SITE_URL,
}

MARK_OK = "✅"
MARK_NO = "❌"
MARK_WARN = "⚠️"

CSV_COLUMNS = [
    "address", "eligible", "status", "amount", "multiplier",
    "protocols", "claimed", "claimed_to", "tx",
]


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, default)))
    except ValueError:
        return default


CONCURRENCY = _env_int("CONCURRENCY", 5)
DELAY = _env_float("DELAY", 0.3)


# --------------------------------------------------------------------------- #
# Извлечение адресов
# --------------------------------------------------------------------------- #

B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_CLASS = "1-9A-HJ-NP-Za-km-z"

EVM_RE = re.compile(r"(?<![0-9a-zA-Z])0x[0-9a-fA-F]{40}(?![0-9a-zA-Z])")
SOLANA_RE = re.compile(rf"(?<![{_B58_CLASS}])[{_B58_CLASS}]{{32,44}}(?![{_B58_CLASS}])")


def _b58_decoded_len(s: str) -> int:
    num = 0
    for ch in s:
        num = num * 58 + B58_ALPHABET.index(ch)
    body = (num.bit_length() + 7) // 8
    leading_zeros = len(s) - len(s.lstrip("1"))
    return leading_zeros + body


def is_solana_address(s: str) -> bool:
    """Solana-адрес — base58, который декодируется ровно в 32 байта."""
    if not 32 <= len(s) <= 44 or any(ch not in B58_ALPHABET for ch in s):
        return False
    return _b58_decoded_len(s) == 32


def is_evm_address(s: str) -> bool:
    return bool(re.fullmatch(r"0x[0-9a-fA-F]{40}", s))


def detect_namespace(address: str) -> str:
    return "evm" if is_evm_address(address) else "solana"


def extract_addresses(text: str) -> list[str]:
    """Достаёт EVM и Solana адреса из произвольного текста, без дублей, в порядке появления."""
    found: list[tuple[int, str]] = []
    for m in EVM_RE.finditer(text):
        found.append((m.start(), m.group()))
    # Вырезаем EVM-адреса (и прочие 0x-хеши), чтобы их хвосты не приняли за base58.
    masked = re.sub(r"0x[0-9a-fA-F]+", lambda m: " " * len(m.group()), text)
    for m in SOLANA_RE.finditer(masked):
        if is_solana_address(m.group()):
            found.append((m.start(), m.group()))
    found.sort(key=lambda x: x[0])

    seen: set[str] = set()
    result: list[str] = []
    for _, addr in found:
        key = addr.lower() if addr.startswith("0x") else addr
        if key not in seen:
            seen.add(key)
            result.append(addr)
    return result


# --------------------------------------------------------------------------- #
# Форматирование
# --------------------------------------------------------------------------- #

def fmt_amount(value: Optional[Decimal]) -> str:
    """40000 -> '40 000', 1234.5678 -> '1 234.5678'."""
    if value is None:
        return ""
    q = value.quantize(Decimal("0.0001")).normalize()
    sign = "-" if q < 0 else ""
    q = abs(q)
    int_part = int(q)
    frac = q - int_part
    s = f"{int_part:,}".replace(",", " ")
    if frac:
        s += "." + format(frac, "f").split(".")[1]
    return sign + s


def fmt_plain(value: Optional[Decimal]) -> str:
    if value is None:
        return ""
    return format(value.normalize(), "f")


def fmt_multiplier(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return fmt_plain(Decimal(str(value)))
    except InvalidOperation:
        return str(value)


def fmt_protocols(contributions) -> str:
    parts = []
    for c in contributions or []:
        if not isinstance(c, dict):
            continue
        name = c.get("protocol") or "?"
        try:
            vol = f"${Decimal(str(c.get('volumeUsd') or 0)):,.0f}"
        except InvalidOperation:
            vol = f"${c.get('volumeUsd')}"
        trades = c.get("tradeCount")
        parts.append(f"{name}: {vol} / {trades if trades is not None else '?'} tx")
    return "; ".join(parts)


def short(s: str, head: int = 6, tail: int = 4) -> str:
    return s if len(s) <= head + tail + 1 else f"{s[:head]}…{s[-tail:]}"


# --------------------------------------------------------------------------- #
# Проверка
# --------------------------------------------------------------------------- #

@dataclass
class CheckResult:
    address: str
    namespace: str
    mark: str = MARK_WARN
    eligible: Optional[bool] = None
    status: str = ""
    reasons: list[str] = field(default_factory=list)
    amount: Optional[Decimal] = None
    multiplier: str = ""
    protocols: str = ""
    claimed: Optional[bool] = None
    claimed_to: str = ""
    tx: str = ""
    error: str = ""

    def csv_row(self) -> dict:
        def yes_no(v: Optional[bool]) -> str:
            return "" if v is None else ("да" if v else "нет")

        status = self.status
        if self.reasons:
            status += f" ({', '.join(self.reasons)})"
        if self.error:
            status = f"{status}: {self.error}" if status else self.error
        return {
            "address": self.address,
            "eligible": yes_no(self.eligible) if self.mark != MARK_WARN else "?",
            "status": status,
            "amount": fmt_plain(self.amount),
            "multiplier": self.multiplier,
            "protocols": self.protocols,
            "claimed": yes_no(self.claimed),
            "claimed_to": self.claimed_to,
            "tx": self.tx,
        }

    def details(self) -> str:
        """Описание без адреса и метки (для консоли и бота)."""
        if self.mark == MARK_WARN:
            return self.error or self.status or "непонятный ответ"
        if self.mark == MARK_NO:
            s = f"не eligible, status: {self.status or '?'}"
            if self.reasons:
                s += f" ({', '.join(self.reasons)})"
            return s
        parts = [f"{fmt_amount(self.amount)} {TOKEN_SYMBOL}" if self.amount is not None else "сумма ?"]
        if self.multiplier:
            parts[0] += f" ×{self.multiplier}"
        if self.protocols:
            parts.append(self.protocols)
        if self.claimed:
            parts.append(f"claimed: да → {self.claimed_to or '?'}, tx {self.tx or '?'}")
        elif self.claimed is False:
            parts.append("claimed: нет")
        return " | ".join(parts)

    def line(self) -> str:
        return f"{self.mark} {self.address} | {self.details()}"


def parse_response(address: str, namespace: str, payload) -> CheckResult:
    res = CheckResult(address=address, namespace=namespace)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        res.error = "в ответе нет data"
        return res

    src = (data.get("source") or {}).get("address")
    if not src or str(src).lower() != address.lower():
        res.status = "mismatch"
        res.error = f"ответ по другому адресу: {src or '—'}"
        return res

    elig = data.get("eligibility") or {}
    res.status = str(elig.get("status") or "")
    res.reasons = [str(r) for r in (elig.get("reasonCodes") or [])]

    alloc = data.get("allocation") or {}
    raw = alloc.get("currentRaw") or alloc.get("estimatedRaw")
    if raw not in (None, ""):
        try:
            decimals = int(alloc.get("tokenDecimals") or 0)
            res.amount = Decimal(str(raw)).scaleb(-decimals)
        except (InvalidOperation, ValueError, TypeError):
            pass
    res.multiplier = fmt_multiplier(alloc.get("multiplier"))
    res.protocols = fmt_protocols(data.get("contributions"))

    claimed = data.get("claimed")
    res.claimed = bool(claimed)
    if isinstance(claimed, dict):
        res.claimed_to = str(claimed.get("recipient") or "")
        res.tx = str(claimed.get("txHash") or "")

    if res.status == "eligible":
        res.eligible, res.mark = True, MARK_OK
    elif res.status:
        res.eligible, res.mark = False, MARK_NO
    else:
        res.error = "в ответе нет eligibility.status"
    return res


async def check_address(session: aiohttp.ClientSession, address: str) -> CheckResult:
    namespace = detect_namespace(address)
    url = f"{API_BASE.rstrip('/')}/api/v1/airdrops/{AIRDROP_ID}/eligibility"
    params = {"namespace": namespace, "address": address}

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429 and attempt < MAX_RETRIES:
                    pause = RETRY_BASE_DELAY * 2 ** attempt
                    try:
                        pause = max(pause, float(resp.headers.get("Retry-After", 0)))
                    except ValueError:
                        pass
                    await asyncio.sleep(pause)
                    continue
                if resp.status != 200:
                    body = (await resp.text())[:200].strip()
                    msg = f"HTTP {resp.status}"
                    if resp.status == 429:
                        msg += f" (rate limit, {MAX_RETRIES} ретрая не помогли)"
                    if body:
                        msg += f": {body}"
                    return CheckResult(address, namespace, error=msg)
                try:
                    payload = await resp.json(content_type=None)
                except ValueError:
                    return CheckResult(address, namespace, error="ответ не JSON")
                return parse_response(address, namespace, payload)
        except asyncio.TimeoutError:
            return CheckResult(address, namespace, error=f"таймаут {REQUEST_TIMEOUT}с")
        except aiohttp.ClientError as e:
            return CheckResult(address, namespace, error=f"сеть: {type(e).__name__}: {e}")
    return CheckResult(address, namespace, error="HTTP 429")  # недостижимо


ProgressCb = Callable[[int, int], Awaitable[None]]


def make_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        headers=HEADERS,
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        trust_env=True,  # HTTPS_PROXY / HTTP_PROXY / NO_PROXY из окружения
    )


async def check_many(
    addresses: list[str],
    concurrency: int = CONCURRENCY,
    delay: float = DELAY,
    on_progress: Optional[ProgressCb] = None,
) -> list[CheckResult]:
    """Проверяет адреса параллельно; результаты в исходном порядке."""
    sem = asyncio.Semaphore(concurrency)
    results: list[Optional[CheckResult]] = [None] * len(addresses)
    done = 0

    async with make_session() as session:
        async def worker(i: int, addr: str) -> None:
            nonlocal done
            async with sem:
                results[i] = await check_address(session, addr)
                if delay:
                    await asyncio.sleep(delay)
            done += 1
            if on_progress:
                await on_progress(done, len(addresses))

        await asyncio.gather(*(worker(i, a) for i, a in enumerate(addresses)))
    return [r for r in results if r is not None]


# --------------------------------------------------------------------------- #
# Итоги и CSV
# --------------------------------------------------------------------------- #

@dataclass
class Summary:
    total: int
    eligible: int
    errors: int
    total_amount: Decimal
    claimed: int
    claimed_amount: Decimal

    @property
    def to_claim(self) -> int:
        return self.eligible - self.claimed

    def lines(self) -> list[str]:
        lines = [
            f"Eligible: {self.eligible}/{self.total}",
            f"Сумма {TOKEN_SYMBOL} по eligible: {fmt_amount(self.total_amount)}",
            f"Уже склеймлено: {self.claimed} ({fmt_amount(self.claimed_amount)} {TOKEN_SYMBOL})",
            f"Ошибок/непонятных: {self.errors}",
        ]
        if self.to_claim and CLAIM_URL:
            lines.append(f"Клейм ({self.to_claim} ещё не склеймлено): {CLAIM_URL}")
        return lines


def summarize(results: Iterable[CheckResult]) -> Summary:
    results = list(results)
    ok = [r for r in results if r.mark == MARK_OK]
    claimed = [r for r in ok if r.claimed]
    return Summary(
        total=len(results),
        eligible=len(ok),
        errors=sum(r.mark == MARK_WARN for r in results),
        total_amount=sum((r.amount or Decimal(0) for r in ok), Decimal(0)),
        claimed=len(claimed),
        claimed_amount=sum((r.amount or Decimal(0) for r in claimed), Decimal(0)),
    )


def results_to_csv(results: Iterable[CheckResult]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for r in results:
        writer.writerow(r.csv_row())
    return buf.getvalue()


def write_csv(results: Iterable[CheckResult], path: str) -> None:
    # utf-8-sig — чтобы Excel корректно открыл кириллицу
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(results_to_csv(results))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

async def run_cli(path: str, out: str) -> int:
    with open(path, encoding="utf-8", errors="ignore") as f:
        addresses = extract_addresses(f.read())
    if not addresses:
        print("Адреса не найдены.")
        return 1

    print(f"Проверяю {len(addresses)} адресов (CONCURRENCY={CONCURRENCY}, DELAY={DELAY})…\n")

    async def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total}", end="", file=sys.stderr, flush=True)

    results = await check_many(addresses, on_progress=progress)
    print("\r" + " " * 20 + "\r", end="", file=sys.stderr)

    for r in results:
        print(r.line())
    write_csv(results, out)

    print()
    for line in summarize(results).lines():
        print(line)
    print(f"CSV: {out}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except AttributeError:
                pass

    parser = argparse.ArgumentParser(description="Проверка eligibility кошельков для аирдропа $short")
    parser.add_argument("file", nargs="?", help="файл с адресами (любой текст)")
    parser.add_argument("--bot", action="store_true", help="запустить Telegram-бота")
    parser.add_argument("-o", "--out", default="results.csv", help="куда сохранить CSV (по умолчанию results.csv)")
    args = parser.parse_args(argv)

    if args.bot:
        from bot import run_bot

        asyncio.run(run_bot())
        return 0
    if not args.file:
        parser.print_help()
        return 2
    return asyncio.run(run_cli(args.file, args.out))


if __name__ == "__main__":
    sys.exit(main())
