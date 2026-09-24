"""`traits`: collect tokenURI metadata of recent mints together with their PoW digest,
to find out what the traits (e.g. Rarity) depend on."""
import base64
import csv
import json
import statistics
from collections import Counter, defaultdict

import requests

SEL_TOKEN_URI = "c87b56dd"  # tokenURI(uint256)
IPFS_GATEWAY = "https://ipfs.io/ipfs/"


def decode_abi_string(hexdata):
    data = bytes.fromhex(hexdata[2:] if hexdata.startswith("0x") else hexdata)
    off = int.from_bytes(data[0:32], "big")
    n = int.from_bytes(data[off:off + 32], "big")
    return data[off + 32:off + 32 + n].decode("utf-8", errors="replace")


def load_metadata(uri, timeout=15):
    if uri.startswith("data:"):
        head, _, body = uri.partition(",")
        if ";base64" in head:
            body = base64.b64decode(body).decode("utf-8", errors="replace")
        else:
            from urllib.parse import unquote
            body = unquote(body)
        return json.loads(body)
    if uri.startswith("ipfs://"):
        uri = IPFS_GATEWAY + uri[7:]
    r = requests.get(uri, timeout=timeout)
    r.raise_for_status()
    return r.json()


def attributes(meta):
    out = {}
    for a in meta.get("attributes") or []:
        if isinstance(a, dict) and "trait_type" in a:
            out[str(a["trait_type"])] = str(a.get("value"))
    return out


def leading_zero_bits(digest_hex):
    v = int(digest_hex, 16)
    return 256 - v.bit_length()


def collect(chain, count=200, chunk=2000, max_blocks=400000, progress=print):
    head = chain.block_number()
    logs, to = [], head
    while len(logs) < count and head - to < max_blocks:
        frm = max(0, to - chunk + 1)
        try:
            part = chain.mint_logs(frm, to)
        except Exception:
            if chunk > 200:
                chunk //= 2
                continue
            raise
        logs = part + logs
        progress("  блоки %d..%d: минтов %d" % (frm, to, len(logs)))
        to = frm - 1
        if to <= 0:
            break
    logs = sorted(logs, key=lambda m: m["token_id"])[-count:]
    rows = []
    for i, m in enumerate(logs):
        row = {"token_id": m["token_id"], "block": m["block"], "miner": m["miner"],
               "digest": m["digest"], "anchor_block": m["anchor_block"], "price_wei": m["price"],
               "zero_bits": leading_zero_bits(m["digest"]), "digest_last_byte": int(m["digest"][-2:], 16),
               "uri": "", "error": ""}
        try:
            uri = decode_abi_string(chain.call("0x" + SEL_TOKEN_URI + "%064x" % m["token_id"]))
            row["uri"] = uri if not uri.startswith("data:") else uri[:40] + "…"
            meta = load_metadata(uri)
            for k, v in attributes(meta).items():
                row["trait:" + k] = v
        except Exception as exc:
            row["error"] = str(exc)[:120]
        rows.append(row)
        if (i + 1) % 20 == 0:
            progress("  метаданные: %d/%d" % (i + 1, len(logs)))
    return rows


def write_csv(rows, path):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(str(path), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summary(rows, trait="Rarity"):
    """Text report: distribution of a trait and how it relates to digest / tokenId."""
    key = "trait:" + trait
    lines = []
    have = [r for r in rows if key in r]
    if not have:
        names = sorted({k[6:] for r in rows for k in r if k.startswith("trait:")})
        errs = [r["error"] for r in rows if r.get("error")]
        lines.append("трейт %r не найден. Найденные трейты: %s" % (trait, ", ".join(names) or "нет"))
        if errs:
            lines.append("ошибки чтения метаданных, пример: %s" % errs[0])
        return lines
    by = defaultdict(list)
    for r in have:
        by[r[key]].append(r)
    total = len(have)
    lines.append("%s: %d токенов (#%d..#%d)" % (trait, total, have[0]["token_id"], have[-1]["token_id"]))
    lines.append("%-14s %6s %7s   нулевых бит digest (мин/медиана/макс)   последний байт digest (мин..макс)" % (
        "значение", "шт", "доля"))
    for val, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        z = [r["zero_bits"] for r in rs]
        lb = [r["digest_last_byte"] for r in rs]
        lines.append("%-14s %6d %6.1f%%   %3d / %5.1f / %3d                         %3d..%3d" % (
            val[:14], len(rs), 100.0 * len(rs) / total, min(z), statistics.median(z), max(z), min(lb), max(lb)))
    ex = by.get("Legendary") or []
    if ex:
        lines.append("Legendary: #" + ", #".join(str(r["token_id"]) for r in ex[:30]))
    return lines


def main_counter(rows, trait="Rarity"):
    return Counter(r.get("trait:" + trait) for r in rows)


def push20_addresses(code_hex):
    """Addresses embedded in bytecode as PUSH20 constants."""
    code = bytes.fromhex(code_hex[2:] if code_hex.startswith("0x") else code_hex)
    out, i = [], 0
    while i < len(code):
        op = code[i]
        if 0x60 <= op <= 0x7f:
            n = op - 0x5f
            if op == 0x73:
                a = "0x" + code[i + 1:i + 21].hex()
                if a not in out and int(a, 16) > 0xffff:
                    out.append(a)
            i += n + 1
        else:
            i += 1
    return out


def dump_code(chain, contract, directory):
    """Save bytecode of the contract and of every contract it references (renderer, …)."""
    saved = {}
    queue = [contract.lower()]
    while queue and len(saved) < 12:
        addr = queue.pop(0)
        if addr in saved:
            continue
        try:
            code = chain.rpc.call("eth_getCode", [addr, "latest"])
        except Exception:
            continue
        if not code or code == "0x":
            continue
        saved[addr] = len(code) // 2 - 1
        (directory / ("code_%s.hex" % addr)).write_text(code, encoding="ascii")
        queue += [a for a in push20_addresses(code) if a not in saved]
    return saved
