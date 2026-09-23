"""Тесты с локальным aiohttp mock-сервером. Запуск: python -m pytest -v"""

import asyncio
import os
import sys
from decimal import Decimal

import pytest
from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import checker  # noqa: E402

EVM_OK = "0x27dF1a5bE6F3e2b0c4A1d9e8F7a6B5c4D3e2F1a0"
SOL_NO = "So11111111111111111111111111111111111111112"
SOL_FOREIGN = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
SOL_OTHER = "7bLzPbGyCWrUv8Wf3y1AZX3oVhe7Tb1wP4xkZ2uHrXhD"
EVM_429 = "0x" + "1" * 40
EVM_500 = "0x" + "2" * 40


def payload(address, namespace, status="eligible", reasons=(), claimed=None):
    return {
        "data": {
            "source": {"namespace": namespace, "address": address},
            "allocation": {
                "currentRaw": "40000000000000000000000",
                "estimatedRaw": "40000000000000000000000",
                "tokenDecimals": 18,
                "multiplier": 3,
                "provisional": False,
            },
            "contributions": [
                {"protocol": "pump", "tradeCount": "167", "volumeUsd": "41851.16", "share": 28}
            ],
            "claimed": claimed,
            "eligibility": {"status": status, "reasonCodes": list(reasons)},
        },
        "meta": {"requestId": "test"},
    }


@pytest.fixture
def mock_api(monkeypatch):
    hits = {"429": 0}
    seen_headers = {}

    async def handler(request: web.Request):
        ns = request.query["namespace"]
        addr = request.query["address"]
        seen_headers.update(request.headers)
        if addr == EVM_OK:
            assert ns == "evm"
            # API вернул адрес в нижнем регистре — должно совпасть без учёта регистра
            return web.json_response(payload(addr.lower(), ns, claimed={
                "amountRaw": "40000000000000000000000",
                "recipient": "0x27df00000000000000000000000000000000beef",
                "createdAt": "2026-09-23T07:55:53Z",
                "txHash": "0xa296",
            }))
        if addr == SOL_NO:
            assert ns == "solana"
            return web.json_response(payload(addr, ns, status="ineligible",
                                             reasons=["LOW_VOLUME", "SYBIL"]))
        if addr == SOL_FOREIGN:
            return web.json_response(payload(SOL_OTHER, ns))
        if addr == EVM_429:
            hits["429"] += 1
            if hits["429"] <= 2:
                return web.json_response({"error": "slow down"}, status=429)
            return web.json_response(payload(addr, ns))
        if addr == EVM_500:
            return web.Response(status=500, text="boom")
        return web.json_response({"error": "not found"}, status=404)

    async def start():
        app = web.Application()
        app.router.add_get("/api/v1/airdrops/{airdrop}/eligibility", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        return runner, f"http://127.0.0.1:{port}"

    # никакие прокси из окружения не должны перехватывать запросы к localhost
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(checker, "RETRY_BASE_DELAY", 0.01)

    async def run(addresses):
        runner, base = await start()
        monkeypatch.setattr(checker, "API_BASE", base)
        try:
            return await checker.check_many(addresses, concurrency=3, delay=0)
        finally:
            await runner.cleanup()

    return run, hits, seen_headers


def test_three_cases_marks(mock_api):
    run, _, headers = mock_api
    results = asyncio.run(run([EVM_OK, SOL_NO, SOL_FOREIGN]))
    ok, no, foreign = results

    assert ok.mark == checker.MARK_OK
    assert ok.eligible is True
    assert ok.amount == Decimal(40000)
    assert ok.multiplier == "3"
    assert ok.protocols == "pump: $41,851 / 167 tx"
    assert ok.claimed is True and ok.tx == "0xa296"
    assert "40 000 SHORT ×3" in ok.line()

    assert no.mark == checker.MARK_NO
    assert no.eligible is False
    assert no.status == "ineligible"
    assert no.reasons == ["LOW_VOLUME", "SYBIL"]
    assert "LOW_VOLUME, SYBIL" in no.line()

    assert foreign.mark == checker.MARK_WARN
    assert foreign.mark != checker.MARK_OK
    assert "ответ по другому адресу" in foreign.line()
    assert foreign.amount is None

    assert headers["Origin"] == "https://airdrop.bigshort.xyz"
    assert headers["Referer"] == "https://airdrop.bigshort.xyz/"
    assert headers["Accept"] == "application/json"

    s = checker.summarize(results)
    assert (s.eligible, s.total, s.claimed, s.errors) == (1, 3, 1, 1)
    assert s.total_amount == Decimal(40000)

    csv_text = checker.results_to_csv(results)
    assert csv_text.splitlines()[0] == ",".join(checker.CSV_COLUMNS)


def test_429_retry_and_http_errors(mock_api):
    run, hits, _ = mock_api
    r429, r500, r404 = asyncio.run(run([EVM_429, EVM_500, "0x" + "3" * 40]))
    assert r429.mark == checker.MARK_OK and hits["429"] == 3
    assert r500.mark == checker.MARK_WARN and "HTTP 500" in r500.error
    assert r404.mark == checker.MARK_WARN and "HTTP 404" in r404.error


def test_network_error(monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(checker, "API_BASE", "http://127.0.0.1:1")
    (r,) = asyncio.run(checker.check_many([EVM_OK], delay=0))
    assert r.mark == checker.MARK_WARN and r.error.startswith("сеть:")


def test_extract_addresses():
    text = f"""
    wallets: {EVM_OK}, {EVM_OK.lower()} ; {SOL_NO}
    tx 0x{'a' * 64} — не адрес
    {SOL_NO} дубль, мусор abc 123, {SOL_FOREIGN}
    """
    assert checker.extract_addresses(text) == [EVM_OK, SOL_NO, SOL_FOREIGN]
    assert checker.detect_namespace(EVM_OK) == "evm"
    assert checker.detect_namespace(SOL_NO) == "solana"


def test_formatting():
    assert checker.fmt_amount(Decimal("40000")) == "40 000"
    assert checker.fmt_amount(Decimal("1234567.5")) == "1 234 567.5"
    assert checker.fmt_amount(Decimal("12")) == "12"
