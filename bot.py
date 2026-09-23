"""Telegram-бот (aiogram 3) для bigshort_checker. Запуск: python checker.py --bot"""

from __future__ import annotations

import html
import logging
import os
import time

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message

from checker import (
    CheckResult,
    check_many,
    extract_addresses,
    results_to_csv,
    summarize,
)

MAX_ADDRESSES = 1000
LIST_LIMIT = 30  # до стольких адресов отвечаем списком, больше — CSV
PROGRESS_STEP = 10
MAX_FILE_SIZE = 2 * 1024 * 1024
TG_MESSAGE_LIMIT = 4000  # чуть меньше 4096 с запасом

log = logging.getLogger("bigshort_bot")
dp = Dispatcher()

START_TEXT = (
    "👋 Проверка eligibility кошельков для аирдропа <b>$short</b> "
    "(airdrop.bigshort.xyz).\n\n"
    "Пришлите адреса <b>текстом</b> или <b>.txt-файлом</b> — в любом формате, "
    "я сам найду EVM (0x…) и Solana адреса и уберу дубли.\n\n"
    f"• до {LIST_LIMIT} адресов — ответ списком;\n"
    f"• больше — итог + results.csv;\n"
    f"• максимум {MAX_ADDRESSES} адресов за раз.\n\n"
    "✅ eligible · ❌ не eligible · ⚠️ ошибка / непонятный ответ"
)


def result_html(r: CheckResult) -> str:
    return f"{r.mark} <code>{html.escape(r.address)}</code>\n    {html.escape(r.details())}"


def summary_html(results: list[CheckResult]) -> str:
    return "\n".join(html.escape(line) for line in summarize(results).lines())


def split_messages(blocks: list[str], limit: int = TG_MESSAGE_LIMIT) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for block in blocks:
        if len(block) > limit:
            block = block[: limit - 1] + "…"
        candidate = f"{cur}\n\n{block}" if cur else block
        if len(candidate) > limit:
            chunks.append(cur)
            cur = block
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    return chunks


async def safe_edit(msg: Message, text: str) -> None:
    try:
        await msg.edit_text(text)
    except TelegramRetryAfter:
        pass  # прогресс не критичен — пропустим обновление
    except TelegramBadRequest as e:
        if "not modified" not in str(e):
            log.warning("edit_text: %s", e)


async def process(message: Message, text: str) -> None:
    addresses = extract_addresses(text)
    if not addresses:
        await message.answer("Не нашёл ни одного EVM или Solana адреса 🤷")
        return
    if len(addresses) > MAX_ADDRESSES:
        await message.answer(
            f"Слишком много адресов: {len(addresses)}. Лимит — {MAX_ADDRESSES} за раз."
        )
        return

    total = len(addresses)
    status = await message.answer(f"Проверяю {total} адресов…")
    last_edit = 0.0

    async def on_progress(done: int, all_: int) -> None:
        nonlocal last_edit
        if done == all_ or done % PROGRESS_STEP:
            return
        # не чаще раза в секунду, чтобы не ловить flood-limit
        if time.monotonic() - last_edit < 1:
            return
        last_edit = time.monotonic()
        await safe_edit(status, f"Проверяю {all_} адресов… {done}/{all_}")

    try:
        results = await check_many(addresses, on_progress=on_progress)
    except Exception as e:  # noqa: BLE001 — пользователь должен узнать, что проверка упала
        log.exception("check_many failed")
        await safe_edit(status, f"⚠️ Проверка упала: {html.escape(str(e))}")
        return
    summary = summary_html(results)

    if total <= LIST_LIMIT:
        await safe_edit(status, f"Готово: {total}/{total}")
        blocks = [result_html(r) for r in results] + [f"<b>Итог</b>\n{summary}"]
        for chunk in split_messages(blocks):
            await message.answer(chunk)
    else:
        await safe_edit(status, f"Готово: {total}/{total}\n\n{summary}")
        data = results_to_csv(results).encode("utf-8-sig")
        await message.answer_document(
            BufferedInputFile(data, filename="results.csv"),
            caption=summary,
        )


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(START_TEXT)


@dp.message(F.document)
async def on_document(message: Message, bot: Bot) -> None:
    doc = message.document
    name = (doc.file_name or "").lower()
    if not (name.endswith(".txt") or doc.mime_type == "text/plain"):
        await message.answer("Пришлите .txt-файл или адреса текстом.")
        return
    if doc.file_size and doc.file_size > MAX_FILE_SIZE:
        await message.answer("Файл слишком большой (максимум 2 МБ).")
        return
    buf = await bot.download(doc)
    text = buf.read().decode("utf-8", errors="ignore")
    if message.caption:
        text += "\n" + message.caption
    await process(message, text)


@dp.message(F.text)
async def on_text(message: Message) -> None:
    await process(message, message.text)


async def run_bot() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise SystemExit("BOT_TOKEN не задан (см. .env.example)")

    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    session = AiohttpSession(proxy=proxy) if proxy else AiohttpSession()
    bot = Bot(token, session=session, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    log.info("Бот запущен%s", f" через прокси {proxy}" if proxy else "")
    try:
        # handle_as_tasks (по умолчанию) — долгие проверки не блокируют других пользователей
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
