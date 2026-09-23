# bigshort_checker

Массовая проверка eligibility кошельков для аирдропа **$short** ([airdrop.bigshort.xyz](https://airdrop.bigshort.xyz/)).
Поддерживаются EVM (`0x…`) и Solana (base58) адреса. Два режима: CLI и Telegram-бот.

Метки:

| | значение |
|---|---|
| ✅ | eligible |
| ❌ | не eligible (показываются `status` и `reasonCodes`) |
| ⚠️ | ошибка HTTP/сети, 429 после ретраев, непонятный ответ или **ответ по другому адресу** |

Если API вернул данные по адресу, который не совпадает с запрошенным (сравнение без учёта регистра),
строка помечается ⚠️ «ответ по другому адресу», а не ✅.

## Установка

Нужен Python 3.10+.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

## Настройка `.env`

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

| переменная | по умолчанию | описание |
|---|---|---|
| `CONCURRENCY` | `5` | сколько адресов проверяется параллельно |
| `DELAY` | `0.3` | пауза (сек) после каждого запроса в каждом потоке |
| `BOT_TOKEN` | — | токен бота от @BotFather (только для `--bot`) |
| `HTTPS_PROXY` | — | прокси, например `http://127.0.0.1:10809` |
| `CLAIM_URL` | `https://airdrop.bigshort.xyz/?ref=EVM-5150` | ссылка на клейм в итоге (если есть eligible, ещё не склеймленные); пустое значение — не показывать |

Прокси из `HTTPS_PROXY` используется и для запросов к API (через `trust_env=True`), и для Telegram.
Если получаете много ⚠️ `HTTP 429`, уменьшите `CONCURRENCY` и/или увеличьте `DELAY`.

## Запуск: CLI

Положите адреса в `wallets.txt` — формат любой: по одному в строке, через запятую,
вперемешку с текстом. Адреса извлекаются регуляркой, дубли удаляются (EVM — без учёта регистра).

```bash
python checker.py wallets.txt
python checker.py wallets.txt -o my_results.csv   # другой путь для CSV
```

Пример вывода:

```
✅ 0x27dF…F1a0 | 40 000 SHORT ×3 | pump: $41,851 / 167 tx | claimed: да → 0x27df…, tx 0xa296…
❌ So111…1112 | не eligible, status: ineligible (LOW_VOLUME)
⚠️ Tokenkeg…5DA | ответ по другому адресу: 7bLz…
⚠️ 0x2222…2222 | HTTP 500: boom

Eligible: 1/4
Сумма SHORT по eligible: 40 000
Уже склеймлено: 1 (40 000 SHORT)
Ошибок/непонятных: 2
```

Результаты сохраняются в `results.csv` (UTF-8 с BOM, открывается в Excel) с колонками:
`address, eligible, status, amount, multiplier, protocols, claimed, claimed_to, tx`.
В CSV `amount` — обычное число (`40000`), чтобы его можно было суммировать в таблице.

## Запуск: Telegram-бот

1. Создайте бота у [@BotFather](https://t.me/BotFather) и пропишите `BOT_TOKEN` в `.env`.
2. Запустите:

   ```bash
   python checker.py --bot
   ```

3. В боте: `/start` — инструкция. Отправьте адреса текстом или `.txt`-файлом.
   - до 30 адресов — ответ списком;
   - больше 30 — итог в сообщении + файл `results.csv`;
   - лимит — 1000 адресов за раз;
   - прогресс обновляется примерно каждые 10 адресов.

## Тесты

Тесты поднимают локальный aiohttp mock-сервер (eligible / ineligible с reasonCodes /
ответ с чужим `source.address`, плюс 429-ретраи и ошибки HTTP/сети):

```bash
python -m pytest -v
```
