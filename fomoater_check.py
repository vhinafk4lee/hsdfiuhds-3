#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fomoater_check.py

Проверка авторизации ОДНОГО ЛИЧНОГО аккаунта на https://www.fomoater.com
через готовый session-токен, взятый вручную из cookies браузера после входа
через X (Twitter) OAuth.

Никаких прокси, антидетекта и мультиаккаунтинга — только один свой аккаунт.
Скрипт лишь подставляет уже полученную куку и смотрит, залогинен ли пользователь.
"""

import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# КОНФИГ
# ---------------------------------------------------------------------------

# Имя session-куки. По умолчанию "sessionid" — это лишь предположение.
# Как узнать настоящее имя куки:
#   1. Войдите на сайт вручную через X (Twitter).
#   2. Откройте DevTools (F12) → вкладка Application (Приложение)
#      → раздел Cookies → выберите https://www.fomoater.com
#   3. Найдите куку, которая хранит сессию (обычно длинное hex-значение),
#      и подставьте её имя сюда.
COOKIE_NAME = "sessionid"

# Токен НЕ хардкодим — берём из переменной окружения FOMOATER_TOKEN.
SESSION_TOKEN = os.environ.get("FOMOATER_TOKEN")

# Целевой сайт.
TARGET_URL = "https://www.fomoater.com"

# Путь для скриншота.
SCREENSHOT_PATH = "fomoater_check.png"

# Сколько секунд подождать после загрузки страницы (чтобы прогрузился JS/контент).
WAIT_SECONDS = 3


# ---------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------------

def validate_token(token: str) -> None:
    """
    Мягкая валидация токена.

    Если значение не выглядит как hex-строка или его длина не 32/40 символов —
    выводим предупреждение, но НЕ прерываем работу (сайт может использовать
    другой формат/длину).
    """
    is_hex = re.fullmatch(r"[0-9a-fA-F]+", token) is not None
    if not is_hex:
        print("[!] Предупреждение: токен не похож на hex-строку. "
              "Проверьте, что скопировали значение куки целиком.")
    elif len(token) not in (32, 40):
        print(f"[!] Предупреждение: длина токена {len(token)} символов "
              f"(ожидалось 32 или 40). Возможно, это не тот токен — "
              f"но продолжаем.")


def check_auth(page):
    """
    Проверка авторизации на открытой странице.

    Возвращает кортеж (authorized: bool | None, reason: str):
      - (True,  причина)  — найден положительный признак (мы залогинены);
      - (False, причина)  — найден отрицательный признак (мы НЕ залогинены);
      - (None,  причина)  — ничего однозначного не нашли ("не удалось определить").
    """
    # 1) Проверяем URL: редирект на страницу логина / OAuth X говорит о том,
    #    что мы не авторизованы.
    current_url = (page.url or "").lower()
    if "login" in current_url or "x.com/i/oauth" in current_url:
        return False, f"после загрузки произошёл редирект на страницу входа: {page.url}"

    # 2) Положительные признаки — видимый текст, который есть только у залогиненного
    #    пользователя. Регистронезависимо через regex.
    positive_patterns = [
        r"logout",
        r"sign\s*out",
        r"log\s*out",
        r"profile",
        r"my\s*card",
    ]
    for pattern in positive_patterns:
        locator = page.get_by_text(re.compile(pattern, re.IGNORECASE))
        # Берём первый элемент и проверяем видимость (count>0 недостаточно —
        # элемент может быть в скрытом меню).
        try:
            if locator.count() > 0 and locator.first.is_visible():
                return True, f"найден положительный признак авторизации: '{pattern}'"
        except Exception:
            # Если локатор по какой-то причине не проверился — просто идём дальше.
            continue

    # 3) Отрицательные признаки — видимые кнопки/ссылки для входа.
    negative_texts = [
        "Login with X",
        "Sign in with X",
        "Connect X",
        "Login with Twitter",
    ]
    for text in negative_texts:
        locator = page.get_by_text(re.compile(re.escape(text), re.IGNORECASE))
        try:
            if locator.count() > 0 and locator.first.is_visible():
                return False, f"на странице видна кнопка входа: '{text}'"
        except Exception:
            continue

    # 4) Ничего однозначного не нашли.
    return None, ("не найдено ни положительных, ни отрицательных признаков — "
                  "не удалось определить статус авторизации автоматически "
                  "(посмотрите скриншот вручную)")


# ---------------------------------------------------------------------------
# ОСНОВНАЯ ЛОГИКА
# ---------------------------------------------------------------------------

def main() -> int:
    # Токен обязателен.
    if not SESSION_TOKEN:
        print("[X] Ошибка: не задана переменная окружения FOMOATER_TOKEN.")
        print("    Задайте токен сессии перед запуском:")
        print("      Windows PowerShell:  $env:FOMOATER_TOKEN=\"<ваш_токен>\"")
        print("      Linux/macOS:         export FOMOATER_TOKEN=\"<ваш_токен>\"")
        return 1

    # Мягкая валидация — только предупреждения, работу не останавливаем.
    validate_token(SESSION_TOKEN)

    # Импорт Playwright с понятной ошибкой, если он не установлен.
    try:
        from playwright.sync_api import sync_playwright
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    except ImportError:
        print("[X] Ошибка: не установлен пакет playwright.")
        print("    Установите его и браузер командами:")
        print("      pip install playwright")
        print("      playwright install chromium")
        return 1

    browser = None
    try:
        with sync_playwright() as p:
            # 3) Запуск Chromium в видимом режиме.
            try:
                browser = p.chromium.launch(headless=False)
            except PlaywrightError as e:
                # Частый случай — не установлен сам браузер Chromium.
                print("[X] Ошибка запуска браузера Chromium.")
                print("    Возможно, браузер не установлен. Выполните:")
                print("      playwright install chromium")
                print(f"    Детали: {e}")
                return 1

            # 4) Контекст с реалистичными параметрами обычного Chrome на Windows.
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            )
            context = browser.new_context(
                user_agent=user_agent,
                viewport={"width": 1366, "height": 768},
                locale="ru-RU",
            )

            # 5) Добавляем session-куку.
            try:
                context.add_cookies([{
                    "name": COOKIE_NAME,
                    "value": SESSION_TOKEN,
                    "domain": ".fomoater.com",
                    "path": "/",
                    "httpOnly": True,
                    "secure": True,
                    "sameSite": "Lax",
                }])
            except Exception as e:
                print("[X] Ошибка при добавлении session-куки.")
                print(f"    Проверьте имя куки (COOKIE_NAME) и значение токена. Детали: {e}")
                return 1

            page = context.new_page()

            # 6) Переход на целевой сайт.
            try:
                page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30_000)
            except PlaywrightTimeoutError:
                print("[X] Ошибка: страница не загрузилась за 30 секунд (таймаут).")
                print("    Проверьте интернет-соединение и доступность сайта.")
                # Всё равно попробуем сделать скриншот того, что успело загрузиться.
                _safe_screenshot(page)
                return 1
            except PlaywrightError as e:
                # Сетевые ошибки / сайт недоступен.
                print("[X] Ошибка сети или сайт недоступен.")
                print(f"    Не удалось открыть {TARGET_URL}. Детали: {e}")
                return 1

            # Небольшая пауза, чтобы прогрузился динамический контент.
            time.sleep(WAIT_SECONDS)

            # 7) Проверка авторизации.
            authorized, reason = check_auth(page)
            if authorized is True:
                print("[OK] Авторизация успешна")
                print(f"     Причина: {reason}")
            elif authorized is False:
                print("[FAIL] Авторизация не удалась")
                print(f"       Причина: {reason}")
            else:
                print("[?] Не удалось определить статус авторизации")
                print(f"    Причина: {reason}")

            # 8) Скриншот в любом случае.
            _safe_screenshot(page)

            return 0

    except Exception as e:
        # 10) Любая прочая ошибка — с типом исключения.
        print(f"[X] Непредвиденная ошибка ({type(e).__name__}): {e}")
        return 1
    finally:
        # 9) Закрываем браузер в любом случае.
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass


def _safe_screenshot(page) -> None:
    """Делает скриншот всей страницы, не роняя скрипт при ошибке."""
    try:
        page.screenshot(path=SCREENSHOT_PATH, full_page=True)
        print(f"[i] Скриншот сохранён: {SCREENSHOT_PATH}")
    except Exception as e:
        print(f"[!] Не удалось сохранить скриншот: {e}")


if __name__ == "__main__":
    sys.exit(main())
