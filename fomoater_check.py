#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fomoater_check.py

Проверка авторизации ОДНОГО ЛИЧНОГО аккаунта на https://www.fomoater.com
через готовый session-токен, взятый вручную из cookies браузера после входа
через X (Twitter) OAuth.

Версия на requests — БЕЗ открытия браузера. Скрипт делает обычный HTTP-запрос
с подставленной session-кукой и анализирует ответ (редиректы, статус, HTML)
на признаки авторизации.

Никаких прокси, антидетекта и мультиаккаунтинга — только один свой аккаунт.
"""

import os
import re
import sys

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

# Токен НЕ хардкодим. Берём его так:
#   1) сначала из переменной окружения FOMOATER_TOKEN (если задана);
#   2) если её нет — просто спрашиваем в консоли при запуске (см. get_token()).
SESSION_TOKEN = os.environ.get("FOMOATER_TOKEN")

# Целевой сайт.
TARGET_URL = "https://www.fomoater.com"

# Домен, на который ставится кука.
COOKIE_DOMAIN = ".fomoater.com"

# Куда сохранить полученный HTML (для ручного просмотра вместо скриншота).
RESPONSE_HTML_PATH = "fomoater_check.html"

# Таймаут HTTP-запроса, секунд.
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------------

def get_token() -> str:
    """
    Возвращает session-токен.

    Сначала пытается взять из переменной окружения FOMOATER_TOKEN.
    Если её нет — просто спрашивает токен в консоли (можно вставить и нажать Enter).
    Так софт работает по принципу: запустил → вставил токен → он всё сделал сам.
    """
    if SESSION_TOKEN:
        return SESSION_TOKEN.strip()

    print("Вставьте session-токен и нажмите Enter.")
    print("(Где взять: DevTools → Application → Cookies → www.fomoater.com)")
    try:
        token = input("Токен: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[X] Ввод отменён.")
        return ""
    return token


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


def check_auth(response):
    """
    Проверка авторизации по HTTP-ответу.

    Возвращает кортеж (authorized: bool | None, reason: str):
      - (True,  причина)  — найден положительный признак (мы залогинены);
      - (False, причина)  — найден отрицательный признак (мы НЕ залогинены);
      - (None,  причина)  — ничего однозначного не нашли ("не удалось определить").
    """
    final_url = (response.url or "").lower()
    html = response.text or ""
    html_lower = html.lower()

    # 1) Проверяем итоговый URL: редирект на логин / OAuth X = не авторизованы.
    if "login" in final_url or "x.com/i/oauth" in final_url:
        return False, f"после запроса произошёл редирект на страницу входа: {response.url}"

    # 2) Положительные признаки — текст, который есть только у залогиненного
    #    пользователя (регистронезависимо).
    positive_patterns = [
        r"logout",
        r"sign\s*out",
        r"log\s*out",
        r"profile",
        r"my\s*card",
    ]
    for pattern in positive_patterns:
        if re.search(pattern, html_lower, re.IGNORECASE):
            return True, f"в ответе найден положительный признак авторизации: '{pattern}'"

    # 3) Отрицательные признаки — кнопки/ссылки для входа.
    negative_texts = [
        "Login with X",
        "Sign in with X",
        "Connect X",
        "Login with Twitter",
    ]
    for text in negative_texts:
        if re.search(re.escape(text), html, re.IGNORECASE):
            return False, f"в ответе видна кнопка входа: '{text}'"

    # 4) Ничего однозначного не нашли.
    return None, ("не найдено ни положительных, ни отрицательных признаков — "
                  "не удалось определить статус авторизации автоматически "
                  f"(HTTP {response.status_code}; см. сохранённый HTML)")


def save_html(response) -> None:
    """Сохраняет тело ответа в файл для ручного просмотра."""
    try:
        with open(RESPONSE_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(response.text or "")
        print(f"[i] HTML ответа сохранён: {RESPONSE_HTML_PATH}")
    except Exception as e:
        print(f"[!] Не удалось сохранить HTML: {e}")


def check_session(token: str, save_response: bool = False):
    """
    Делает один HTTP-запрос со session-кукой и проверяет авторизацию.

    Переиспользуется и основным скриптом, и почасовым монитором.

    Возвращает кортеж (status, reason, response):
      - status:  True  — авторизован;
                 False — не авторизован (токен истёк / неверный);
                 None  — не удалось определить (см. HTML/статус).
      - reason:  человекочитаемая причина;
      - response: объект ответа requests (или None при сетевой ошибке).

    При сетевых ошибках возвращает (None, "<текст ошибки>", None) —
    исключения не пробрасываются, чтобы монитор не падал в цикле.
    """
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                   "image/avif,image/webp,*/*;q=0.8"),
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        session.cookies.set(COOKIE_NAME, token, domain=COOKIE_DOMAIN, path="/")
    except Exception as e:
        return None, f"ошибка при добавлении session-куки: {e}", None

    try:
        response = session.get(
            TARGET_URL,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,  # чтобы увидеть возможный редирект на логин
        )
    except requests.exceptions.Timeout:
        return None, f"сайт не ответил за {REQUEST_TIMEOUT} секунд (таймаут)", None
    except requests.exceptions.ConnectionError as e:
        return None, f"ошибка сети / сайт недоступен: {e}", None
    except requests.exceptions.RequestException as e:
        return None, f"ошибка HTTP-запроса ({type(e).__name__}): {e}", None

    status, reason = check_auth(response)
    if save_response:
        save_html(response)
    return status, reason, response


# ---------------------------------------------------------------------------
# ОСНОВНАЯ ЛОГИКА
# ---------------------------------------------------------------------------

def main() -> int:
    # Получаем токен: из переменной окружения или спрашиваем в консоли.
    token = get_token()
    if not token:
        print("[X] Ошибка: токен не введён. Запустите софт ещё раз и вставьте токен.")
        return 1

    # Мягкая валидация — только предупреждения, работу не останавливаем.
    validate_token(token)

    # Импорт requests с понятной ошибкой, если он не установлен.
    try:
        import requests  # noqa: F401
    except ImportError:
        print("[X] Ошибка: не установлен пакет requests.")
        print("    Установите его командой:")
        print("      pip install requests")
        return 1

    # Одна проверка сессии (с сохранением HTML для ручного просмотра).
    status, reason, response = check_session(token, save_response=True)

    if response is not None:
        print(f"[i] HTTP {response.status_code}, итоговый URL: {response.url}")

    if status is True:
        print("[OK] Авторизация успешна")
        print(f"     Причина: {reason}")
    elif status is False:
        print("[FAIL] Авторизация не удалась")
        print(f"       Причина: {reason}")
    else:
        print("[?] Не удалось определить статус авторизации")
        print(f"    Причина: {reason}")

    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as e:
        # Любая прочая непредвиденная ошибка — с типом исключения.
        print(f"[X] Непредвиденная ошибка ({type(e).__name__}): {e}")
    finally:
        # Пауза, чтобы окно не закрывалось мгновенно при запуске двойным кликом.
        try:
            input("\nНажмите Enter, чтобы закрыть...")
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(exit_code)
