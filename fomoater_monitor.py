#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fomoater_monitor.py

Почасовой монитор session-токена для ОДНОГО ЛИЧНОГО аккаунта на fomoater.com.

Раз в час проверяет, не истёк ли твой session-токен, и пишет статус в консоль
с отметкой времени. Как только токен перестаёт быть валидным (истёк / разлогин),
громко сообщает об этом.

Использует ту же логику проверки, что и fomoater_check.py (функция check_session),
поэтому оба файла должны лежать рядом.

Никакого фарма, мультиаккаунтинга и автоматизации заданий — только наблюдение
за собственной сессией.
"""

import sys
import time
from datetime import datetime

# Переиспользуем логику из основного скрипта.
try:
    from fomoater_check import get_token, validate_token, check_session
except ImportError:
    print("[X] Ошибка: не найден файл fomoater_check.py рядом с монитором.")
    print("    Положите fomoater_monitor.py в ту же папку, что и fomoater_check.py.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# КОНФИГ
# ---------------------------------------------------------------------------

# Интервал проверки в секундах. По умолчанию 1 час.
CHECK_INTERVAL_SECONDS = 3600

# Через сколько секунд повторить попытку, если была сетевая ошибка
# (сайт недоступен / таймаут) — чтобы не ждать целый час из-за разовой ошибки сети.
RETRY_AFTER_ERROR_SECONDS = 300


def now_str() -> str:
    """Текущее время в читаемом виде для лога."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_monitor(token: str) -> int:
    """Бесконечный цикл почасовой проверки токена. Ctrl+C для выхода."""
    print(f"[{now_str()}] Монитор запущен. "
          f"Проверка каждые {CHECK_INTERVAL_SECONDS // 60} мин. "
          f"Ctrl+C — выход.")

    while True:
        status, reason, _response = check_session(token)

        if status is True:
            print(f"[{now_str()}] [OK] Токен действителен — вы авторизованы.")
            sleep_for = CHECK_INTERVAL_SECONDS
        elif status is False:
            # Главное событие ради которого всё затевалось: токен истёк.
            print(f"[{now_str()}] [!!!] ТОКЕН ИСТЁК или недействителен — "
                  f"вы больше не авторизованы.")
            print(f"           Причина: {reason}")
            print(f"           Войдите на сайт заново и обновите токен "
                  f"(перезапустите монитор с новым токеном).")
            sleep_for = CHECK_INTERVAL_SECONDS
        else:
            # Не удалось определить: чаще всего сетевая ошибка или нестандартная вёрстка.
            print(f"[{now_str()}] [?] Не удалось проверить токен.")
            print(f"           Причина: {reason}")
            print(f"           Повторю попытку через "
                  f"{RETRY_AFTER_ERROR_SECONDS // 60} мин.")
            sleep_for = RETRY_AFTER_ERROR_SECONDS

        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            print(f"\n[{now_str()}] Монитор остановлен пользователем.")
            return 0


def main() -> int:
    # Токен: из переменной окружения FOMOATER_TOKEN или ввод в консоли.
    token = get_token()
    if not token:
        print("[X] Ошибка: токен не введён. Запустите монитор ещё раз и вставьте токен.")
        return 1

    validate_token(token)

    # Проверим, что requests установлен (check_session импортирует его внутри).
    try:
        import requests  # noqa: F401
    except ImportError:
        print("[X] Ошибка: не установлен пакет requests.")
        print("    Установите его командой:  pip install requests")
        return 1

    return run_monitor(token)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nМонитор остановлен пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"[X] Непредвиденная ошибка ({type(e).__name__}): {e}")
        sys.exit(1)
