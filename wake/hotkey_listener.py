"""Глобальный хоткей (по умолчанию Numpad "+") как альтернативный способ
активировать Jarvis — вместо или вместе с голосовым wake word "hey jarvis".

Полезно как запасной вариант, пока голосовой wake word не настроен/не
работает (например, из-за отсутствующих зависимостей или проблем с
микрофоном) — можно активировать ассистента одной клавишей.

Публичный интерфейс зеркалит wake/wakeword_listener.py::WakeWordListener,
чтобы orchestrator.py мог использовать оба источника активации одинаково:
    HotkeyListener(on_wake, on_ready=...).start()/.stop()
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

# "add" — имя клавиши Numpad "+" в библиотеке keyboard (проверено по её
# документации/маппингу клавиш). При необходимости можно переопределить в
# config.yaml -> hotkey.key на любую другую клавишу, которую понимает keyboard
# (например "f9", "ctrl+j" и т.п.).
DEFAULT_HOTKEY = "add"


def _load_yaml_config() -> dict:
    """Best-effort load of config/config.yaml. Never raises."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        import yaml  # PyYAML — опциональная зависимость этого модуля
    except ImportError:
        logger.warning(
            "PyYAML не установлен — config.yaml игнорируется, "
            "используются значения по умолчанию."
        )
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        logger.exception(
            "Не удалось прочитать %s — используются значения по умолчанию.",
            _CONFIG_PATH,
        )
        return {}


class HotkeyListener:
    """Слушает глобальный хоткей (по умолчанию Numpad "+") и вызывает on_wake."""

    def __init__(
        self,
        on_wake: Callable[[], None],
        hotkey: Optional[str] = None,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Args:
            on_wake: вызывается (без аргументов) при каждом нажатии хоткея.
                Исключения внутри логируются и проглатываются — прослушивание
                хоткея продолжается.
            hotkey: имя клавиши в формате библиотеки `keyboard` (по умолчанию
                "add" — Numpad "+"). Берётся из config.yaml -> hotkey.key,
                если не передано явно.
            on_ready: вызывается один раз сразу после успешной регистрации
                хоткея — доказательство, что перехват клавиатуры реально
                заработал (а не тихо упал в фоновом потоке).
        """
        self._on_wake = on_wake
        self._on_ready = on_ready
        cfg = (_load_yaml_config().get("hotkey") or {})
        self._hotkey = hotkey or cfg.get("key", DEFAULT_HOTKEY)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._registered = False

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def start(self) -> None:
        """Запустить прослушивание хоткея в фоновом daemon-потоке (неблокирующе)."""
        if self._thread and self._thread.is_alive():
            logger.warning("HotkeyListener уже запущен.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name="hotkey-listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Остановить прослушивание и дождаться завершения потока."""
        self._stop_event.set()
        if self._registered:
            try:
                import keyboard  # type: ignore

                keyboard.remove_hotkey(self._hotkey)
            except Exception:
                logger.debug("Не удалось снять регистрацию хоткея (не критично).", exc_info=True)
        if self._thread:
            self._thread.join(timeout=5.0)

    def run(self) -> None:
        """Блокирующий цикл: регистрирует хоткей и ждёт stop()/stop_event.

        Поднимает ImportError/RuntimeError при невозможности перехватить
        клавиатуру (нет пакета `keyboard`, недостаточно прав) — orchestrator.py
        ловит это так же, как и для WakeWordListener.
        """
        try:
            import keyboard  # type: ignore
        except ImportError:
            logger.exception(
                "Пакет keyboard не установлен. Установите: pip install keyboard"
            )
            raise

        def _on_press() -> None:
            logger.info("Хоткей '%s' нажат.", self._hotkey)
            try:
                self._on_wake()
            except Exception:
                logger.exception("Ошибка в обработчике on_wake() — продолжаю слушать хоткей.")

        try:
            keyboard.add_hotkey(self._hotkey, _on_press)
            self._registered = True
        except Exception:
            logger.exception(
                "Не удалось зарегистрировать глобальный хоткей '%s' "
                "(возможно, нужны права администратора).",
                self._hotkey,
            )
            raise

        logger.info(
            "Hotkey listener запущен. Нажмите '%s' (по умолчанию Numpad +), "
            "чтобы активировать Jarvis без голоса.",
            self._hotkey,
        )
        if self._on_ready is not None:
            try:
                self._on_ready()
            except Exception:
                logger.exception("Ошибка в обработчике on_ready() — продолжаю слушать хоткей.")

        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)
        finally:
            if self._registered:
                try:
                    keyboard.remove_hotkey(self._hotkey)
                except Exception:
                    pass
            logger.info("Hotkey listener остановлен.")
