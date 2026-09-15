"""
Трей-иконка голосового ассистента Jarvis.

Публичный интерфейс, рассчитанный на использование из orchestrator.py:

    from service import tray

    tray.start(on_exit=my_shutdown_callback)   # один раз при старте оркестратора
    tray.set_state("idle")                      # сон / ожидание wake word
    tray.set_state("listening")                 # идёт запись команды
    tray.set_state("thinking")                  # LLM/инструменты обрабатывают запрос
    ...
    tray.stop()                                 # при штатном завершении оркестратора

Состояния и цвета:
    idle      — серый   — сон, ожидание кодового слова "Jarvis"
    listening — зелёный — слушает команду пользователя
    thinking  — жёлтый  — думает / выполняет действие

Пункты меню (правая кнопка мыши по иконке):
    "Выключить"    — останавливает процесс ассистента.
                     Если оркестратор передал on_exit в tray.start(), вызывается он
                     (ожидается корректное завершение потоков/ресурсов оркестратора);
                     иначе процесс завершается немедленно (os._exit).
    "Перезапустить" — запускает новый процесс с теми же аргументами командной строки
                      и завершает текущий.

Иконки генерируются программно через Pillow — внешних файлов не требуется.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from typing import Callable, Literal, Optional

from PIL import Image, ImageDraw
import pystray

State = Literal["idle", "listening", "thinking"]

_LOG = logging.getLogger("jarvis.tray")

_ICON_SIZE = 64
_CIRCLE_MARGIN = 6

_COLORS: dict[State, tuple[int, int, int]] = {
    "idle": (128, 128, 128),        # серый — сон
    "listening": (39, 174, 96),     # зелёный — слушаю
    "thinking": (241, 196, 15),     # жёлтый — думаю
}

_TITLES: dict[State, str] = {
    "idle": "Jarvis — сон",
    "listening": "Jarvis — слушаю",
    "thinking": "Jarvis — думаю",
}

_icon: Optional["pystray.Icon"] = None
_icon_lock = threading.Lock()
_on_exit: Optional[Callable[[], None]] = None
_current_state: State = "idle"


def _make_image(color: tuple[int, int, int]) -> Image.Image:
    """Генерирует квадратное изображение с закрашенным кругом заданного цвета."""
    image = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    bbox = (
        _CIRCLE_MARGIN,
        _CIRCLE_MARGIN,
        _ICON_SIZE - _CIRCLE_MARGIN,
        _ICON_SIZE - _CIRCLE_MARGIN,
    )
    draw.ellipse(bbox, fill=(*color, 255), outline=(30, 30, 30, 255), width=2)
    return image


def _handle_exit(icon: "pystray.Icon", item: object) -> None:
    _LOG.info("Трей: получена команда 'Выключить'")
    callback = _on_exit
    icon.stop()
    if callback is not None:
        try:
            callback()
        except Exception:
            _LOG.exception("Ошибка в on_exit-колбэке оркестратора")
    else:
        os._exit(0)


def _handle_restart(icon: "pystray.Icon", item: object) -> None:
    _LOG.info("Трей: получена команда 'Перезапустить'")
    icon.stop()
    try:
        subprocess.Popen([sys.executable, *sys.argv])
    finally:
        os._exit(0)


def _build_menu() -> "pystray.Menu":
    return pystray.Menu(
        pystray.MenuItem("Выключить", _handle_exit),
        pystray.MenuItem("Перезапустить", _handle_restart),
    )


def start(on_exit: Optional[Callable[[], None]] = None) -> None:
    """Создаёт и запускает трей-иконку в фоновом потоке.

    on_exit: колбэк без аргументов, вызываемый при выборе пункта "Выключить"
    (например, для корректной остановки потоков wake/stt/tts перед выходом).
    Если не передан — процесс завершается немедленно.

    Идемпотентно: повторный вызов при уже запущенной иконке ничего не делает.
    """
    global _icon, _on_exit

    with _icon_lock:
        if _icon is not None:
            _LOG.debug("tray.start() вызван повторно — иконка уже запущена")
            return

        _on_exit = on_exit
        icon = pystray.Icon(
            name="jarvis",
            icon=_make_image(_COLORS["idle"]),
            title=_TITLES["idle"],
            menu=_build_menu(),
        )
        _icon = icon

    thread = threading.Thread(target=icon.run, name="jarvis-tray", daemon=True)
    thread.start()
    _LOG.info("Трей-иконка запущена")


def set_state(state: State) -> None:
    """Обновляет цвет/подсказку иконки. Потокобезопасно."""
    global _current_state

    if state not in _COLORS:
        raise ValueError(f"Неизвестное состояние трея: {state!r}")

    with _icon_lock:
        _current_state = state
        if _icon is None:
            _LOG.debug("set_state(%s) вызван до tray.start() — состояние запомнено", state)
            return
        _icon.icon = _make_image(_COLORS[state])
        _icon.title = _TITLES[state]


def get_state() -> State:
    """Возвращает текущее состояние (полезно для тестов/диагностики)."""
    return _current_state


def stop() -> None:
    """Останавливает иконку без завершения процесса (для штатного shutdown)."""
    with _icon_lock:
        if _icon is not None:
            _icon.stop()


if __name__ == "__main__":
    # Ручная проверка: python -m service.tray
    logging.basicConfig(level=logging.INFO)
    import time

    start()
    for demo_state in ("idle", "listening", "thinking", "idle"):
        set_state(demo_state)  # type: ignore[arg-type]
        time.sleep(3)
