"""Orchestrator: склеивает wake word -> STT -> brain (Ollama+tools) -> TTS в один
рабочий цикл голосового ассистента Jarvis, плюс трей-иконку и логирование.

Режимы запуска:
    python orchestrator.py            Полный голосовой режим (нужны все ML-зависимости
                                       из requirements.txt: openwakeword, faster-whisper,
                                       torch/Silero, pystray, keyboard). Активировать можно
                                       голосом ("hey jarvis"), хоткеем (Numpad "+") ИЛИ
                                       просто напечатав команду в этой же консоли и нажав
                                       Enter (см. stdin_fallback_reader) — все три способа
                                       работают одновременно, это не отдельный режим.
    python orchestrator.py --text     Текстовый fallback-режим: читает команды из stdin,
                                       печатает ответ в stdout. Не требует ни микрофона,
                                       ни openwakeword/faster-whisper/torch/pystray/keyboard —
                                       использует только brain/tools (requests, pyyaml), нет
                                       ни трея, ни wake word, ни хоткея. Полезно для проверки
                                       без установки тяжёлых аудио-зависимостей вообще.

Barge-in: пока идёт озвучка ответа (tts.speak), повторное срабатывание wake word не
запускает новую команду поверх текущей, а немедленно останавливает воспроизведение
(interrupt_event) — после чего пользователь может сразу говорить новую команду.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from brain.ollama_client import OllamaClient  # noqa: E402

logger = logging.getLogger("jarvis.orchestrator")

MAX_HISTORY_MESSAGES = 12  # последние ~6 пар реплик — сессионная память без диска
NO_SPEECH_REPLY = "Прошу прощения, я не расслышал. Повторите, пожалуйста."


def _setup_logging() -> None:
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "action.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def _trim_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return history[-MAX_HISTORY_MESSAGES:]


def _force_utf8_console() -> None:
    """Консоль Windows по умолчанию не в UTF-8 (обычно cp1251/cp866), из-за
    чего кириллица на вводе/выводе портится (проверено вживую: искажённый
    ввод реально уходил в модель и ломал ответы). Принудительно переключаем
    stdin/stdout/stderr на UTF-8, если Python это поддерживает."""
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                logger.warning("Не удалось переключить %s на UTF-8.", stream_name)


def run_text_mode() -> None:
    """Fallback: диалог через stdin/stdout, без аудио и без openwakeword/whisper/torch."""
    _force_utf8_console()
    client = OllamaClient()
    history: list[dict[str, Any]] = []
    print("Jarvis (текстовый режим). Пустая строка или Ctrl+C — выход.")
    while True:
        try:
            text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        logger.info("USER: %s", text)
        response = client.chat(text, history)
        logger.info("JARVIS: %s", response)
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response})
        history[:] = _trim_history(history)
        print(f"Jarvis: {response}")


def run_voice_mode() -> None:
    """Полный цикл: wake word -> запись -> STT -> brain -> TTS, с barge-in и треем."""
    # Импорты тяжёлых ML-модулей только здесь, чтобы --text работал без них.
    from service import tray
    from stt.transcriber import record_until_silence, transcribe
    from tts.speaker import play_ready_chime, play_wake_chime, speak
    from wake.hotkey_listener import HotkeyListener
    from wake.wakeword_listener import WakeWordListener

    client = OllamaClient()
    history: list[dict[str, Any]] = []

    stop_event = threading.Event()
    interrupt_event = threading.Event()
    speaking_event = threading.Event()
    # None в очереди = "сработал голос/хоткей, нужно ЕЩЁ записать и распознать
    # речь"; строка = "текст команды уже известен" (пришёл из stdin-fallback'а
    # ниже) — записывать/распознавать ничего не нужно, сразу в brain.
    trigger_queue: "queue.Queue[Optional[str]]" = queue.Queue()

    def on_wake() -> None:
        # Колбэк выполняется синхронно в потоке wake-listener'а — должен быть
        # быстрым и не блокировать дальнейшее прослушивание. Пока идёт озвучка
        # ответа — это barge-in (прерываем речь), иначе — обычный новый запрос.
        if speaking_event.is_set():
            logger.info("Barge-in: повторное wake word во время озвучки — прерываю.")
            interrupt_event.set()
        else:
            trigger_queue.put(None)

    def stdin_fallback_reader() -> None:
        # Запасной способ отдать команду, если голос/хоткей почему-то не
        # срабатывают: печатаешь текст в ту же консоль и жмёшь Enter. Работает
        # ОДНОВРЕМЕННО с голосом и хоткеем, без переключения режима (в отличие
        # от `orchestrator.py --text`, у которого нет ни wake word, ни трея).
        if sys.stdin is None or not sys.stdin.readable():
            logger.info(
                "stdin недоступен (например, запуск как скрытая служба без консоли) — "
                "текстовый fallback-ввод отключён, доступны только голос/хоткей."
            )
            return
        logger.info(
            "Текстовый fallback-ввод активен: можно просто напечатать команду "
            "в этой консоли и нажать Enter — не обязательно говорить или жать Numpad +."
        )
        while not stop_event.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt, OSError):
                break
            text = line.strip()
            if not text:
                continue
            if speaking_event.is_set():
                logger.info("Barge-in: текстовый ввод во время озвучки — прерываю.")
                interrupt_event.set()
            trigger_queue.put(text)

    stdin_thread = threading.Thread(
        target=stdin_fallback_reader, name="stdin-fallback", daemon=True
    )
    stdin_thread.start()

    def on_wake_ready() -> None:
        # Подтверждение (звук + лог), что модель загружена И микрофон реально
        # открылся — без этого молчаливый сбой в фоновом потоке (нет пакета,
        # не скачались веса модели, нет микрофона) выглядит как "не реагирует",
        # хотя на деле поток вообще не запустился.
        logger.info("Jarvis слушает голос — микрофон и модель wake word готовы.")
        play_ready_chime()

    def on_hotkey_ready() -> None:
        # Отдельного звука не даём (чтобы не путать с голосовым ready-чимом) —
        # достаточно строки в логе для диагностики "запустился ли хоткей".
        logger.info("Hotkey Numpad '+' готов — можно жать вместо голоса.")

    listener = WakeWordListener(on_wake=on_wake, on_ready=on_wake_ready)
    hotkey = HotkeyListener(on_wake=on_wake, on_ready=on_hotkey_ready)

    def shutdown() -> None:
        logger.info("Завершение работы Jarvis...")
        stop_event.set()
        listener.stop()
        hotkey.stop()
        tray.stop()

    tray.start(on_exit=shutdown)
    listener.start()
    hotkey.start()
    logger.info("Jarvis запущен, жду подтверждения от wake-listener'ов...")

    # Даём потокам время загрузить модель/открыть микрофон/перехватить
    # клавиатуру, затем проверяем, что они реально живы — если поток упал
    # (ImportError/RuntimeError/OSError внутри run()), он тихо умирает без
    # видимого краша всего процесса, и без этой проверки это выглядело бы
    # как "просто не реагирует".
    threading.Event().wait(3.0)
    if not listener._thread or not listener._thread.is_alive():
        logger.error(
            "Wake word listener не запустился (поток умер) — Jarvis НЕ слышит "
            "голос и не будет реагировать на «Хей, Джарвис». Смотрите строку с "
            "ошибкой чуть выше в этом же логе (обычно: не установлен пакет "
            "openwakeword/sounddevice, не скачались веса модели — нужен "
            "интернет при первом запуске, — или не найден микрофон)."
        )
    if not hotkey._thread or not hotkey._thread.is_alive():
        logger.error(
            "Hotkey listener не запустился — Numpad '+' не будет работать. "
            "Смотрите строку с ошибкой чуть выше (обычно: не установлен пакет "
            "keyboard, либо нужны права администратора для перехвата клавиатуры)."
        )

    try:
        while not stop_event.is_set():
            try:
                trigger = trigger_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            tray.set_state("listening")
            if trigger is None:
                # Сработал голос или хоткей — ещё нужно записать и распознать речь.
                play_wake_chime()
                audio = record_until_silence()
                text = transcribe(audio, language="ru")
            else:
                # Команда уже пришла текстом (stdin fallback) — распознавать нечего.
                text = trigger

            if not text:
                logger.info("Пустая расшифровка — прошу повторить.")
                speaking_event.set()
                interrupt_event.clear()
                speak(NO_SPEECH_REPLY, interrupt_event)
                speaking_event.clear()
                tray.set_state("idle")
                continue

            logger.info("USER: %s", text)
            tray.set_state("thinking")
            response = client.chat(text, history)
            logger.info("JARVIS: %s", response)

            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": response})
            history[:] = _trim_history(history)

            interrupt_event.clear()
            speaking_event.set()
            speak(response, interrupt_event)
            speaking_event.clear()
            tray.set_state("idle")
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis — локальный голосовой ассистент")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Текстовый fallback-режим (без микрофона/openwakeword/whisper/torch/трея).",
    )
    args = parser.parse_args()

    _force_utf8_console()
    _setup_logging()

    if args.text:
        run_text_mode()
    else:
        run_voice_mode()


if __name__ == "__main__":
    main()
