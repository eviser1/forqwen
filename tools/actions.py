"""Реализация функций белого списка инструментов Jarvis.

Каждая функция обязана валидировать свои аргументы (типы, допустимые значения,
пути) и никогда не выполнять ничего за пределами явно разрешённого — при
некорректных аргументах возвращается текст отказа на русском, исключения наружу
не пробрасываются.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("jarvis.tools.actions")

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

REFUSAL_PREFIX = "Не могу выполнить это действие: "


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except OSError as exc:
        logger.error("Не удалось прочитать config.yaml (%s): %s", _CONFIG_PATH, exc)
        return {}
    except yaml.YAMLError as exc:
        logger.error("config.yaml повреждён: %s", exc)
        return {}


_CONFIG = _load_config()
_TOOLS_CFG = _CONFIG.get("tools", {}) or {}

ALLOWED_ROOTS: list[Path] = [
    Path(p).resolve() for p in (_TOOLS_CFG.get("allowed_roots") or [])
]
ALLOWED_APPS: dict[str, str] = {
    str(k).strip().lower(): v for k, v in (_TOOLS_CFG.get("allowed_apps") or {}).items()
}

# Жёстко захардкоженный белый список безопасных dev-команд.
# ВНИМАНИЕ: сознательно не читается из config.yaml — правка конфига не должна
# иметь возможности расширить набор команд, которые модель может выполнить.
# Сверка запрошенной строки идёт РОВНО по совпадению, никакой shell=True и
# никакой сборки argv из текста модели.
SAFE_COMMANDS: dict[str, list[str]] = {
    "git status": ["git", "status"],
    "git diff": ["git", "diff"],
    "npm test": ["npm", "test"],
}

# Разрешено пользователем явно (голосом, "если я скажу"): install_package,
# install_app, restart_self. Имя пакета/программы валидируется по этой маске
# (буквы/цифры/точки/дефисы/подчёркивания/плюсы) — НИКАКИХ пробелов, `&`, `|`,
# `;`, `/`, кавычек и т.п., поэтому собрать через это произвольную shell-команду
# нельзя, даже если модель попытается передать что-то похожее на инъекцию.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,213}$")


def _is_within_allowed_roots(path: Path) -> bool:
    if not ALLOWED_ROOTS:
        return False
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in ALLOWED_ROOTS:
        if resolved == root or root in resolved.parents:
            return True
    return False


def open_app(name: str) -> str:
    """Открыть приложение из белого списка ALLOWED_APPS (config.tools.allowed_apps)."""
    if not isinstance(name, str) or not name.strip():
        return REFUSAL_PREFIX + "не указано имя приложения."
    key = name.strip().lower()
    target = ALLOWED_APPS.get(key)
    if not target:
        logger.warning("Отказ: приложение %r не входит в белый список", name)
        return REFUSAL_PREFIX + f"приложение «{name}» не входит в разрешённый список."
    try:
        subprocess.Popen([target], shell=False)
    except OSError as exc:
        logger.error("Ошибка запуска приложения %s (%s): %s", name, target, exc)
        return REFUSAL_PREFIX + f"не удалось запустить «{name}»."
    logger.info("Открыто приложение: %s (%s)", name, target)
    return f"Открываю {name}."


def open_url(url: str) -> str:
    """Открыть URL в браузере по умолчанию. Разрешены только http/https."""
    if not isinstance(url, str) or not url.strip():
        return REFUSAL_PREFIX + "не указан адрес."
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return REFUSAL_PREFIX + "разрешены только корректные адреса http/https."
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception as exc:  # webbrowser редко бросает, но подстрахуемся
        logger.error("Ошибка открытия URL %s: %s", url, exc)
        return REFUSAL_PREFIX + "не удалось открыть страницу."
    logger.info("Открыт URL: %s", url)
    return f"Открываю {url}."


def web_search(query: str) -> str:
    """Веб-поиск через DuckDuckGo HTML (без API-ключа), топ-3 результата кратко."""
    if not isinstance(query, str) or not query.strip():
        return REFUSAL_PREFIX + "пустой поисковый запрос."
    query = query.strip()[:300]
    cfg = _TOOLS_CFG.get("web_search", {}) or {}
    timeout = int(cfg.get("timeout_seconds", 10))
    max_results = int(cfg.get("max_results", 3))

    try:
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        logger.warning(
            "TODO: пакеты requests/bs4 не установлены — web_search использует "
            "резервный разбор через urllib+re (менее надёжный)."
        )
        return _web_search_fallback(query, timeout, max_results)

    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error("Ошибка веб-поиска по запросу %r: %s", query, exc)
        return "Не удалось выполнить веб-поиск: нет соединения или сервис недоступен."

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[str] = []
    for block in soup.select(".result__body")[:max_results]:
        title_el = block.select_one(".result__title")
        snippet_el = block.select_one(".result__snippet")
        title = title_el.get_text(strip=True) if title_el else ""
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        if title:
            results.append(f"{title} — {snippet}" if snippet else title)

    if not results:
        return f"По запросу «{query}» ничего не найдено."
    logger.info("Веб-поиск по запросу %r: %d результат(ов)", query, len(results))
    return f"Результаты по запросу «{query}»: " + "; ".join(results)


def _web_search_fallback(query: str, timeout: int, max_results: int) -> str:
    """Резервная реализация поиска без requests/bs4 (urllib + регулярки)."""
    try:
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=data,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # urllib поднимает разные типы исключений
        logger.error("Ошибка веб-поиска (fallback) по запросу %r: %s", query, exc)
        return "Не удалось выполнить веб-поиск: нет соединения или сервис недоступен."

    raw_titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
    titles = [re.sub("<[^>]+>", "", t).strip() for t in raw_titles[:max_results]]
    titles = [t for t in titles if t]
    if not titles:
        return f"По запросу «{query}» ничего не найдено (упрощённый разбор без bs4)."
    return f"Результаты по запросу «{query}»: " + "; ".join(titles)


def read_file(path: str) -> str:
    """Прочитать текстовый файл, только если он внутри allowed_roots."""
    if not isinstance(path, str) or not path.strip():
        return REFUSAL_PREFIX + "не указан путь к файлу."
    target = Path(path.strip())
    if not _is_within_allowed_roots(target):
        logger.warning("Отказ в чтении файла вне разрешённых директорий: %s", path)
        return REFUSAL_PREFIX + "путь находится вне разрешённых рабочих директорий."
    resolved = target.resolve()
    if not resolved.is_file():
        return REFUSAL_PREFIX + "файл не найден."
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.error("Ошибка чтения файла %s: %s", resolved, exc)
        return REFUSAL_PREFIX + "не удалось прочитать файл."
    max_len = 8000
    if len(content) > max_len:
        content = content[:max_len] + "\n...[обрезано]"
    logger.info("Прочитан файл: %s", resolved)
    return content


def write_code_file(path: str, content: str) -> str:
    """Записать текстовый файл, только если путь внутри allowed_roots."""
    if not isinstance(path, str) or not path.strip():
        return REFUSAL_PREFIX + "не указан путь к файлу."
    if not isinstance(content, str):
        return REFUSAL_PREFIX + "содержимое файла должно быть текстом."
    target = Path(path.strip())
    if not _is_within_allowed_roots(target):
        logger.warning("Отказ в записи файла вне разрешённых директорий: %s", path)
        return REFUSAL_PREFIX + "путь находится вне разрешённых рабочих директорий."
    resolved = target.resolve()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except OSError as exc:
        logger.error("Ошибка записи файла %s: %s", resolved, exc)
        return REFUSAL_PREFIX + "не удалось записать файл."
    logger.info("Записан файл: %s (%d символов)", resolved, len(content))
    return f"Файл {resolved.name} записан."


def run_safe_command(cmd: str) -> str:
    """Выполнить команду СТРОГО из захардкоженного SAFE_COMMANDS (без shell)."""
    if not isinstance(cmd, str):
        return REFUSAL_PREFIX + "команда должна быть текстом."
    normalized = cmd.strip()
    argv = SAFE_COMMANDS.get(normalized)
    if argv is None:
        logger.warning("Отказ в выполнении команды вне белого списка: %r", cmd)
        return REFUSAL_PREFIX + f"команда «{cmd}» не входит в разрешённый список."
    try:
        result = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Ошибка выполнения команды %s: %s", argv, exc)
        return REFUSAL_PREFIX + "не удалось выполнить команду."
    logger.info("Выполнена команда: %s (код возврата %d)", normalized, result.returncode)
    output = ((result.stdout or "") + (result.stderr or "")).strip()[:4000]
    return output or f"Команда «{normalized}» выполнена без вывода."


def install_package(name: str) -> str:
    """pip install <name> — только имя пакета (без версий/флагов/URL/индексов).

    Разрешено пользователем явно ("устанавливать/улучшать себя"). Риск голосового
    управления: STT может ошибиться и передать не то имя — pip в худшем случае
    установит несуществующий или не тот пакет, что обратимо (можно удалить), но
    НЕ может выполнить произвольный shell, т.к. имя жёстко провалидировано и
    subprocess вызывается без shell=True с argv-списком.
    """
    if not isinstance(name, str) or not _SAFE_NAME_RE.match(name.strip()):
        logger.warning("Отказ в установке пакета — недопустимое имя: %r", name)
        return REFUSAL_PREFIX + "недопустимое имя пакета (без пробелов/спецсимволов/версий/URL)."
    name = name.strip()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", name],
            shell=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        logger.error("Установка пакета %s не уложилась в таймаут", name)
        return f"Установка «{name}» заняла слишком много времени и была прервана."
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Ошибка установки пакета %s: %s", name, exc)
        return REFUSAL_PREFIX + "не удалось запустить установку (pip недоступен)."

    if result.returncode == 0:
        logger.info("pip install %s — успех", name)
        return f"Пакет «{name}» установлен."
    tail = (result.stderr or result.stdout or "").strip()[-400:]
    logger.warning("pip install %s — код возврата %d: %s", name, result.returncode, tail)
    return f"Не удалось установить «{name}» (код {result.returncode}): {tail or 'см. лог'}"


def install_app(name: str) -> str:
    """winget install <id/имя> — официальный менеджер пакетов Windows.

    Та же валидация имени, что и в install_package. `--accept-*-agreements`
    нужны, чтобы winget не завис на интерактивном подтверждении лицензии в
    неинтерактивном режиме (мы и так уже даём согласие явным вызовом этой
    функции по команде пользователя).
    """
    if not isinstance(name, str) or not _SAFE_NAME_RE.match(name.strip()):
        logger.warning("Отказ в установке приложения — недопустимое имя: %r", name)
        return REFUSAL_PREFIX + "недопустимое имя приложения (без пробелов/спецсимволов)."
    name = name.strip()
    try:
        result = subprocess.run(
            [
                "winget", "install", "--id", name, "-e",
                "--accept-package-agreements", "--accept-source-agreements",
                "--silent",
            ],
            shell=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        logger.error("Установка приложения %s не уложилась в таймаут", name)
        return f"Установка «{name}» заняла слишком много времени и была прервана."
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Ошибка установки приложения %s: %s", name, exc)
        return REFUSAL_PREFIX + "не удалось запустить winget."

    if result.returncode == 0:
        logger.info("winget install %s — успех", name)
        return f"Приложение «{name}» установлено."
    tail = (result.stdout or result.stderr or "").strip()[-400:]
    logger.warning("winget install %s — код возврата %d: %s", name, result.returncode, tail)
    return f"Не удалось установить «{name}» (код {result.returncode}): {tail or 'проверьте точный ID через winget search'}"


def restart_self() -> str:
    """Перезапустить процесс ассистента (после самостоятельной правки кода
    через write_code_file). Возвращает подтверждение СРАЗУ, а сам перезапуск
    происходит через 2 секунды в отдельном потоке — чтобы TTS успел озвучить
    ответ до того, как процесс завершится."""

    def _delayed_restart() -> None:
        time.sleep(2.0)
        logger.info("Перезапуск процесса Jarvis по команде пользователя.")
        try:
            subprocess.Popen([sys.executable, *sys.argv])
        finally:
            os._exit(0)

    threading.Thread(target=_delayed_restart, daemon=True).start()
    return "Перезапускаюсь, чтобы применить изменения."


def set_volume(level: int) -> str:
    """Установить системную громкость (0-100). pycaw при наличии, иначе winmm."""
    if not isinstance(level, int) or isinstance(level, bool):
        return REFUSAL_PREFIX + "уровень громкости должен быть целым числом."
    if not 0 <= level <= 100:
        return REFUSAL_PREFIX + "уровень громкости должен быть в диапазоне от 0 до 100."

    try:
        _set_volume_pycaw(level)
        logger.info("Громкость установлена через pycaw: %d%%", level)
        return f"Громкость установлена на {level}%."
    except ImportError:
        logger.info("pycaw недоступен, использую резервный способ (winmm)")
    except Exception as exc:
        logger.error("Ошибка pycaw при установке громкости: %s", exc)

    try:
        _set_volume_winmm(level)
        logger.info("Громкость установлена через winmm (приблизительно): %d%%", level)
        return f"Громкость установлена на {level}% (приблизительно)."
    except Exception as exc:
        logger.error("Не удалось установить громкость: %s", exc)
        return REFUSAL_PREFIX + "не удалось изменить громкость на этом устройстве."


def _set_volume_pycaw(level: int) -> None:
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL  # type: ignore
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevelScalar(level / 100.0, None)


def _set_volume_winmm(level: int) -> None:
    """Резервный вариант на чистом stdlib (ctypes + winmm.dll), без доп. пакетов."""
    scaled = int(level / 100 * 0xFFFF)
    value = (scaled & 0xFFFF) | (scaled << 16)
    winmm = ctypes.windll.winmm  # type: ignore[attr-defined]
    winmm.waveOutSetVolume(0, value)


def lock_screen() -> str:
    """Заблокировать экран (Windows API LockWorkStation)."""
    try:
        result = ctypes.windll.user32.LockWorkStation()  # type: ignore[attr-defined]
    except Exception as exc:
        logger.error("Ошибка блокировки экрана: %s", exc)
        return REFUSAL_PREFIX + "не удалось заблокировать экран."
    if not result:
        return REFUSAL_PREFIX + "не удалось заблокировать экран."
    logger.info("Экран заблокирован")
    return "Экран заблокирован."


def open_explorer(path: Optional[str] = None) -> str:
    """Открыть Проводник; при указанном path — только если он внутри allowed_roots."""
    if path is None or (isinstance(path, str) and not path.strip()):
        try:
            subprocess.Popen(["explorer.exe"], shell=False)
        except OSError as exc:
            logger.error("Ошибка открытия проводника: %s", exc)
            return REFUSAL_PREFIX + "не удалось открыть проводник."
        logger.info("Открыт проводник (без пути)")
        return "Открываю проводник."

    if not isinstance(path, str):
        return REFUSAL_PREFIX + "путь должен быть текстом."
    target = Path(path.strip())
    if not _is_within_allowed_roots(target):
        logger.warning("Отказ в открытии проводника вне разрешённых директорий: %s", path)
        return REFUSAL_PREFIX + "путь находится вне разрешённых рабочих директорий."
    resolved = target.resolve()
    if not resolved.exists():
        return REFUSAL_PREFIX + "указанный путь не существует."
    try:
        subprocess.Popen(["explorer.exe", str(resolved)], shell=False)
    except OSError as exc:
        logger.error("Ошибка открытия проводника по пути %s: %s", resolved, exc)
        return REFUSAL_PREFIX + "не удалось открыть проводник по этому пути."
    logger.info("Открыт проводник по пути: %s", resolved)
    return f"Открываю проводник: {resolved}."
