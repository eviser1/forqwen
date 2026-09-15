"""Реестр белого списка функций, доступных модели Ollama через tool calling.

Единственная точка входа для выполнения инструмента — call_tool(). Она ищет
имя СТРОГО в TOOL_REGISTRY; любое имя, отсутствующее в реестре, приводит
к отказу без какого-либо выполнения.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from tools import actions

logger = logging.getLogger("jarvis.tools.registry")

REFUSAL_UNKNOWN_TOOL = "Отказ: запрошенный инструмент не входит в разрешённый список."

ToolHandler = Callable[..., str]

TOOL_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "open_app",
        "description": "Открыть разрешённое приложение по имени из белого списка (config.tools.allowed_apps).",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Имя приложения, например 'notepad' или 'блокнот'.",
                }
            },
            "required": ["name"],
        },
        "handler": actions.open_app,
    },
    {
        "name": "open_url",
        "description": "Открыть веб-страницу в браузере по умолчанию. Разрешены только адреса http/https.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Полный адрес страницы, например 'https://example.com'.",
                }
            },
            "required": ["url"],
        },
        "handler": actions.open_url,
    },
    {
        "name": "web_search",
        "description": "Найти информацию в интернете через DuckDuckGo и вернуть краткую сводку топ-3 результатов.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос на русском или английском.",
                }
            },
            "required": ["query"],
        },
        "handler": actions.web_search,
    },
    {
        "name": "read_file",
        "description": "Прочитать текстовый файл. Доступ только к файлам внутри разрешённых рабочих директорий проекта.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Абсолютный или относительный путь к файлу внутри разрешённых директорий.",
                }
            },
            "required": ["path"],
        },
        "handler": actions.read_file,
    },
    {
        "name": "write_code_file",
        "description": "Записать текстовый/код-файл. Доступ только внутри разрешённых рабочих директорий проекта.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к файлу внутри разрешённых директорий.",
                },
                "content": {
                    "type": "string",
                    "description": "Полное содержимое файла для записи.",
                },
            },
            "required": ["path", "content"],
        },
        "handler": actions.write_code_file,
    },
    {
        "name": "run_safe_command",
        "description": (
            "Выполнить одну из заранее одобренных dev-команд: 'git status', "
            "'git diff' или 'npm test'. Никакие другие команды не выполняются."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "Точная строка команды из разрешённого списка.",
                    "enum": ["git status", "git diff", "npm test"],
                }
            },
            "required": ["cmd"],
        },
        "handler": actions.run_safe_command,
    },
    {
        "name": "install_package",
        "description": (
            "Установить Python-пакет через pip install. Разрешено пользователем явно "
            "('устанавливать/улучшать себя'). Только имя пакета — без версий, флагов, URL."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Имя пакета PyPI, например 'requests' (без версии/флагов).",
                }
            },
            "required": ["name"],
        },
        "handler": actions.install_package,
    },
    {
        "name": "install_app",
        "description": (
            "Установить программу через winget (официальный менеджер пакетов Windows). "
            "Только имя/ID пакета winget — без версий, флагов, URL."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "ID пакета winget, например 'Git.Git' или 'Notepad++.Notepad++'.",
                }
            },
            "required": ["name"],
        },
        "handler": actions.install_app,
    },
    {
        "name": "restart_self",
        "description": (
            "Перезапустить процесс ассистента Jarvis — использовать после того, как "
            "пользователь попросил изменить/улучшить собственный код ассистента "
            "(через write_code_file), чтобы изменения вступили в силу."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "handler": actions.restart_self,
    },
    {
        "name": "set_volume",
        "description": "Установить громкость системы в процентах (0-100).",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Целевой уровень громкости от 0 до 100.",
                }
            },
            "required": ["level"],
        },
        "handler": actions.set_volume,
    },
    {
        "name": "lock_screen",
        "description": "Заблокировать экран компьютера.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "handler": actions.lock_screen,
    },
    {
        "name": "open_explorer",
        "description": "Открыть Проводник Windows, опционально сразу на указанном пути (внутри разрешённых директорий).",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": ["string", "null"],
                    "description": "Путь для открытия в Проводнике или null для открытия без пути.",
                }
            },
            "required": [],
        },
        "handler": actions.open_explorer,
    },
]

_REGISTRY_BY_NAME: dict[str, dict[str, Any]] = {tool["name"]: tool for tool in TOOL_REGISTRY}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Схемы инструментов в формате, ожидаемом полем "tools" Ollama chat API."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters_json_schema"],
            },
        }
        for tool in TOOL_REGISTRY
    ]


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Выполнить инструмент строго по имени из реестра.

    Если имени нет в реестре или аргументы некорректны — возвращает текст
    отказа на русском и ничего не выполняет. Любое исключение из handler'а
    перехватывается здесь и превращается в текстовый ответ, наружу не летит.
    """
    if not isinstance(name, str) or name not in _REGISTRY_BY_NAME:
        logger.warning("Запрошен неизвестный/запрещённый инструмент: %r", name)
        return REFUSAL_UNKNOWN_TOOL

    if not isinstance(arguments, dict):
        logger.warning("Некорректные аргументы (не dict) для инструмента %s: %r", name, arguments)
        return f"Отказ: некорректные аргументы для инструмента «{name}»."

    tool = _REGISTRY_BY_NAME[name]
    handler: ToolHandler = tool["handler"]

    # Защита в глубину: передаём handler'у только те ключи, что заявлены
    # в схеме параметров, даже если модель прислала лишние поля.
    allowed_params = set(tool["parameters_json_schema"].get("properties", {}).keys())
    filtered_args = {k: v for k, v in arguments.items() if k in allowed_params}

    try:
        result = handler(**filtered_args)
    except TypeError as exc:
        logger.error("Некорректные типы/состав аргументов для %s: %s", name, exc)
        return f"Отказ: некорректные аргументы для инструмента «{name}»."
    except Exception:  # последняя линия обороны — наружу исключение не идёт
        logger.exception("Необработанная ошибка при вызове инструмента %s", name)
        return f"Произошла ошибка при выполнении «{name}»."

    return result if isinstance(result, str) else str(result)
