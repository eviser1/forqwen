"""Клиент к локальному Ollama HTTP API с поддержкой tool calling.

OllamaClient.chat(user_text, history=None) реализует цикл:
"модель -> (опционально) tool_calls -> результат каждого вызова в историю как
role='tool' -> повторный запрос модели -> ... -> финальный текст без
tool_calls возвращается вызывающему". Клиент не хранит состояние между
вызовами chat() сам по себе — ведение истории сессии между репликами
пользователя остаётся на стороне orchestrator.py (передаётся через history).

Ничего наружу не бросает: любая ошибка сети/парсинга/Ollama логируется через
logging и превращается в понятную русскую строку-отказ (_GENERIC_ERROR).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from brain.personality import get_system_prompt
from tools.registry import call_tool, get_tool_schemas

logger = logging.getLogger("jarvis.brain.ollama_client")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

_GENERIC_ERROR = (
    "Прошу прощения, служба Ollama сейчас недоступна. Проверьте, что она "
    "запущена, и повторите запрос позже."
)

_DEFAULT_DEEP_KEYWORDS = [
    "подумай как следует",
    "глубокий анализ",
    "подумай тщательно",
    "серьёзно подумай",
    "проанализируй глубоко",
]


def _load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except OSError as exc:
        logger.error("Не удалось прочитать config.yaml (%s): %s", config_path, exc)
        return {}
    except yaml.YAMLError as exc:
        logger.error("config.yaml повреждён: %s", exc)
        return {}


class OllamaClient:
    """Обёртка над /api/chat локального Ollama с обработкой tool_calls."""

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.config = config if config is not None else _load_config(config_path)
        ollama_cfg = self.config.get("ollama", {}) or {}
        assistant_cfg = self.config.get("assistant", {}) or {}

        self.base_url: str = str(ollama_cfg.get("base_url", "http://localhost:11434")).rstrip("/")
        self.model: str = ollama_cfg.get("model", "qwen2.5:7b-instruct")
        self.deep_model: str = ollama_cfg.get("deep_model", "gemma4:26b")
        self.timeout: float = float(ollama_cfg.get("timeout_seconds", 60))
        self.max_tool_iterations: int = int(ollama_cfg.get("max_tool_iterations", 5))
        self.deep_keywords: list[str] = [
            kw.lower() for kw in (ollama_cfg.get("deep_analysis_keywords") or _DEFAULT_DEEP_KEYWORDS)
        ]

        personality_enabled = assistant_cfg.get("personality_enabled", True)
        assistant_name = assistant_cfg.get("name", "Jarvis")
        self.system_prompt: str = (
            get_system_prompt(assistant_name)
            if personality_enabled
            else (
                f"Тебя зовут {assistant_name}. Отвечай кратко и по делу на русском языке. "
                "У тебя есть доступ только к явно переданным инструментам."
            )
        )

        self._tool_schemas = get_tool_schemas()

    def _select_model(self, user_text: str) -> str:
        """Эвристика "глубокого анализа": ключевые слова в тексте пользователя
        (config.ollama.deep_analysis_keywords) переключают на deep_model."""
        lowered = (user_text or "").lower()
        for keyword in self.deep_keywords:
            if keyword in lowered:
                logger.info("Запрошен глубокий анализ, используется модель %s", self.deep_model)
                return self.deep_model
        return self.model

    def chat(self, user_text: str, history: Optional[list[dict[str, Any]]] = None) -> str:
        """Отправить реплику пользователя модели и вернуть финальный текст ответа.

        Никогда не бросает исключение наружу — любая ошибка превращается
        в _GENERIC_ERROR и логируется.
        """
        try:
            return self._chat_impl(user_text, history)
        except Exception:  # последняя линия обороны
            logger.exception("Необработанная ошибка в OllamaClient.chat")
            return _GENERIC_ERROR

    def _chat_impl(self, user_text: str, history: Optional[list[dict[str, Any]]]) -> str:
        if not isinstance(user_text, str) or not user_text.strip():
            return "Вы ничего не сказали — повторите, пожалуйста."

        model = self._select_model(user_text)
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_text})

        for _ in range(self.max_tool_iterations):
            response = self._request(model, messages)
            if response is None:
                return _GENERIC_ERROR

            message = response.get("message")
            if not isinstance(message, dict):
                logger.error("Некорректный формат ответа Ollama: %r", response)
                return _GENERIC_ERROR

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
                return "Прошу прощения, не удалось сформировать ответ."

            messages.append(message)
            for call in tool_calls:
                name, arguments = self._parse_tool_call(call)
                result = call_tool(name, arguments)
                result_text = result if isinstance(result, str) else str(result)
                messages.append({"role": "tool", "name": name, "content": result_text})

        logger.warning("Превышен лимит итераций tool calling (%d)", self.max_tool_iterations)
        return "Не удалось завершить обработку запроса за отведённое число обращений к инструментам."

    @staticmethod
    def _parse_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = function.get("name", "") if isinstance(function, dict) else ""
        raw_args = function.get("arguments", {}) if isinstance(function, dict) else {}

        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                logger.warning("Не удалось разобрать JSON-аргументы tool_call %s: %r", name, raw_args)
                arguments = {}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {}

        return name, arguments

    def _request(self, model: str, messages: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        try:
            import requests
        except ImportError:
            logger.error("Пакет requests не установлен — обращение к Ollama невозможно.")
            return None

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "tools": self._tool_schemas,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("Ollama недоступна или вернула ошибку (%s): %s", self.base_url, exc)
            return None

        try:
            return resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Не удалось разобрать JSON-ответ Ollama: %s", exc)
            return None


_default_client: Optional[OllamaClient] = None


def _get_default_client() -> OllamaClient:
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client


def chat(user_text: str, history: Optional[list[dict[str, Any]]] = None) -> str:
    """Модульная точка входа: чат с Ollama через ленивый клиент по умолчанию."""
    return _get_default_client().chat(user_text, history)
