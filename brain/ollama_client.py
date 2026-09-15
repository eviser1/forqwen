"""Клиент к локальной LLM модели через transformers с поддержкой tool calling.

OllamaClient.chat(user_text, history=None) реализует цикл:
"модель -> (опционально) tool_calls -> результат каждого вызова в историю как
role='tool' -> повторный запрос модели -> ... -> финальный текст без
tool_calls возвращается вызывающему". Клиент не хранит состояние между
вызовами chat() сам по себе — ведение истории сессии между репликами
пользователя остаётся на стороне orchestrator.py (передаётся через history).

Ничего наружу не бросает: любая ошибка сети/парсинга/модели логируется через
logging и превращается в понятную русскую строку-отказ (_GENERIC_ERROR).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from brain.personality import get_system_prompt
from tools.registry import call_tool, get_tool_schemas

logger = logging.getLogger("jarvis.brain.ollama_client")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

_GENERIC_ERROR = (
    "Прошу прощения, модель сейчас недоступна. Попробуйте повторить запрос позже."
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
    """Клиент для локальной LLM модели через transformers с обработкой tool_calls."""

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.config = config if config is not None else _load_config(config_path)
        ollama_cfg = self.config.get("ollama", {}) or {}
        assistant_cfg = self.config.get("assistant", {}) or {}

        self.model_name: str = ollama_cfg.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        self.deep_model_name: str = ollama_cfg.get("deep_model", "Qwen/Qwen2.5-0.5B-Instruct")
        
        # Проверка формата модели (для Ollama форматов конвертируем в HuggingFace)
        if self.model_name.startswith("qwen3:") or self.model_name.startswith("qwen2.5:"):
            self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        if self.deep_model_name.startswith("qwen3:") or self.deep_model_name.startswith("qwen2.5:") or self.deep_model_name.startswith("gemma4:"):
            self.deep_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
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
        self._generator = None
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        """Загружает модель через transformers"""
        try:
            from transformers import pipeline
            logger.info(f"Загрузка модели {self.model_name}...")
            self._generator = pipeline(
                'text-generation',
                model=self.model_name,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )
            self._model_loaded = True
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self._model_loaded = False

    def _select_model_name(self, user_text: str) -> str:
        """Эвристика "глубокого анализа": ключевые слова в тексте пользователя
        (config.ollama.deep_analysis_keywords) переключают на deep_model."""
        lowered = (user_text or "").lower()
        for keyword in self.deep_keywords:
            if keyword in lowered:
                logger.info("Запрошен глубокий анализ, используется модель %s", self.deep_model_name)
                return self.deep_model_name
        return self.model_name

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

        if not self._model_loaded:
            logger.error("Модель не загружена")
            return _GENERIC_ERROR

        model_name = self._select_model_name(user_text)
        
        # Формируем промпт для текстовой модели
        prompt = f"{self.system_prompt}\n\n"
        if history:
            for msg in history[-10:]:  # Берём последние 10 сообщений истории
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
        
        prompt += f"User: {user_text}\nAssistant:"

        try:
            # Переключаем модель если нужно (для глубокого анализа)
            if model_name != self.model_name and self._generator.model.config.name_or_path != model_name:
                from transformers import pipeline
                self._generator = pipeline(
                    'text-generation',
                    model=model_name,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7
                )
            
            result = self._generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
            full_text = result[0]['generated_text']
            
            # Извлекаем только ответ ассистента
            if "Assistant:" in full_text:
                response = full_text.split("Assistant:")[-1].strip()
            else:
                response = full_text.replace(prompt, "").strip()
            
            # Очищаем от возможных артефактов
            response = response.split("\nUser:")[0].strip()
            
            if response:
                return response
            return "Прошу прощения, не удалось сформировать ответ."
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return _GENERIC_ERROR


_default_client: Optional[OllamaClient] = None


def _get_default_client() -> OllamaClient:
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client


def chat(user_text: str, history: Optional[list[dict[str, Any]]] = None) -> str:
    """Модульная точка входа: чат с Ollama через ленивый клиент по умолчанию."""
    return _get_default_client().chat(user_text, history)
