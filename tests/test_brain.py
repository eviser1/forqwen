# -*- coding: utf-8 -*-
"""
Юнит-тесты для brain/ollama_client.py.

brain/ollama_client.py к моменту написания этого файла уже появился на диске
(backend-agent писал параллельно) — тесты ниже сверены с РЕАЛЬНЫМ кодом:

    class OllamaClient:
        def __init__(self, config: dict | None = None, config_path: Path = ...)
        def chat(self, user_text: str, history: list[dict] | None = None) -> str
            - никогда не бросает исключение наружу (обёрнуто в try/except Exception
              с логированием и возвратом _GENERIC_ERROR)
            - пустой user_text -> "Вы ничего не сказали — повторите, пожалуйста."
              БЕЗ обращения к сети
            - выбор модели: self._select_model(user_text) ищет вхождение любого
              слова из self.deep_keywords (по умолчанию включает
              "подумай как следует") в lowered user_text; если нашлось —
              используется self.deep_model (по умолчанию "gemma4:26b" — как
              того требует spec.md), иначе self.model
            - цикл: пока в ответе Ollama есть message["tool_calls"] — для
              каждого вызывается tools.registry.call_tool(name, arguments)
              (импортирован в модуль как `call_tool`), результат (строка)
              добавляется в историю как {"role": "tool", "name":.., "content":..}
              и модель запрашивается повторно; при отсутствии tool_calls
              возвращается message["content"]
            - лимит self.max_tool_iterations — если исчерпан, возвращается
              фиксированная строка про превышение числа обращений к инструментам
            - requests импортируется ЛОКАЛЬНО внутри _request(), поэтому мокать
              нужно глобальный `requests.post`, а не atribut на модуле
              ollama_client
            - requests.exceptions.RequestException (включая ConnectionError,
              Timeout, HTTPError через raise_for_status) -> _request()
              возвращает None -> chat() возвращает _GENERIC_ERROR (константа
              модуля, текст на русском, объясняющий недоступность Ollama)
            - невалидный JSON в ответе (resp.json() бросает ValueError) ->
              тоже None -> _GENERIC_ERROR

    def chat(user_text: str, history=None) -> str
        - модульная обёртка над ленивым singleton-клиентом
          (_get_default_client()); тестируется отдельно и изолированно от
          singleton через патч самой _get_default_client.

Ничего из HTTP не выполняется по-настоящему — requests.post мокается через
unittest.mock. call_tool из tools.registry тоже мокается, чтобы не выполнять
никаких реальных системных действий.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests as real_requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ollama_client = pytest.importorskip(
    "brain.ollama_client",
    reason="brain/ollama_client.py недоступен для импорта",
)


def _mock_http_response(json_payload):
    resp = MagicMock()
    resp.json.return_value = json_payload
    resp.raise_for_status = MagicMock()
    return resp


def _make_client(**ollama_overrides) -> "ollama_client.OllamaClient":
    """Клиент с полностью изолированным конфигом (не трогает реальный
    config/config.yaml и не использует personality.py, чтобы тест не зависел
    от содержимого системного промпта личности)."""
    ollama_cfg = {
        "base_url": "http://localhost:11434",
        "model": "test-fast-model",
        "deep_model": "gemma4:26b",
        "timeout_seconds": 5,
        "max_tool_iterations": 3,
    }
    ollama_cfg.update(ollama_overrides)
    config = {
        "ollama": ollama_cfg,
        "assistant": {"personality_enabled": False, "name": "Jarvis"},
    }
    return ollama_client.OllamaClient(config=config)


class TestPlainResponse:
    def test_response_without_tool_call_is_returned_as_is(self):
        client = _make_client()
        payload = {
            "message": {
                "role": "assistant",
                "content": "Добрый день. Чем могу помочь?",
            }
        }
        with patch("requests.post", return_value=_mock_http_response(payload)) as mock_post:
            result = client.chat("Привет")

        assert result == "Добрый день. Чем могу помочь?"
        assert mock_post.call_count == 1

    def test_empty_user_text_returns_prompt_without_network_call(self):
        client = _make_client()
        with patch("requests.post") as mock_post:
            result = client.chat("   ")

        assert not mock_post.called
        assert "ничего не сказали" in result.lower()

    def test_normal_request_uses_fast_model(self):
        client = _make_client(model="test-fast-model", deep_model="gemma4:26b")
        payload = {"message": {"role": "assistant", "content": "Окей."}}
        with patch("requests.post", return_value=_mock_http_response(payload)) as mock_post:
            client.chat("Привет, как дела?")

        sent_json = mock_post.call_args.kwargs.get("json")
        assert sent_json is not None
        assert sent_json["model"] == "test-fast-model"

    def test_deep_analysis_phrase_switches_to_deep_model(self):
        client = _make_client(model="test-fast-model", deep_model="gemma4:26b")
        payload = {"message": {"role": "assistant", "content": "Обдумал основательно."}}
        with patch("requests.post", return_value=_mock_http_response(payload)) as mock_post:
            client.chat("Jarvis, подумай как следует над структурой проекта")

        sent_json = mock_post.call_args.kwargs.get("json")
        assert sent_json["model"] == "gemma4:26b"
        assert sent_json["model"] == client.deep_model


class TestToolCallLoop:
    def test_tool_call_triggers_registry_and_second_request(self):
        client = _make_client()
        first_payload = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "open_app",
                            "arguments": {"name": "notepad"},
                        }
                    }
                ],
            }
        }
        final_payload = {
            "message": {
                "role": "assistant",
                "content": "Открыл блокнот, как и просили.",
            }
        }

        with patch(
            "requests.post",
            side_effect=[
                _mock_http_response(first_payload),
                _mock_http_response(final_payload),
            ],
        ) as mock_post, patch.object(
            ollama_client,
            "call_tool",
            return_value="Открываю notepad.",
        ) as mock_call_tool:
            result = client.chat("Jarvis, открой блокнот")

        mock_call_tool.assert_called_once_with("open_app", {"name": "notepad"})
        assert mock_post.call_count == 2
        assert result == "Открыл блокнот, как и просили."

        # Проверяем, что результат инструмента реально попал в историю
        # второго запроса как сообщение роли "tool".
        second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == "Открываю notepad."
        assert tool_messages[0]["name"] == "open_app"

    def test_tool_call_arguments_as_json_string_are_parsed(self):
        """Ollama иногда присылает arguments как JSON-строку, а не dict."""
        client = _make_client()
        first_payload = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "курс доллара"}',
                        }
                    }
                ],
            }
        }
        final_payload = {
            "message": {"role": "assistant", "content": "Курс около 95 рублей."}
        }

        with patch(
            "requests.post",
            side_effect=[
                _mock_http_response(first_payload),
                _mock_http_response(final_payload),
            ],
        ), patch.object(
            ollama_client, "call_tool", return_value="Курс 95."
        ) as mock_call_tool:
            result = client.chat("Найди курс доллара")

        mock_call_tool.assert_called_once_with("web_search", {"query": "курс доллара"})
        assert result == "Курс около 95 рублей."

    def test_multiple_sequential_tool_calls_eventually_return_final_text(self):
        client = _make_client(max_tool_iterations=5)
        step1 = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}
                ],
            }
        }
        step2 = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_safe_command",
                            "arguments": {"cmd": "git status"},
                        }
                    }
                ],
            }
        }
        final = {"message": {"role": "assistant", "content": "Готово."}}

        with patch(
            "requests.post",
            side_effect=[
                _mock_http_response(step1),
                _mock_http_response(step2),
                _mock_http_response(final),
            ],
        ) as mock_post, patch.object(
            ollama_client, "call_tool", return_value="ok"
        ) as mock_call_tool:
            result = client.chat("Проверь git и покажи файл")

        assert mock_post.call_count == 3
        assert mock_call_tool.call_count == 2
        assert result == "Готово."

    def test_tool_iteration_limit_stops_infinite_loop(self):
        client = _make_client(max_tool_iterations=2)
        never_ending = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "lock_screen", "arguments": {}}}
                ],
            }
        }

        with patch(
            "requests.post", return_value=_mock_http_response(never_ending)
        ) as mock_post, patch.object(
            ollama_client, "call_tool", return_value="ok"
        ):
            result = client.chat("Зациклись, пожалуйста")

        assert mock_post.call_count == 2  # ровно max_tool_iterations
        assert "не удалось завершить" in result.lower()


class TestOllamaUnavailable:
    def test_connection_error_returns_generic_refusal_without_raising(self):
        client = _make_client()
        with patch(
            "requests.post",
            side_effect=real_requests.exceptions.ConnectionError("refused"),
        ):
            try:
                result = client.chat("Привет")
            except Exception as exc:  # noqa: BLE001
                pytest.fail(
                    f"chat() не должен пробрасывать исключение при недоступной "
                    f"Ollama, получено: {exc!r}"
                )

        assert result == ollama_client._GENERIC_ERROR
        assert any(ch.isalpha() and ord(ch) > 127 for ch in result), (
            "Ожидался текст отказа на русском языке (кириллица)"
        )

    def test_timeout_returns_generic_refusal_without_raising(self):
        client = _make_client()
        with patch(
            "requests.post", side_effect=real_requests.exceptions.Timeout("timed out")
        ):
            try:
                result = client.chat("Привет")
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"chat() не должен пробрасывать исключение, получено: {exc!r}")

        assert result == ollama_client._GENERIC_ERROR

    def test_http_error_status_returns_generic_refusal_without_raising(self):
        client = _make_client()
        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = real_requests.exceptions.HTTPError(
            "500 error"
        )
        with patch("requests.post", return_value=bad_response):
            try:
                result = client.chat("Привет")
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"chat() не должен пробрасывать исключение, получено: {exc!r}")

        assert result == ollama_client._GENERIC_ERROR

    def test_malformed_json_response_does_not_raise(self):
        client = _make_client()
        bad_response = MagicMock()
        bad_response.raise_for_status = MagicMock()
        bad_response.json.side_effect = ValueError("invalid json")

        with patch("requests.post", return_value=bad_response):
            try:
                result = client.chat("Привет")
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"chat() не должен пробрасывать исключение, получено: {exc!r}")

        assert result == ollama_client._GENERIC_ERROR

    def test_response_missing_message_key_does_not_raise(self):
        client = _make_client()
        with patch("requests.post", return_value=_mock_http_response({"unexpected": True})):
            try:
                result = client.chat("Привет")
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"chat() не должен пробрасывать исключение, получено: {exc!r}")

        assert result == ollama_client._GENERIC_ERROR

    def test_unexpected_internal_exception_is_caught_by_outer_guard(self):
        """Даже неожиданная ошибка внутри _chat_impl (не связанная с сетью)
        не должна долетать до вызывающего кода — это последняя линия обороны
        в chat()."""
        client = _make_client()
        with patch.object(
            client, "_select_model", side_effect=RuntimeError("boom")
        ):
            try:
                result = client.chat("Привет")
            except Exception as exc:  # noqa: BLE001
                pytest.fail(
                    f"chat() обязан перехватывать любые внутренние исключения, "
                    f"получено: {exc!r}"
                )

        assert result == ollama_client._GENERIC_ERROR


class TestModuleLevelConvenienceFunction:
    def test_module_chat_delegates_to_default_client(self):
        fake_client = MagicMock()
        fake_client.chat.return_value = "ответ по умолчанию"

        with patch.object(
            ollama_client, "_get_default_client", return_value=fake_client
        ):
            result = ollama_client.chat("привет", history=[{"role": "user", "content": "x"}])

        fake_client.chat.assert_called_once_with(
            "привет", [{"role": "user", "content": "x"}]
        )
        assert result == "ответ по умолчанию"
