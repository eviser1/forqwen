# -*- coding: utf-8 -*-
"""
Юнит-тесты для tools/actions.py и tools/registry.py.

ВАЖНО: tools/actions.py и tools/registry.py к моменту написания этого файла
уже были готовы на диске (backend-agent писал параллельно) — все тесты ниже
сверены с РЕАЛЬНЫМ кодом (сигнатуры, возвращаемые значения, имена внутренних
функций). pytest.importorskip оставлен как защита на случай запуска тестов до
появления модулей в другом окружении/порядке — тогда коллекция не упадёт, а
эти тесты будут аккуратно пропущены.

Зафиксированный (по факту) контракт tools/registry.py:
    REFUSAL_UNKNOWN_TOOL: str — константа-отказ для неизвестного инструмента.
    TOOL_REGISTRY: list[dict] — список записей {"name", "description",
        "parameters_json_schema", "handler"} (НЕ dict вида {имя: функция}).
    call_tool(name: str, arguments: dict) -> str
        - name не в реестре -> возвращает REFUSAL_UNKNOWN_TOOL, handler НЕ
          вызывается, исключение не бросается
        - arguments не dict -> отказ, handler не вызывается
        - лишние ключи в arguments, не описанные в parameters_json_schema,
          отфильтровываются перед вызовом handler(**filtered_args)
        - исключение внутри handler (включая TypeError на несоответствии
          аргументов) перехватывается и превращается в строку-отказ, наружу
          не пробрасывается

Зафиксированный (по факту) контракт tools/actions.py:
    REFUSAL_PREFIX = "Не могу выполнить это действие: "
    Все функции возвращают str. Успех — обычный текст (например "Открываю
    notepad."). Отказ — текст, начинающийся с REFUSAL_PREFIX. Исключения
    наружу не пробрасываются — перехватываются внутри и превращаются в отказ.

    SAFE_COMMANDS: dict[str, list[str]] — РОВНО {"git status": [...],
        "git diff": [...], "npm test": [...]}. run_safe_command(cmd: str) -> str
        ищет cmd.strip() в SAFE_COMMANDS по точному совпадению ключа; для всего
        остального НЕ вызывает subprocess.run вообще.
    ALLOWED_ROOTS: list[Path] — read_file(path) / write_code_file(path, content)
        отклоняют пути, не лежащие внутри одного из ALLOWED_ROOTS (через
        _is_within_allowed_roots, сравнение по resolve()).
    ALLOWED_APPS: dict[str, str] (ключ — lower-case имя) — open_app(name)
        вызывает subprocess.Popen([target], shell=False) ТОЛЬКО если
        name.strip().lower() есть в ALLOWED_APPS.
    open_url(url) — валидирует urllib.parse.urlparse(url).scheme in
        ("http", "https") и наличие netloc, затем вызывает webbrowser.open(url)
        (импортируется локально внутри функции).
    set_volume(level: int) — 0 <= level <= 100, вызывает _set_volume_pycaw(level),
        при ImportError — резервный _set_volume_winmm(level).
    lock_screen() — вызывает ctypes.windll.user32.LockWorkStation().
    open_explorer(path=None) — без пути: subprocess.Popen(["explorer.exe"]);
        с путём — только если путь внутри ALLOWED_ROOTS и существует.
    web_search(query) — DuckDuckGo HTML без ключа; пустой запрос отклоняется
        без сетевых вызовов; при недоступности requests/bs4 используется
        локальный fallback на urllib.request.

Ничего из системных вызовов не выполняется по-настоящему: subprocess, ctypes,
webbrowser, urllib, requests — везде unittest.mock.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Позволяет запускать `pytest` из корня проекта без установки пакета.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

actions = pytest.importorskip(
    "tools.actions",
    reason="tools/actions.py недоступен для импорта",
)
registry = pytest.importorskip(
    "tools.registry",
    reason="tools/registry.py недоступен для импорта",
)


def _is_refusal(result) -> bool:
    """Толерантная проверка отказа — распознаёт обе реально используемые в
    проекте конвенции: actions.REFUSAL_PREFIX ("Не могу выполнить это
    действие: ...") и registry.py ("Отказ: ...", "Произошла ошибка при
    выполнении «...».")."""
    if isinstance(result, str):
        lowered = result.lower()
        markers = ("не могу", "отказ", "произошла ошибка")
        return any(marker in lowered for marker in markers)
    if isinstance(result, dict):
        for key in ("success", "ok"):
            if key in result:
                val = result[key]
                return not (val is True or (isinstance(val, str) and val.lower() == "ok"))
        return "error" in result
    return False


def _is_success(result) -> bool:
    assert isinstance(result, (str, dict)), "Ожидалась строка или словарь"
    return not _is_refusal(result)


# ---------------------------------------------------------------------------
# run_safe_command — ГЛАВНЫЙ security-тест: белый список shell-команд
# ---------------------------------------------------------------------------

ALLOWED_COMMANDS = ["git status", "git diff", "npm test"]

FORBIDDEN_COMMANDS = [
    "del file.txt",
    "del /f /q C:\\*",
    "shutdown /s /t 0",
    "rm -rf /",
    "rm -rf ~",
    "format C:",
    "reg delete HKLM",
    "какая-то произвольная строка",
    "git push --force",  # git, но не из белого списка
    "npm install malicious-package",  # npm, но не из белого списка
    "git status && del *.*",  # попытка инъекции через разрешённый префикс
    "",
    "   ",
]


class TestRunSafeCommandWhitelist:
    @pytest.mark.parametrize("command", ALLOWED_COMMANDS)
    def test_allowed_command_executes(self, command):
        with patch.object(actions, "subprocess") as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(
                returncode=0, stdout="ok", stderr=""
            )
            result = actions.run_safe_command(command)

        assert mock_subprocess.run.called, (
            f"Разрешённая команда {command!r} должна была вызвать subprocess.run"
        )
        assert _is_success(result), f"Ожидался успех для разрешённой команды {command!r}"

    @pytest.mark.parametrize("command", FORBIDDEN_COMMANDS)
    def test_forbidden_command_is_rejected_without_executing(self, command):
        with patch.object(actions, "subprocess") as mock_subprocess:
            result = actions.run_safe_command(command)

        assert not mock_subprocess.run.called, (
            f"Команда {command!r} НЕ входит в белый список — "
            f"subprocess.run НЕ должен был вызываться"
        )
        assert _is_refusal(result), f"Ожидался отказ для команды {command!r}"

    def test_run_safe_command_never_uses_shell_true(self):
        """Даже для разрешённой команды нельзя запускать через shell=True."""
        with patch.object(actions, "subprocess") as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(
                returncode=0, stdout="", stderr=""
            )
            actions.run_safe_command("git status")

        _, kwargs = mock_subprocess.run.call_args
        assert kwargs.get("shell", False) is False, (
            "run_safe_command не должен вызывать subprocess.run с shell=True"
        )

    def test_allowed_command_uses_argv_list_not_string(self):
        """SAFE_COMMANDS должен передавать argv-список, а не собранную строку."""
        with patch.object(actions, "subprocess") as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(
                returncode=0, stdout="", stderr=""
            )
            actions.run_safe_command("git status")

        args, _ = mock_subprocess.run.call_args
        argv = args[0]
        assert isinstance(argv, list), "Ожидался список argv, а не строка команды"
        assert argv == ["git", "status"]


# ---------------------------------------------------------------------------
# install_package / install_app / restart_self — разрешено пользователем явно
# (см. DECISIONS.md), но имя пакета/программы обязано проходить валидацию
# по маске _SAFE_NAME_RE до того, как subprocess вообще вызывается.
# ---------------------------------------------------------------------------

VALID_PACKAGE_NAMES = ["requests", "numpy", "some-package_2.0"]
INJECTION_ATTEMPT_NAMES = [
    "requests && del *.*",
    "requests; rm -rf /",
    "requests | shutdown",
    "some package",
    "../../etc/passwd",
    "requests`whoami`",
    "",
]


class TestInstallPackage:
    @pytest.mark.parametrize("name", VALID_PACKAGE_NAMES)
    def test_valid_name_calls_pip_via_argv(self, name):
        with patch.object(actions, "subprocess") as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = actions.install_package(name)

        assert mock_subprocess.run.called
        args, kwargs = mock_subprocess.run.call_args
        argv = args[0]
        assert isinstance(argv, list)
        assert argv[-1] == name
        assert "pip" in argv
        assert "install" in argv
        assert kwargs.get("shell", False) is False
        assert _is_success(result)

    @pytest.mark.parametrize("name", INJECTION_ATTEMPT_NAMES)
    def test_invalid_name_rejected_without_calling_subprocess(self, name):
        with patch.object(actions, "subprocess") as mock_subprocess:
            result = actions.install_package(name)

        assert not mock_subprocess.run.called, (
            f"Имя {name!r} не проходит валидацию — pip не должен был вызываться"
        )
        assert _is_refusal(result)


class TestInstallApp:
    @pytest.mark.parametrize("name", VALID_PACKAGE_NAMES)
    def test_valid_name_calls_winget_via_argv(self, name):
        with patch.object(actions, "subprocess") as mock_subprocess:
            mock_subprocess.run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = actions.install_app(name)

        assert mock_subprocess.run.called
        args, kwargs = mock_subprocess.run.call_args
        argv = args[0]
        assert isinstance(argv, list)
        assert argv[0] == "winget"
        assert name in argv
        assert kwargs.get("shell", False) is False
        assert _is_success(result)

    @pytest.mark.parametrize("name", INJECTION_ATTEMPT_NAMES)
    def test_invalid_name_rejected_without_calling_subprocess(self, name):
        with patch.object(actions, "subprocess") as mock_subprocess:
            result = actions.install_app(name)

        assert not mock_subprocess.run.called
        assert _is_refusal(result)


class TestRestartSelf:
    def test_returns_confirmation_and_spawns_thread(self):
        with patch.object(actions, "subprocess") as mock_subprocess, patch.object(
            actions, "threading"
        ) as mock_threading:
            result = actions.restart_self()

        assert isinstance(result, str) and result
        assert mock_threading.Thread.called, "restart_self должен планировать отложенный перезапуск в потоке"
        # Само subprocess.Popen не должно вызываться синхронно — только внутри
        # запланированного потока (который мы здесь замокали и не исполняли).
        assert not mock_subprocess.Popen.called


# ---------------------------------------------------------------------------
# read_file / write_code_file — ограничение allowed_roots
# ---------------------------------------------------------------------------


class TestFileSandbox:
    def test_read_file_within_allowed_root_succeeds(self, tmp_path, monkeypatch):
        allowed_dir = tmp_path / "project"
        allowed_dir.mkdir()
        target = allowed_dir / "notes.txt"
        target.write_text("привет", encoding="utf-8")

        monkeypatch.setattr(actions, "ALLOWED_ROOTS", [allowed_dir], raising=False)

        result = actions.read_file(str(target))

        assert _is_success(result)
        assert "привет" in result

    def test_read_file_outside_allowed_root_is_rejected(self, tmp_path, monkeypatch):
        allowed_dir = tmp_path / "project"
        allowed_dir.mkdir()
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("секрет", encoding="utf-8")

        monkeypatch.setattr(actions, "ALLOWED_ROOTS", [allowed_dir], raising=False)

        result = actions.read_file(str(outside_file))

        assert _is_refusal(result)
        assert "секрет" not in result

    def test_read_file_rejects_traversal(self, tmp_path, monkeypatch):
        allowed_dir = tmp_path / "project"
        allowed_dir.mkdir()
        (tmp_path / "secret.txt").write_text("секрет", encoding="utf-8")

        monkeypatch.setattr(actions, "ALLOWED_ROOTS", [allowed_dir], raising=False)

        traversal_path = str(allowed_dir / ".." / "secret.txt")
        result = actions.read_file(traversal_path)

        assert _is_refusal(result)
        assert "секрет" not in result

    def test_read_file_missing_file_is_refusal_not_exception(self, tmp_path, monkeypatch):
        allowed_dir = tmp_path / "project"
        allowed_dir.mkdir()
        monkeypatch.setattr(actions, "ALLOWED_ROOTS", [allowed_dir], raising=False)

        result = actions.read_file(str(allowed_dir / "does_not_exist.txt"))

        assert _is_refusal(result)

    def test_write_code_file_within_allowed_root_succeeds(self, tmp_path, monkeypatch):
        allowed_dir = tmp_path / "project"
        allowed_dir.mkdir()
        target = allowed_dir / "generated.py"

        monkeypatch.setattr(actions, "ALLOWED_ROOTS", [allowed_dir], raising=False)

        result = actions.write_code_file(str(target), "print('hi')\n")

        assert _is_success(result)
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "print('hi')\n"

    def test_write_code_file_outside_allowed_root_is_rejected_and_no_io(
        self, tmp_path, monkeypatch
    ):
        allowed_dir = tmp_path / "project"
        allowed_dir.mkdir()
        outside_target = tmp_path / "hack.py"

        monkeypatch.setattr(actions, "ALLOWED_ROOTS", [allowed_dir], raising=False)

        result = actions.write_code_file(str(outside_target), "import os\n")

        assert _is_refusal(result)
        assert not outside_target.exists(), (
            "Запись за пределы allowed_roots не должна была создать файл"
        )

    def test_write_code_file_rejects_system_path(self, monkeypatch):
        # Явный системный путь Windows должен быть отклонён вне зависимости
        # от того, что настроено в ALLOWED_ROOTS для теста.
        monkeypatch.setattr(
            actions, "ALLOWED_ROOTS", [Path("D:\\jarvis_project")], raising=False
        )
        result = actions.write_code_file(
            "C:\\Windows\\System32\\drivers\\etc\\hosts", "0.0.0.0 example.com"
        )
        assert _is_refusal(result)


# ---------------------------------------------------------------------------
# open_url / open_app / open_explorer / set_volume / lock_screen
# ---------------------------------------------------------------------------


class TestSystemActions:
    def test_open_url_calls_webbrowser_with_validated_url(self):
        with patch("webbrowser.open") as mock_open:
            result = actions.open_url("https://www.google.com/search?q=курс+доллара")

        mock_open.assert_called_once_with(
            "https://www.google.com/search?q=курс+доллара"
        )
        assert _is_success(result)

    @pytest.mark.parametrize(
        "bad_url",
        [
            "javascript:alert(1)",
            "file:///C:/Windows/System32",
            "not a url",
            "ftp://internal-server/secrets",
            "",
        ],
    )
    def test_open_url_rejects_non_http_scheme(self, bad_url):
        with patch("webbrowser.open") as mock_open:
            result = actions.open_url(bad_url)

        assert not mock_open.called
        assert _is_refusal(result)

    def test_open_app_calls_popen_for_whitelisted_app(self, monkeypatch):
        monkeypatch.setattr(
            actions, "ALLOWED_APPS", {"notepad": "notepad.exe"}, raising=False
        )
        with patch.object(actions, "subprocess") as mock_subprocess:
            result = actions.open_app("notepad")

        mock_subprocess.Popen.assert_called_once_with(["notepad.exe"], shell=False)
        assert _is_success(result)

    def test_open_app_is_case_and_space_insensitive_for_lookup(self, monkeypatch):
        monkeypatch.setattr(
            actions, "ALLOWED_APPS", {"notepad": "notepad.exe"}, raising=False
        )
        with patch.object(actions, "subprocess") as mock_subprocess:
            result = actions.open_app("  Notepad  ")

        mock_subprocess.Popen.assert_called_once_with(["notepad.exe"], shell=False)
        assert _is_success(result)

    def test_open_app_rejects_app_not_in_whitelist(self, monkeypatch):
        monkeypatch.setattr(
            actions, "ALLOWED_APPS", {"notepad": "notepad.exe"}, raising=False
        )
        with patch.object(actions, "subprocess") as mock_subprocess:
            result = actions.open_app("C:\\malware\\evil.exe")

        assert not mock_subprocess.Popen.called
        assert _is_refusal(result)

    def test_open_app_rejects_empty_name(self, monkeypatch):
        monkeypatch.setattr(
            actions, "ALLOWED_APPS", {"notepad": "notepad.exe"}, raising=False
        )
        with patch.object(actions, "subprocess") as mock_subprocess:
            result = actions.open_app("")

        assert not mock_subprocess.Popen.called
        assert _is_refusal(result)

    def test_open_explorer_without_path_opens_plain_explorer(self):
        with patch.object(actions, "subprocess") as mock_subprocess:
            result = actions.open_explorer()

        mock_subprocess.Popen.assert_called_once_with(["explorer.exe"], shell=False)
        assert _is_success(result)

    def test_open_explorer_with_allowed_existing_path(self, tmp_path, monkeypatch):
        allowed_dir = tmp_path / "project"
        allowed_dir.mkdir()
        monkeypatch.setattr(actions, "ALLOWED_ROOTS", [allowed_dir], raising=False)

        with patch.object(actions, "subprocess") as mock_subprocess:
            result = actions.open_explorer(str(allowed_dir))

        mock_subprocess.Popen.assert_called_once_with(
            ["explorer.exe", str(allowed_dir.resolve())], shell=False
        )
        assert _is_success(result)

    def test_open_explorer_with_path_outside_allowed_root_is_rejected(
        self, tmp_path, monkeypatch
    ):
        allowed_dir = tmp_path / "project"
        allowed_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        monkeypatch.setattr(actions, "ALLOWED_ROOTS", [allowed_dir], raising=False)

        with patch.object(actions, "subprocess") as mock_subprocess:
            result = actions.open_explorer(str(outside_dir))

        assert not mock_subprocess.Popen.called
        assert _is_refusal(result)

    def test_set_volume_accepts_valid_level(self):
        with patch.object(actions, "_set_volume_pycaw") as mock_pycaw:
            result = actions.set_volume(50)

        mock_pycaw.assert_called_once_with(50)
        assert _is_success(result)

    def test_set_volume_falls_back_to_winmm_when_pycaw_unavailable(self):
        with patch.object(
            actions, "_set_volume_pycaw", side_effect=ImportError("no pycaw")
        ), patch.object(actions, "_set_volume_winmm") as mock_winmm:
            result = actions.set_volume(30)

        mock_winmm.assert_called_once_with(30)
        assert _is_success(result)

    @pytest.mark.parametrize("bad_level", [-10, 101, 1000, -1])
    def test_set_volume_rejects_out_of_range(self, bad_level):
        with patch.object(actions, "_set_volume_pycaw") as mock_pycaw, patch.object(
            actions, "_set_volume_winmm"
        ) as mock_winmm:
            result = actions.set_volume(bad_level)

        assert not mock_pycaw.called
        assert not mock_winmm.called
        assert _is_refusal(result)

    def test_set_volume_rejects_non_int(self):
        with patch.object(actions, "_set_volume_pycaw") as mock_pycaw:
            result = actions.set_volume("50")  # type: ignore[arg-type]

        assert not mock_pycaw.called
        assert _is_refusal(result)

    def test_lock_screen_calls_win32_lock_workstation(self):
        with patch.object(actions, "ctypes") as mock_ctypes:
            mock_ctypes.windll.user32.LockWorkStation.return_value = 1
            result = actions.lock_screen()

        mock_ctypes.windll.user32.LockWorkStation.assert_called_once_with()
        assert _is_success(result)

    def test_lock_screen_failure_return_value_is_refusal(self):
        with patch.object(actions, "ctypes") as mock_ctypes:
            mock_ctypes.windll.user32.LockWorkStation.return_value = 0
            result = actions.lock_screen()

        assert _is_refusal(result)


# ---------------------------------------------------------------------------
# web_search — DuckDuckGo без ключа
# ---------------------------------------------------------------------------


class TestWebSearch:
    def test_empty_query_is_rejected_without_network_call(self, monkeypatch):
        urlopen_mock = MagicMock(side_effect=AssertionError("не должен вызываться"))
        monkeypatch.setattr(
            actions.urllib.request, "urlopen", urlopen_mock, raising=False
        )
        result = actions.web_search("   ")

        assert _is_refusal(result)
        assert not urlopen_mock.called

    def test_fallback_network_failure_returns_refusal_text(self, monkeypatch):
        # Заставляем основной путь (requests/bs4) отсутствовать, чтобы
        # detereministично проверить резервную реализацию на чистом urllib.
        monkeypatch.setitem(sys.modules, "requests", None)
        monkeypatch.setitem(sys.modules, "bs4", None)

        with patch.object(
            actions.urllib.request,
            "urlopen",
            side_effect=OSError("network unreachable"),
        ):
            result = actions.web_search("курс доллара")

        assert isinstance(result, str)
        assert "не удал" in result.lower() or _is_refusal(result)

    def test_success_path_with_requests_and_bs4(self):
        requests_mod = pytest.importorskip("requests")
        pytest.importorskip("bs4")

        # bs4-селекторы, используемые в actions.py, ищут .result__title /
        # .result__snippet внутри .result__body — минимальная подходящая разметка.
        fake_html = """
        <div class="result__body">
            <div class="result__title">Курс доллара сегодня</div>
            <div class="result__snippet">Актуальный курс 95 рублей</div>
        </div>
        """
        fake_response = MagicMock()
        fake_response.text = fake_html
        fake_response.raise_for_status = MagicMock()

        with patch.object(requests_mod, "post", return_value=fake_response):
            result = actions.web_search("курс доллара")

        assert _is_success(result)
        assert "Курс доллара сегодня" in result


# ---------------------------------------------------------------------------
# tools.registry.call_tool
# ---------------------------------------------------------------------------


def _registered_tool_names() -> set[str]:
    return {tool["name"] for tool in registry.TOOL_REGISTRY}


def _patch_registered_handler(monkeypatch, name: str, replacement):
    """TOOL_REGISTRY захватывает ссылку на функцию actions.<name> в момент
    импорта registry.py (handler: actions.open_url), поэтому патчить
    tools.actions.open_url ПОСЛЕ импорта регистра бесполезно — нужно подменить
    сам ключ "handler" в записи реестра (тот же dict-объект используется и в
    TOOL_REGISTRY, и в _REGISTRY_BY_NAME, так что одной подмены достаточно)."""
    tool_entry = next(t for t in registry.TOOL_REGISTRY if t["name"] == name)
    monkeypatch.setitem(tool_entry, "handler", replacement)


class TestRegistryCallTool:
    def test_unknown_function_name_returns_refusal_without_raising(self):
        try:
            result = registry.call_tool("delete_everything", {})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"call_tool не должен пробрасывать исключение наружу, "
                f"получено: {exc!r}"
            )

        assert result == registry.REFUSAL_UNKNOWN_TOOL
        assert "delete_everything" not in _registered_tool_names()

    def test_non_dict_arguments_are_rejected_without_calling_handler(self, monkeypatch):
        mock_handler = MagicMock()
        _patch_registered_handler(monkeypatch, "open_url", mock_handler)

        result = registry.call_tool("open_url", "https://example.com")  # type: ignore[arg-type]

        assert not mock_handler.called
        assert _is_refusal(result)

    def test_known_function_delegates_with_filtered_arguments(self, monkeypatch):
        mock_handler = MagicMock(return_value="Открываю https://example.com.")
        _patch_registered_handler(monkeypatch, "open_url", mock_handler)

        result = registry.call_tool(
            "open_url",
            {"url": "https://example.com", "unexpected_extra_field": "x"},
        )

        mock_handler.assert_called_once_with(url="https://example.com")
        assert result == "Открываю https://example.com."

    def test_exception_inside_handler_is_caught_and_returns_refusal(self, monkeypatch):
        mock_handler = MagicMock(side_effect=RuntimeError("boom"))
        _patch_registered_handler(monkeypatch, "open_url", mock_handler)

        try:
            result = registry.call_tool("open_url", {"url": "https://example.com"})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"Исключение из обработчика не должно долетать до "
                f"вызывающего кода, получено: {exc!r}"
            )

        assert isinstance(result, str)
        assert _is_refusal(result)

    def test_dangerous_names_never_registered(self):
        """Дополнительная защита: в реестре не должно быть функций удаления/
        форматирования/выключения ПК — они должны физически отсутствовать в
        коде, а не просто быть недоступны через промпт (см. spec.md, п.5).

        Исключение: `restart_self` — по явному запросу пользователя (см.
        DECISIONS.md) разрешён перезапуск ПРОЦЕССА ассистента (после
        write_code_file), это не выключение/перезагрузка компьютера.
        """
        forbidden_substrings = (
            "delete",
            "remove",
            "format",
            "shutdown",
            "reboot",
            "reg_",
            "uninstall",
            "rm_",
        )
        allowed_exceptions = {"restart_self"}
        for name in _registered_tool_names():
            if name in allowed_exceptions:
                continue
            lowered = name.lower()
            for bad in forbidden_substrings:
                assert bad not in lowered, (
                    f"Найдена потенциально опасная функция в реестре: {name!r}"
                )
            assert "restart" not in lowered, (
                f"Найдена незадокументированная 'restart'-функция: {name!r} "
                "(разрешён только restart_self из allowed_exceptions)"
            )

    def test_get_tool_schemas_matches_registry_names(self):
        schemas = registry.get_tool_schemas()
        schema_names = {s["function"]["name"] for s in schemas}
        assert schema_names == _registered_tool_names()
