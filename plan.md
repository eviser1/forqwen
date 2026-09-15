# PLAN: Jarvis — техническая архитектура и разбивка по ролям

## Архитектура (data flow)
```
Микрофон → wake/ (Porcupine, keyword "Jarvis")
  → [сработало] звук пробуждения (voice/cues.py)
  → stt/ (запись до тишины VAD → faster-whisper → русский текст)
  → brain/ (Ollama chat: системный промпт личности + tool schema + история сессии)
     → tools/ (выполнение функции из белого списка, если запрошено) → назад в brain/
     → финальный текст ответа
  → tts/ (Silero → аудио, поддержка прерывания/barge-in)
  → orchestrator.py возвращается в режим прослушивания
```

## Структура репозитория
```
jarvis/
  spec.md, plan.md, DECISIONS.md, RESEARCH.md, README.md, MANUAL_TEST.md
  requirements.txt
  config/config.yaml
  wake/porcupine_listener.py
  stt/transcriber.py
  tts/speaker.py
  brain/ollama_client.py, brain/personality.py
  tools/registry.py, tools/actions.py
  orchestrator.py
  service/autostart.ps1, service/tray.py
  logs/  (создаётся в рантайме)
  tests/test_tools.py, tests/test_brain.py
```

## Роли и границы (для оркестрации Шага 2 `/allforme`)

- **research-agent** → пишет `RESEARCH.md`. Задачи: (a) найти конкретную модель Ollama для tool-calling, реально доступную сейчас (`ollama pull` кандидата, зафиксировать точный тег и размер); (b) поискать существующие GitHub-репозитории "local voice assistant Ollama whisper wake word" под возможный форк — зафиксировать находки и рекомендацию; (c) актуальный статус лицензии Picovoice Porcupine для 24/7 личного использования. Файлы: только `RESEARCH.md`.

- **backend-agent** → реализует `brain/` и `tools/` и `config/config.yaml` (схема конфига). Ollama tool-calling loop, белый список функций с явной валидацией аргументов (никакого сырого shell от модели), системный промпт личности на русском. Файлы: `brain/*.py`, `tools/*.py`, `config/config.yaml`.

- **voice-agent** (по факту тоже backend-фокус, инфраструктура голоса) → реализует `wake/`, `stt/`, `tts/`, включая звуковой сигнал пробуждения и поддержку прерывания (barge-in hook, вызываемый из `orchestrator.py`). Файлы: `wake/*.py`, `stt/*.py`, `tts/*.py`.

- **frontend-agent** → трей-иконка (`service/tray.py`, 3 состояния: сон/слушаю/думаю), скрипт автозапуска (`service/autostart.ps1` — регистрация задачи в Планировщике, скрытое окно, restart-on-failure), `README.md` (установка и запуск). Файлы: `service/*`, `README.md`.

- **qa-agent** → `tests/test_tools.py` и `tests/test_brain.py` (мокая Ollama-ответы, никакого реального аудио), `MANUAL_TEST.md` с конкретными голосовыми фразами для ручной проверки (включая тест на отказ в запрещённом действии). Файлы: `tests/*`, `MANUAL_TEST.md`.

Итоговая склейка (`orchestrator.py`, `requirements.txt`, `logs/`) — синтез после всех ролей, делает оркестратор (не отдельный агент), т.к. требует результатов всех остальных.

## Общие правила для всех ролей
- Тяжёлые артефакты (venv, модели, кэши) — только на диске D.
- Ни один модуль не вызывает `subprocess`/`os.system` со строкой, собранной из вывода модели без явной валидации по белому списку.
- Код на Python 3.11+, с type hints, без лишних абстракций (YAGNI).
- Прогресс и решения — в `DECISIONS.md`, а не только в переписке.
