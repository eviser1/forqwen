# RESEARCH: Jarvis — результаты research-фазы

## Модель Ollama

**Установлено сейчас** (`ollama list`, машина пользователя):
- `nomic-embed-text:latest` — 274 MB
- `gemma4:e2b` — 7.2 GB (в spec.md упомянута как `gemma4:2b`, реальный тег `gemma4:e2b`)
- `gemma4:26b` — 18 GB (режим глубокого анализа, уже зафиксирован в spec.md)

Ни одна из установленных моделей не является целевой ~7-8B tool-calling моделью общего назначения — нужна отдельная загрузка.

**Рекомендация: `qwen3:8b`**
- Тег для `ollama pull qwen3:8b`.
- Размер на диске: **5.2 GB** (дефолтная квантизация, 40K context window) — источник: https://ollama.com/library/qwen3
- Почему именно она: Qwen3 — самое актуальное (2026) поколение с нативным tool-calling шаблоном в Ollama (карточка модели явно показывает capability `tools`), по независимым сравнениям tool-calling бенчмарков (BFCL) Qwen3-серия сейчас лидирует по стабильности вызовов инструментов среди локальных моделей с наименьшим процентом "dropped tool calls"; 5.2 GB веса оставляют комфортный запас в 12 GB VRAM под KV-cache контекста, faster-whisper (GPU) и системные процессы Windows.
- Резервный вариант, если `qwen3:8b` не устроит по качеству генерации на русском: `qwen2.5:7b-instruct` (тег `qwen2.5:7b-instruct`, дефолт 4.7 GB, Q4_K_M, тоже с явной поддержкой tools) — предыдущее поколение, чуть менее надёжный tool-calling, но хорошо задокументированный русский язык.
- Модель **не скачивалась** в рамках этого research-шага (по инструкции — опционально; решил не тянуть 5.2 GB, чтобы не затягивать research-фазу и не рисковать таймаутом). Загрузку `ollama pull qwen3:8b` нужно выполнить на этапе интеграции (`brain/`).

## Кандидаты для форка

Ничего похожего на "готовый Jarvis под Windows с Porcupine+faster-whisper+Silero+Ollama tool-calling" один в один не нашлось. Ближайшие релевантные проекты:

| Репозиторий | Что делает | Звёзды / активность | Стек | Лицензия |
|---|---|---|---|---|
| [llm-guy/jarvis](https://github.com/llm-guy/jarvis) | Wake word → локальный LLM (Qwen via Ollama, через LangChain) → tool-calling (напр. время) → TTS | **332 ⭐**, последний push 2025-09-08 (год назад, не свежий) | Python, LangChain, Ollama | лицензия не указана в репо (None) |
| [ndunl075/Jarvis](https://github.com/ndunl075/Jarvis) | Полностью локальный ассистент **для Windows**: wake word, STT, Ollama LLM, tools, без подписки | 7 ⭐, push **2026-09-09** (актуально прямо сейчас) | Python | **MIT** |
| [PanPenek/JarvisAi](https://github.com/PanPenek/JarvisAi) | Оффлайн: wake word, screen vision, 31 tool, Iron Man web UI; Whisper STT + Kokoro TTS + Ollama | 9 ⭐, push 2026-03-01 | Python | не указана (None) |
| [Tgerrrrr/Jarvis-Voice-Assistant](https://github.com/Tgerrrrr/Jarvis-Voice-Assistant) | openWakeWord + faster-whisper (CPU, int8) + Ollama (llama3.2) | 0 ⭐, push 2026-08-22, в основном Jupyter Notebook | Python/Notebook | не проверялась |

**Рекомендация: собирать с нуля по plan.md, форк не рекомендую.**
Обоснование:
- Самый "звёздный" (`llm-guy/jarvis`, 332⭐) не обновлялся год, тянет LangChain (лишняя абстракция, конфликтует с YAGNI из plan.md), TTS/wake-word/русский язык не подтверждены.
- Самый свежий и по стеку ближе всего (`ndunl075/Jarvis`, MIT, Windows-специфичный) имеет всего 7 звёзд — нет независимой валидации качества/безопасности кода, а требования spec.md к белому списку действий и физическому запрету опасных операций критичны (безопасность), доверять непроверенному мелкому репо форк-базой рискованно.
- Ни один кандидат не использует связку Porcupine/openWakeWord + faster-whisper GPU + Silero TTS + русский язык + Windows Task Scheduler автозапуск одновременно — интеграция всё равно потребует переписать большую часть кода.
- Итог: код `ndunl075/Jarvis` и `llm-guy/jarvis` можно точечно посмотреть как референс паттернов (структура tool-calling loop, обработка barge-in), но основой репозитория брать не стоит — план.md уже даёт чёткую архитектуру, дешевле реализовать с нуля.

## Porcupine

**Критическая находка, расходится с допущением spec.md** ("встроенное бесплатное ключевое слово"):

- Picovoice официально подтвердил закрытие Free Tier: AccessKey-и бесплатного тарифа отключаются **30 июня 2026**. Источник: https://community.home-assistant.io/t/fyi-picovoice-confirmed-free-tier-accesskeys-will-stop-working-after-june-30-2026/1012744 — цитата от Picovoice: *"After Free Tier AccessKeys are disabled on June 30, 2026, features using those keys will stop working."* и *"Going forward, we'll be focusing on our core business, enterprise deployments. There is no non-commercial tier planned."*
- Сегодня 15 сентября 2026 — то есть бесплатный тариф уже закрыт, свободного некоммерческого использования Porcupine (даже с ключевым словом "Jarvis" в builtin-наборе, которое там действительно было) **больше нет**. Взамен Picovoice предлагает только 7-дневный free trial для команд — не подходит для постоянного 24/7 личного использования без оплаты.
- Ограничения бывшего free-тира (для справки, уже неактуально): до 3 активных пользователей/месяц, требовался `AccessKey`, встроенное слово "Jarvis" в наборе присутствовало (`alexa`, `computer`, `jarvis` и др.).

**Рекомендация: заменить Porcupine на [openWakeWord](https://github.com/dscripka/openWakeWord) (MIT/Apache, полностью бесплатно и офлайн, без access key).**
- В стандартном наборе pretrained-моделей openWakeWord есть готовая модель **`hey_jarvis`** ("hey jarvis") — задокументирована здесь: https://github.com/dscripka/openWakeWord/blob/main/docs/models/hey_jarvis.md — обучена на ~200k синтетических клипов фразы "hey jarvis", готова к использованию из коробки, дообучение не требуется.
- Нюанс: builtin-модель триггерится на фразу **"hey jarvis"**, а не одиночное "Jarvis" (как было у Porcupine) — по документации openWakeWord одиночное "jarvis" тоже может сработать, но с более высоким false-reject rate. Нужно решить на уровне spec.md/UX: либо принять фразу "Hey, Jarvis", либо (вне scope v1 по текущему spec.md — "обучение кастомного wake word") обучать свою модель под голое "Jarvis", что противоречит текущему ограничению "вне scope".
- Это расхождение со spec.md пункт 1 требует явного решения пользователя/оркестратора перед реализацией `wake/` — сам файл spec.md я не трогал.

---

### Итог (для отчёта агента, не для файла)
Модель для tool-calling: `ollama pull qwen3:8b` (5.2 GB, нативная поддержка tools, лучшие бенчмарки tool-calling среди локальных моделей на 2026 год) — не скачивалась, только зафиксирован тег. Готового репозитория под чистый форк не нашлось — рекомендую собирать по plan.md с нуля, при этом можно подсматривать паттерны в `llm-guy/jarvis` (332⭐, но год не обновлялся, LangChain) и `ndunl075/Jarvis` (MIT, Windows, свежий, но всего 7⭐). Самое важное: **бесплатный тариф Picovoice Porcupine закрылся 30 июня 2026 и уже недоступен** — постоянное некоммерческое 24/7 использование невозможно без оплаты; рекомендую заменить на open-source `openWakeWord`, у которого есть готовая модель `hey_jarvis`, но она триггерится на фразу "hey jarvis", а не одиночное "Jarvis" — это расхождение со spec.md, требует явного решения перед реализацией wake-модуля.
