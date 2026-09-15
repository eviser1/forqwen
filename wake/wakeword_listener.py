"""Wake word detection for Jarvis, built on openWakeWord (open-source, free).

Replaces the earlier Picovoice Porcupine implementation: Porcupine's free
tier for personal/24-7 use was discontinued on 2026-06-30, making it
unusable for this project. openWakeWord (https://github.com/dscripka/openWakeWord,
Apache-2.0 code license) requires no access key/registration/account at all —
the only "setup cost" is a one-time download of the pretrained model weights
(a few MB, fetched automatically from GitHub release assets on first run).

*** IMPORTANT — WAKE PHRASE MISMATCH WITH spec.md ***
spec.md specifies the wake word "Jarvis" (bare). openWakeWord's relevant
pretrained model is named "hey_jarvis" and is trained specifically on the
phrase **"hey jarvis"** — a bare "Jarvis" is NOT what this model was trained
to detect and will likely under-trigger on it. There is no openWakeWord
pretrained model for a bare "Jarvis". This is a deliberate, known deviation
from spec.md; the orchestrator/personality layer should adjust any spoken
prompts, docs, or user-facing copy to say "Скажите «Хей, Джарвис»" (or
equivalent) rather than a bare "Jarvis", since that is what will actually
work with this listener. (Training a fully custom "jarvis"-only model was
out of scope here — see spec.md's "вне scope" section, which already
excludes custom wake word training.)

Listens to the default microphone continuously and invokes a callback as
soon as the "hey_jarvis" wake phrase is detected. Runs a blocking loop
(`WakeWordListener.run`) meant to live in its own thread, and stops cleanly
via `threading.Event` (see `WakeWordListener.stop`).

No access key / config.yaml "picovoice.*" section is needed anymore.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

# openWakeWord's audio pipeline is built around 16kHz mono int16 PCM,
# processed in 80ms (1280-sample) chunks — this is both the recommended
# and most efficient chunk size (see openWakeWord README / examples).
SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 1280
DEFAULT_MODEL_NAME = "hey_jarvis"
DEFAULT_THRESHOLD = 0.5
DEFAULT_DEBOUNCE_SEC = 3.0
# Windows only ships onnxruntime as an openWakeWord dependency (no modern
# tflite-runtime support there per the openWakeWord README), so "onnx" is
# the framework that actually works on this project's target OS.
DEFAULT_INFERENCE_FRAMEWORK = "onnx"


def _load_yaml_config() -> dict:
    """Best-effort load of config/config.yaml. Never raises."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        import yaml  # PyYAML is an optional dependency for this module
    except ImportError:
        logger.warning(
            "PyYAML не установлен — config.yaml игнорируется, "
            "используются значения по умолчанию."
        )
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        logger.exception(
            "Не удалось прочитать %s — используются значения по умолчанию.",
            _CONFIG_PATH,
        )
        return {}


class WakeWordListener:
    """Continuously listens for the "hey jarvis" wake phrase via openWakeWord."""

    def __init__(
        self,
        on_wake: Callable[[], None],
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
        debounce_sec: Optional[float] = None,
        inference_framework: Optional[str] = None,
        chunk_size: Optional[int] = None,
        input_device: Optional[int] = None,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Args:
            on_wake: called (with no arguments) each time the wake phrase
                fires. Exceptions raised inside it are logged and swallowed
                so the listener keeps running.
            on_ready: called once, right after the model is loaded AND the
                microphone input stream has actually started successfully —
                i.e. proof the listener is really up and listening, not just
                that the background thread launched. `orchestrator.py` uses
                this to play an audible confirmation, since a failure inside
                `run()` (missing packages, no mic, model download failure)
                otherwise dies silently in a background thread with nothing
                user-visible. Exceptions inside it are logged and swallowed.
            model_name: openWakeWord pretrained model name to load. Defaults
                to config.yaml's `wake.model_name`, falling back to
                "hey_jarvis" (the only relevant pretrained Jarvis-ish model —
                see the module docstring about the "hey jarvis" vs "Jarvis"
                phrase mismatch).
            threshold: score in [0, 1] above which a frame counts as a
                detection. Defaults to config.yaml's `wake.threshold`,
                falling back to 0.5 (openWakeWord's own recommended default).
            debounce_sec: minimum seconds between two detections, to avoid
                firing on_wake multiple times for a single spoken utterance.
                Defaults to config.yaml's `wake.debounce_sec`, falling back
                to 3.0.
            inference_framework: "onnx" or "tflite". Defaults to
                config.yaml's `wake.inference_framework`, falling back to
                "onnx" (the framework openWakeWord actually supports on
                Windows).
            chunk_size: audio frame size in samples fed to the model per
                prediction call. Defaults to config.yaml's `wake.chunk_size`,
                falling back to 1280 (80ms at 16kHz — openWakeWord's
                recommended chunk size).
            input_device: sounddevice input device index; None = system default.
        """
        self._on_wake = on_wake
        self._on_ready = on_ready
        self._input_device = input_device
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        wake_cfg = _load_yaml_config().get("wake") or {}
        self._model_name = model_name if model_name is not None else wake_cfg.get(
            "model_name", DEFAULT_MODEL_NAME
        )
        self._threshold = (
            threshold if threshold is not None else float(wake_cfg.get("threshold", DEFAULT_THRESHOLD))
        )
        self._debounce_sec = (
            debounce_sec
            if debounce_sec is not None
            else float(wake_cfg.get("debounce_sec", DEFAULT_DEBOUNCE_SEC))
        )
        self._inference_framework = inference_framework or wake_cfg.get(
            "inference_framework", DEFAULT_INFERENCE_FRAMEWORK
        )
        self._chunk_size = chunk_size or int(wake_cfg.get("chunk_size", DEFAULT_CHUNK_SIZE))

    @property
    def stop_event(self) -> threading.Event:
        """Exposed so callers can also signal shutdown externally."""
        return self._stop_event

    def start(self) -> None:
        """Start listening in a background daemon thread (non-blocking)."""
        if self._thread and self._thread.is_alive():
            logger.warning("WakeWordListener уже запущен.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name="wake-word-listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener to stop and wait for the thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def run(self) -> None:
        """Blocking loop: listens until `stop()` is called (or `stop_event` is set).

        Raises RuntimeError / ImportError / OSError on unrecoverable setup
        failures (missing packages, missing/undownloadable model weights,
        missing microphone) — callers (orchestrator.py) are expected to catch
        these and decide how to degrade (e.g. fall back to text-only mode),
        since a wake-word listener that cannot access a microphone or model
        cannot function at all.
        """
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError:
            logger.exception(
                "Пакет openwakeword не установлен. Установите: pip install openwakeword"
            )
            raise

        try:
            import sounddevice as sd
        except ImportError:
            logger.exception("Пакет sounddevice не установлен. Установите: pip install sounddevice")
            raise

        # One-time (per machine) download of the pretrained model weights from
        # GitHub release assets. No access key / account needed. Safe to call
        # on every startup: it skips files that already exist locally.
        try:
            openwakeword.utils.download_models(model_names=[self._model_name])
        except Exception:
            logger.warning(
                "Не удалось скачать/обновить веса модели openWakeWord '%s' "
                "(возможно, нет подключения к интернету при первом запуске). "
                "Если модель уже была скачана ранее, работа продолжится с "
                "локальной копией.",
                self._model_name,
                exc_info=True,
            )

        try:
            model = Model(
                wakeword_models=[self._model_name],
                inference_framework=self._inference_framework,
            )
        except Exception:
            logger.exception(
                "Не удалось загрузить модель openWakeWord '%s' (framework=%s). "
                "Убедитесь, что веса модели скачаны (нужен интернет при первом "
                "запуске — pip-пакет и сам код доступа/ключа не требуют).",
                self._model_name,
                self._inference_framework,
            )
            raise

        audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.debug("Audio input status: %s", status)
            audio_queue.put(indata.copy())

        stream = None
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=self._chunk_size,
                dtype="int16",
                channels=1,
                device=self._input_device,
                callback=_callback,
            )
            stream.start()
            logger.info(
                "Wake word listener запущен (openWakeWord, модель='%s', порог=%.2f). "
                "Ожидаю фразу «hey jarvis»...",
                self._model_name,
                self._threshold,
            )
            if self._on_ready is not None:
                try:
                    self._on_ready()
                except Exception:
                    logger.exception("Ошибка в обработчике on_ready() — продолжаю прослушивание.")

            while not self._stop_event.is_set():
                try:
                    frame = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                pcm = frame.reshape(-1)
                if pcm.size == 0:
                    continue

                try:
                    predictions = model.predict(
                        pcm,
                        threshold={self._model_name: self._threshold},
                        debounce_time=self._debounce_sec,
                    )
                except Exception:
                    logger.exception("Ошибка в model.predict() — пропускаю кадр.")
                    continue

                score = predictions.get(self._model_name, 0.0)
                if score >= self._threshold:
                    logger.info("Wake word '%s' обнаружено (score=%.3f).", self._model_name, score)
                    try:
                        self._on_wake()
                    except Exception:
                        logger.exception("Ошибка в обработчике on_wake() — продолжаю прослушивание.")
        except OSError:
            logger.exception(
                "Не удалось открыть аудио-устройство ввода (микрофон). "
                "Проверьте подключение микрофона и права доступа приложения к нему."
            )
            raise
        except Exception:
            logger.exception("Неожиданная ошибка в wake word listener.")
            raise
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    logger.exception("Ошибка при закрытии аудио-потока.")
            logger.info("Wake word listener остановлен.")
