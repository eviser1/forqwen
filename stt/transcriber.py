"""Speech-to-text for Jarvis: microphone capture + faster-whisper.

Two entry points:
    record_until_silence() -> np.ndarray   Record from the mic until the user
                                            stops talking (simple RMS-based
                                            VAD, no external VAD dependency).
    transcribe(audio) -> str               Run faster-whisper over the
                                            recorded audio, Russian by default.

GPU is preferred (device="cuda") for faster-whisper; if CUDA is unavailable
or fails to initialize, this module logs the failure and transparently falls
back to CPU rather than crashing the process.
"""

from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

# (model_size, requested_device) -> (WhisperModel instance, actual_device_used)
_model_cache: dict[tuple[str, str], tuple[object, str]] = {}


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


def _stt_config() -> dict:
    return _load_yaml_config().get("stt") or {}


def record_until_silence(
    sample_rate: Optional[int] = None,
    silence_timeout_sec: Optional[float] = None,
    max_record_sec: Optional[float] = None,
    rms_threshold: Optional[float] = None,
    pre_speech_timeout_sec: Optional[float] = None,
    input_device: Optional[int] = None,
) -> np.ndarray:
    """Record mono int16 PCM from the microphone until a pause is detected.

    Simple amplitude-based VAD: once the RMS of a block crosses
    `rms_threshold`, speech is considered "started"; recording stops once
    `silence_timeout_sec` of continuous below-threshold audio follows. Also
    bounded by `max_record_sec` (safety cap) and `pre_speech_timeout_sec`
    (give up if the user never starts speaking).

    Unset parameters fall back to config.yaml's `stt` section, then to
    hardcoded defaults. Returns an empty array if no audio device is
    available or nothing was captured.
    """
    cfg = _stt_config()
    sample_rate = sample_rate or int(cfg.get("sample_rate", 16000))
    silence_timeout_sec = silence_timeout_sec or float(cfg.get("silence_timeout_sec", 1.2))
    max_record_sec = max_record_sec or float(cfg.get("max_record_sec", 15.0))
    rms_threshold = rms_threshold or float(cfg.get("vad_rms_threshold", 500.0))
    pre_speech_timeout_sec = pre_speech_timeout_sec or float(cfg.get("pre_speech_timeout_sec", 5.0))

    try:
        import sounddevice as sd
    except ImportError:
        logger.exception("Пакет sounddevice не установлен. Установите: pip install sounddevice")
        raise

    block_duration = 0.05  # 50 ms blocks
    block_size = max(1, int(sample_rate * block_duration))

    frames: list[np.ndarray] = []
    speech_started = False
    silence_elapsed = 0.0
    elapsed = 0.0

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def _callback(indata: np.ndarray, frame_count: int, time_info, status) -> None:
        if status:
            logger.debug("Audio input status: %s", status)
        audio_queue.put(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            dtype="int16",
            channels=1,
            device=input_device,
            callback=_callback,
        )
    except OSError:
        logger.exception(
            "Не удалось открыть микрофон для записи речи. "
            "Проверьте подключение микрофона и права доступа."
        )
        return np.zeros(0, dtype=np.int16)
    except Exception:
        logger.exception("Неожиданная ошибка при открытии аудио-потока для записи.")
        return np.zeros(0, dtype=np.int16)

    logger.info(
        "Начинаю запись речи (rms_threshold=%.0f, silence_timeout=%.1fs, max=%.1fs)...",
        rms_threshold,
        silence_timeout_sec,
        max_record_sec,
    )

    try:
        with stream:
            while True:
                try:
                    block = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    logger.warning("Нет данных с микрофона в течение 1с — прекращаю запись.")
                    break

                pcm = block.reshape(-1)
                frames.append(pcm)
                rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if pcm.size else 0.0
                elapsed += block_duration

                if rms >= rms_threshold:
                    speech_started = True
                    silence_elapsed = 0.0
                elif speech_started:
                    silence_elapsed += block_duration

                if speech_started and silence_elapsed >= silence_timeout_sec:
                    logger.debug("Обнаружена тишина после речи — останавливаю запись.")
                    break
                if not speech_started and elapsed >= pre_speech_timeout_sec:
                    logger.warning("Речь не обнаружена в течение %.1fs.", pre_speech_timeout_sec)
                    break
                if elapsed >= max_record_sec:
                    logger.warning("Достигнут лимит записи %.1fs.", max_record_sec)
                    break
    except Exception:
        logger.exception("Ошибка во время записи речи.")

    if not frames:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(frames)


def _load_model(model_size: str, requested_device: str) -> tuple[object, str]:
    """Load (and cache) a faster-whisper model, falling back CUDA -> CPU."""
    cache_key = (model_size, requested_device)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.exception(
            "Пакет faster-whisper не установлен. Установите: pip install faster-whisper"
        )
        raise

    device = requested_device
    compute_type = "float16" if device == "cuda" else "int8"

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info(
            "faster-whisper модель '%s' загружена на устройстве '%s' (compute_type=%s).",
            model_size,
            device,
            compute_type,
        )
    except Exception:
        if device != "cpu":
            logger.exception(
                "Не удалось загрузить faster-whisper на '%s' (вероятно, CUDA/GPU недоступна "
                "или несовместима). Переключаюсь на CPU.",
                device,
            )
            device = "cpu"
            compute_type = "int8"
            try:
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
                logger.info(
                    "faster-whisper модель '%s' загружена на CPU (fallback, compute_type=%s).",
                    model_size,
                    compute_type,
                )
            except Exception:
                logger.exception("Не удалось загрузить faster-whisper даже на CPU.")
                raise
        else:
            logger.exception("Не удалось загрузить faster-whisper на CPU.")
            raise

    result = (model, device)
    _model_cache[cache_key] = result
    return result


def transcribe(
    audio: np.ndarray,
    language: str = "ru",
    model_size: Optional[str] = None,
    device: Optional[str] = None,
) -> str:
    """Transcribe int16 PCM `audio` (mono, sample rate matching recording) to text.

    `model_size`/`device` default to config.yaml's `stt.model_size` /
    `stt.device`, falling back to "small" / "cuda". On any failure to load or
    run the requested device, falls back to CPU automatically (logged).
    Returns "" for empty input or on unrecoverable transcription failure
    (never raises for a normal empty/silent recording).
    """
    if audio is None or getattr(audio, "size", 0) == 0:
        logger.warning("transcribe() получил пустой аудио-буфер — распознавание пропущено.")
        return ""

    cfg = _stt_config()
    model_size = model_size or cfg.get("model_size", "small")
    device = device or cfg.get("device", "cuda")

    try:
        model, actual_device = _load_model(model_size, device)
    except Exception:
        logger.exception("STT недоступен — не удалось загрузить модель распознавания.")
        return ""

    audio_float = audio.astype(np.float32) / 32768.0

    try:
        segments, _info = model.transcribe(audio_float, language=language, beam_size=5)
        text = "".join(segment.text for segment in segments).strip()
    except Exception:
        logger.exception("Ошибка при распознавании речи (устройство=%s).", actual_device)
        return ""

    logger.info("Распознано (lang=%s, device=%s): %s", language, actual_device, text)
    return text
