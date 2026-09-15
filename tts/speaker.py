"""Text-to-speech for Jarvis: Silero TTS (Russian) + playback with barge-in.

    speak(text, interrupt_event)   Synthesize `text` with Silero and play it
                                    back; if `interrupt_event` is set while
                                    playing, playback stops immediately.
    play_wake_chime()              Short synthetic beep (numpy sine, no audio
                                    file needed) played right after the wake
                                    word fires, before recording the user's
                                    speech — separate feedback channel from TTS.

GPU is preferred for Silero; if CUDA is unavailable or fails, this module
logs the failure and falls back to CPU rather than crashing the process.
Missing audio output devices are also logged and swallowed (speak() simply
does not produce sound rather than raising).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

# cache key f"{model_id}:{requested_device}" -> (model, actual_device_used)
_model_cache: dict[str, tuple[object, str]] = {}


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


def _tts_config() -> dict:
    cfg = _load_yaml_config().get("tts") or {}
    return {
        "model_id": cfg.get("model_id", "v4_ru"),
        "speaker": cfg.get("speaker", "xenia"),
        "sample_rate": int(cfg.get("sample_rate", 48000)),
        "device": cfg.get("device", "cuda"),
    }


def _load_model(model_id: str, requested_device: str) -> tuple[object, str]:
    """Load (and cache) the Silero TTS model, falling back CUDA -> CPU."""
    cache_key = f"{model_id}:{requested_device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        import torch
    except ImportError:
        logger.exception(
            "Пакет torch не установлен. Установите torch (и torchaudio при необходимости) "
            "для Silero TTS."
        )
        raise

    device_str = requested_device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA недоступна для Silero TTS (torch.cuda.is_available() == False) — использую CPU.")
        device_str = "cpu"

    def _load(dev: str):
        model, _example_texts = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=model_id,
        )
        model.to(torch.device(dev))
        return model

    try:
        model = _load(device_str)
        logger.info("Silero TTS модель '%s' загружена на устройстве '%s'.", model_id, device_str)
    except Exception:
        if device_str != "cpu":
            logger.exception(
                "Не удалось загрузить Silero TTS на '%s' — переключаюсь на CPU.", device_str
            )
            device_str = "cpu"
            try:
                model = _load(device_str)
                logger.info("Silero TTS модель '%s' загружена на CPU (fallback).", model_id)
            except Exception:
                logger.exception("Не удалось загрузить Silero TTS даже на CPU.")
                raise
        else:
            logger.exception("Не удалось загрузить Silero TTS на CPU.")
            raise

    result = (model, device_str)
    _model_cache[cache_key] = result
    return result


def speak(text: str, interrupt_event: Optional[threading.Event] = None) -> None:
    """Synthesize `text` (Russian) with Silero and play it back.

    If `interrupt_event` is provided and gets set while audio is playing,
    playback is stopped immediately (barge-in). Never raises: any failure
    (missing packages, no GPU, no audio output device) is logged and this
    function simply returns without producing sound.
    """
    if not text or not text.strip():
        logger.debug("speak() вызван с пустым текстом — пропускаю.")
        return

    cfg = _tts_config()

    try:
        model, actual_device = _load_model(cfg["model_id"], cfg["device"])
    except Exception:
        logger.exception("TTS недоступен — не удалось синтезировать речь: '%s'", text)
        return

    try:
        import torch
    except ImportError:
        logger.exception("Пакет torch недоступен при синтезе речи.")
        return

    try:
        with torch.no_grad():
            audio_tensor = model.apply_tts(
                text=text,
                speaker=cfg["speaker"],
                sample_rate=cfg["sample_rate"],
            )
        audio = audio_tensor.detach().cpu().numpy().astype(np.float32)
    except Exception:
        logger.exception("Ошибка синтеза речи Silero (устройство=%s, текст='%s').", actual_device, text)
        return

    _play_audio(audio, cfg["sample_rate"], interrupt_event)


def _play_audio(audio: np.ndarray, sample_rate: int, interrupt_event: Optional[threading.Event]) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        logger.exception("Пакет sounddevice не установлен — не удаётся воспроизвести речь.")
        return

    try:
        sd.play(audio, samplerate=sample_rate)
    except Exception:
        logger.exception("Не удалось воспроизвести аудио — проверьте аудио-устройство вывода.")
        return

    try:
        while True:
            if interrupt_event is not None and interrupt_event.is_set():
                sd.stop()
                logger.info("Воспроизведение речи прервано (barge-in).")
                return
            stream = sd.get_stream()
            if stream is None or not stream.active:
                break
            time.sleep(0.05)
    except Exception:
        logger.exception("Ошибка во время воспроизведения речи.")
        try:
            sd.stop()
        except Exception:
            pass


def _synth_tone(frequency_hz: float, duration_sec: float, sample_rate: int, volume: float) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    tone = np.sin(2 * np.pi * frequency_hz * t)

    fade_len = max(1, int(sample_rate * 0.01))  # 10ms fade in/out to avoid clicks
    envelope = np.ones_like(tone)
    envelope[:fade_len] = np.linspace(0.0, 1.0, fade_len)
    envelope[-fade_len:] = np.linspace(1.0, 0.0, fade_len)

    return (tone * envelope * volume).astype(np.float32)


def play_wake_chime(
    frequency_hz: float = 880.0,
    duration_sec: float = 0.15,
    sample_rate: int = 44100,
    volume: float = 0.3,
) -> None:
    """Play a short synthetic beep (pure sine, generated in-process — no audio
    file) to acknowledge wake word detection, before recording user speech.

    Logged and swallowed on any audio device failure, never raises.
    """
    try:
        import sounddevice as sd
    except ImportError:
        logger.exception("Пакет sounddevice не установлен — не удаётся воспроизвести сигнал пробуждения.")
        return

    try:
        chime = _synth_tone(frequency_hz, duration_sec, sample_rate, volume)
        sd.play(chime, samplerate=sample_rate)
        sd.wait()
    except Exception:
        logger.exception("Не удалось воспроизвести звуковой сигнал пробуждения.")


def play_ready_chime(sample_rate: int = 44100, volume: float = 0.25) -> None:
    """Two short ascending beeps, played once when the wake-word listener has
    actually started (model loaded + microphone stream open) — distinct from
    play_wake_chime() so the user can tell "I'm alive and listening" apart
    from "I just heard the wake word". Logged and swallowed on failure.
    """
    try:
        import sounddevice as sd
    except ImportError:
        logger.exception("Пакет sounddevice не установлен — не удаётся воспроизвести сигнал готовности.")
        return

    try:
        tone1 = _synth_tone(523.25, 0.09, sample_rate, volume)  # C5
        gap = np.zeros(int(sample_rate * 0.04), dtype=np.float32)
        tone2 = _synth_tone(783.99, 0.12, sample_rate, volume)  # G5
        sd.play(np.concatenate([tone1, gap, tone2]), samplerate=sample_rate)
        sd.wait()
    except Exception:
        logger.exception("Не удалось воспроизвести звуковой сигнал готовности.")
