from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pyttsx3
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


RUSSIAN_VOICE_HINTS = (
    "ru",
    "russian",
    "russia",
    "irina",
    "pavel",
    "elena",
    "dmitry",
    "katya",
    "maria",
    "милена",
    "ирина",
    "павел",
)


def record_wav(
    output_path: str | Path,
    seconds: float = 6.0,
    samplerate: int = 16000,
    channels: int = 1,
) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = sd.rec(
        int(seconds * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
    )
    sd.wait()
    sf.write(str(output_path), audio, samplerate)
    return output_path


def transcribe_question(
    audio_path: str | Path,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str | None = None,
    lang: str | None = "ru",
) -> str:
    if compute_type is None:
        compute_type = "int8" if device == "cpu" else "float16"

    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments, _info = model.transcribe(
        str(audio_path),
        language=lang,
        vad_filter=True,
        beam_size=5,
    )

    text = " ".join((seg.text or "").strip() for seg in segments).strip()
    return re.sub(r"\s+", " ", text)


def _clean_tts_text(text: str) -> str:
    text = text.split("Источники:")[0]
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _voice_text_blob(voice: object) -> str:
    parts: list[str] = []
    for attr in ("id", "name"):
        value = getattr(voice, attr, "")
        if value:
            parts.append(str(value).lower())

    languages = getattr(voice, "languages", None)
    if languages:
        if isinstance(languages, (list, tuple)):
            parts.extend(str(x).lower() for x in languages)
        else:
            parts.append(str(languages).lower())

    return " ".join(parts)


def find_russian_voice_id() -> str | None:
    engine = pyttsx3.init()
    try:
        voices = engine.getProperty("voices") or []
        for voice in voices:
            blob = _voice_text_blob(voice)
            if any(hint in blob for hint in RUSSIAN_VOICE_HINTS):
                return str(getattr(voice, "id"))
        return None
    finally:
        try:
            engine.stop()
        except Exception:
            pass


def list_available_voices() -> list[dict[str, str]]:
    engine = pyttsx3.init()
    try:
        voices = engine.getProperty("voices") or []
        out: list[dict[str, str]] = []
        for voice in voices:
            out.append(
                {
                    "id": str(getattr(voice, "id", "")),
                    "name": str(getattr(voice, "name", "")),
                    "languages": str(getattr(voice, "languages", "")),
                }
            )
        return out
    finally:
        try:
            engine.stop()
        except Exception:
            pass


def speak_text(
    text: str,
    voice_id: str | None = None,
    rate: int = 185,
    volume: float = 1.0,
) -> None:
    text = _clean_tts_text(text)
    if not text:
        return

    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)

        selected_voice_id = voice_id or find_russian_voice_id()
        if not selected_voice_id:
            raise RuntimeError(
                "Не найден русский голос Windows для озвучки. "
                "Установи русский голосовой пакет в Windows или передай voice_id вручную."
            )

        engine.setProperty("voice", selected_voice_id)
        engine.say(text)
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass
