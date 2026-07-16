"""CPU-only STT-based wake-word detection for the Jarvis wake word."""

from __future__ import annotations

import tempfile
import threading
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .stt import transcribe_audio

WAKE_WORD = "jarvis"
_SAMPLE_RATE = 16_000
_CHANNELS = 1
_SAMPLE_WIDTH = 2


def _error(message: str) -> dict[str, Any]:
    return {"event": "error", "error": message}


def _record_chunk(sample_rate: int, frames: int) -> bytes:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is not installed") from exc

    recording = sd.rec(frames, samplerate=sample_rate, channels=_CHANNELS, dtype="int16")
    sd.wait()
    return recording.tobytes()


def _write_wav(audio: bytes, sample_rate: int) -> Path:
    temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temporary.close()
    path = Path(temporary.name)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(_CHANNELS)
        output.setsampwidth(_SAMPLE_WIDTH)
        output.setframerate(sample_rate)
        output.writeframes(audio)
    return path


def listen(
    *,
    stop_event: threading.Event | None = None,
    model: Any = None,
    audio_source: Callable[[int, int], bytes] | None = None,
    transcriber: Callable[..., dict[str, Any]] = transcribe_audio,
    chunk_seconds: float = 1.0,
    overlap_seconds: float = 0.25,
    sample_rate: int = _SAMPLE_RATE,
) -> Iterator[dict[str, Any]]:
    """Continuously transcribe overlapping chunks until shutdown is requested."""
    stop_event = stop_event or threading.Event()
    if chunk_seconds <= 0 or sample_rate <= 0:
        yield _error("chunk_seconds and sample_rate must be positive")
        return
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        yield _error("overlap_seconds must be non-negative and less than chunk_seconds")
        return

    source = audio_source or _record_chunk
    chunk_frames = int(chunk_seconds * sample_rate)
    capture_frames = max(1, chunk_frames - int(overlap_seconds * sample_rate))
    overlap_bytes = int(overlap_seconds * sample_rate) * _CHANNELS * _SAMPLE_WIDTH
    previous_tail = b""
    first_chunk = True
    yield {"event": "listening", "wake_word": WAKE_WORD}

    while not stop_event.is_set():
        audio_path: Path | None = None
        try:
            frames = chunk_frames if first_chunk else capture_frames
            fresh_audio = source(sample_rate, frames)
            first_chunk = False
            if not isinstance(fresh_audio, bytes) or not fresh_audio:
                raise RuntimeError("audio source returned no audio data")

            audio = previous_tail + fresh_audio
            previous_tail = audio[-overlap_bytes:] if overlap_bytes else b""
            audio_path = _write_wav(audio, sample_rate)
            result = transcriber(str(audio_path), model=model)
            if not result.get("success"):
                yield _error(result.get("error", "speech transcription failed"))
                continue

            transcript = str(result.get("text", "")).strip()
            yield {"event": "transcript", "text": transcript}
            if WAKE_WORD in transcript.casefold():
                yield {
                    "event": "wake_word_detected",
                    "wake_word": WAKE_WORD,
                    "transcript": transcript,
                }
        except KeyboardInterrupt:
            return
        except Exception as exc:
            yield _error(str(exc))
        finally:
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)


def listen_for_wake_word(**kwargs: Any) -> Iterator[dict[str, Any]]:
    """Alias for :func:`listen`."""
    return listen(**kwargs)


def test_wake_word_shutdown_without_microphone() -> None:
    """Validation test that stops before accessing microphone dependencies."""
    stop_event = threading.Event()
    stop_event.set()
    events = list(listen(stop_event=stop_event))
    assert events == [{"event": "listening", "wake_word": WAKE_WORD}]


__all__ = ["listen", "listen_for_wake_word", "WAKE_WORD"]
