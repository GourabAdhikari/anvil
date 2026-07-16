"""CPU-only speech-to-text transcription with faster-whisper."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_NAME = "base"


@lru_cache(maxsize=1)
def _load_model(model_name: str = MODEL_NAME) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed") from exc
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_audio(audio_path: str, *, model: Any = None) -> dict[str, Any]:
    """Transcribe an audio file using a CPU-only faster-whisper model."""
    if not isinstance(audio_path, str) or not audio_path.strip():
        return {"success": False, "audio_path": audio_path, "error": "audio_path must be a non-empty path."}

    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        return {"success": False, "audio_path": audio_path, "error": f"Audio file does not exist: {audio_path}"}

    try:
        transcriber = model or _load_model()
        segments, info = transcriber.transcribe(str(path))
        text = " ".join(
            segment.text.strip()
            for segment in segments
            if getattr(segment, "text", "").strip()
        ).strip()
    except Exception as exc:
        return {"success": False, "audio_path": str(path), "error": str(exc)}

    return {
        "success": True,
        "audio_path": str(path),
        "text": text,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
    }


def transcribe(audio_path: str, *, model: Any = None) -> dict[str, Any]:
    """Alias for :func:`transcribe_audio`."""
    return transcribe_audio(audio_path, model=model)


def test_transcribe_rejects_missing_audio() -> None:
    """Basic regression test that avoids loading the speech model."""
    result = transcribe_audio("/path/that/does/not/exist.wav")
    assert result["success"] is False
    assert "does not exist" in result["error"]


__all__ = ["transcribe_audio", "transcribe"]
