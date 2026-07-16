"""CPU-only text-to-speech synthesis with Coqui TTS."""

from __future__ import annotations

import math
import struct
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"


@lru_cache(maxsize=1)
def _load_model(model_name: str = MODEL_NAME) -> Any:
    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise RuntimeError("Coqui TTS is not installed") from exc
    return TTS(model_name=model_name, progress_bar=False, gpu=False)


def _window_rms(raw: bytes, start_frame: int, end_frame: int, sample_width: int, channels: int) -> float:
    """Return normalized RMS for a range of interleaved PCM frames."""
    start = start_frame * channels
    end = end_frame * channels
    if sample_width == 1:
        samples = [(sample - 128) for sample in raw[start:end]]
        scale = 128.0
    elif sample_width == 2:
        samples = struct.unpack(f"<{end - start}h", raw[start * 2:end * 2])
        scale = 32768.0
    elif sample_width == 4:
        samples = struct.unpack(f"<{end - start}i", raw[start * 4:end * 4])
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / scale


def _trim_trailing_audio(path: Path, *, buffer_ms: int = 300, window_ms: int = 20) -> dict[str, float]:
    """Trim low-energy audio after the final speech-containing window."""
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError("Only uncompressed PCM WAV files can be trimmed")
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        frame_rate = source.getframerate()
        frame_count = source.getnframes()
        raw = source.readframes(frame_count)
        parameters = source.getparams()

    if frame_count == 0 or frame_rate <= 0:
        return {"original_duration": 0.0, "duration": 0.0}

    window_frames = max(1, int(frame_rate * window_ms / 1000))
    rms_values = [
        _window_rms(raw, start, min(start + window_frames, frame_count), sample_width, channels)
        for start in range(0, frame_count, window_frames)
    ]
    peak = max(rms_values, default=0.0)
    threshold = max(0.015, peak * 0.1)
    speech_windows = [index for index, rms in enumerate(rms_values) if rms >= threshold]
    if not speech_windows:
        return {
            "original_duration": frame_count / frame_rate,
            "duration": frame_count / frame_rate,
        }

    final_window = speech_windows[-1]
    speech_end = min((final_window + 1) * window_frames, frame_count)
    buffer_frames = int(frame_rate * buffer_ms / 1000)
    keep_frames = min(frame_count, speech_end + buffer_frames)
    if keep_frames < frame_count:
        with wave.open(str(path), "wb") as destination:
            destination.setparams(parameters)
            frame_size = channels * sample_width
            destination.writeframes(raw[: keep_frames * frame_size])

    return {
        "original_duration": frame_count / frame_rate,
        "duration": keep_frames / frame_rate,
    }


def synthesize(text: str, output_path: str, *, model: Any = None) -> dict[str, Any]:
    """Generate a speech audio file from ``text`` using CPU inference."""
    if not isinstance(text, str) or not text.strip():
        return {"success": False, "output_path": output_path, "error": "text must be a non-empty string."}
    if not isinstance(output_path, str) or not output_path.strip():
        return {"success": False, "error": "output_path must be a non-empty path."}

    path = Path(output_path).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        synthesizer = model if model is not None else _load_model()
        synthesizer.tts_to_file(text=text.strip(), file_path=str(path), split_sentences=False)
        duration = _trim_trailing_audio(path)
    except Exception as exc:
        return {"success": False, "output_path": str(path), "error": str(exc)}

    return {
        "success": True,
        "output_path": str(path),
        "text": text.strip(),
        "model": MODEL_NAME,
        "device": "cpu",
        **duration,
    }


def text_to_speech(text: str, output_path: str, *, model: Any = None) -> dict[str, Any]:
    """Alias for :func:`synthesize`."""
    return synthesize(text, output_path, model=model)


def test_synthesize_rejects_empty_text() -> None:
    """Basic validation test that does not load the speech model."""
    result = synthesize("", "/tmp/anvil-test.wav")
    assert result["success"] is False
    assert "non-empty" in result["error"]


__all__ = ["synthesize", "text_to_speech"]
