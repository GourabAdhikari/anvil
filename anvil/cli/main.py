"""Typer entry point for Anvil's text, chat, and voice interfaces."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import typer

from anvil.brain import router
from anvil.voice import stt, tts, wake_word

app = typer.Typer(
    name="anvil",
    help="Anvil, a text-controlled developer agent.",
    no_args_is_help=True,
)

_COMMAND_SECONDS = 4.0
_last_llm_seconds: float | None = None


@app.callback()
def _main() -> None:
    """Anvil command-line interface."""


def _log(event: str, *, force: bool = False, **details: Any) -> None:
    if not force and event != "error" and os.getenv("ANVIL_DEBUG") != "1":
        return
    typer.echo(json.dumps({"event": event, **details}, default=str))


def _print_help() -> None:
    typer.echo("Commands: help, clear, exit")
    typer.echo("Use chat mode to type commands or voice mode to press Enter and speak.")


def _record_command() -> dict[str, Any]:
    audio_path: Path | None = None
    _log("recording_started")
    try:
        frames = int(_COMMAND_SECONDS * 16_000)
        audio = wake_word._record_chunk(16_000, frames)
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        _log("recording_finished")

    if not audio:
        _log("no_speech_detected")
        return {"success": False, "error": "no_speech_detected"}

    try:
        audio_path = wake_word._write_wav(audio, 16_000)
        _log("transcribing")
        started = time.perf_counter()
        result = stt.transcribe_audio(str(audio_path))
        result["stt_seconds"] = round(time.perf_counter() - started, 3)
        return result
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)


def _speak_response(response: str) -> dict[str, Any]:
    temporary = tempfile.NamedTemporaryFile(prefix="anvil-response-", suffix=".wav", delete=False)
    temporary.close()
    output = Path(temporary.name)
    result = tts.synthesize(response, str(output))
    if not result.get("success"):
        output.unlink(missing_ok=True)
    return result


def _play_audio(path: str) -> dict[str, Any]:
    players = {
        "Darwin": ["afplay"],
        "Linux": ["paplay"],
        "Windows": ["powershell", "-NoProfile", "-Command", "(New-Object Media.SoundPlayer $args[0]).PlaySync()", path],
    }
    command = players.get(__import__("platform").system())
    if command is None:
        return {"played": False, "command": [], "returncode": None, "error": "Unsupported platform"}
    if command[-1] != path:
        command = [*command, path]
    if shutil.which(command[0]) is None:
        return {"played": False, "command": command, "returncode": None, "error": f"Playback command not found: {command[0]}"}
    try:
        completed = subprocess.run(command, check=False)
        return {"played": completed.returncode == 0, "command": command, "returncode": completed.returncode}
    except OSError as exc:
        return {"played": False, "command": command, "returncode": None, "error": str(exc)}


def _route_command(command: str) -> str | None:
    global _last_llm_seconds
    _log("generating_response")
    started = time.perf_counter()
    try:
        response = router.run(command)
    except Exception as exc:
        _last_llm_seconds = round(time.perf_counter() - started, 3)
        _log("error", error=str(exc))
        return None
    _last_llm_seconds = round(time.perf_counter() - started, 3)

    try:
        from anvil.memory.store import recall, remember
        previous = recall("conversation_history")
        history = previous.get("value", []) if previous.get("found") else []
        if not isinstance(history, list):
            history = []
        history.append({"user": command, "assistant": response})
        remember("conversation_history", history[-50:])
    except Exception:
        pass
    return response


def _voice_speech_text(response: str, *, max_sentences: int = 3) -> str:
    """Create speech from complete sentences without cutting phrases."""
    without_code = re.sub(r"```.*?```", " ", response, flags=re.DOTALL)
    lines = []
    for line in without_code.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line)
        if line.strip():
            lines.append(line.strip())
    plain = tts.clean_for_speech(" ".join(lines))
    sentences = [part.strip() for part in re.findall(r".*?[.!?](?=\s|$)", plain) if part.strip()]
    selected = " ".join(sentences[:max_sentences]).strip()
    if not selected and plain:
        selected = plain.rstrip(".!?").strip() + "."
    elif selected and selected[-1] not in ".!?":
        selected += "."
    return selected


def _speak_generated_response(
    response: str,
    *,
    optimize_for_voice: bool = False,
    lifecycle_events: bool = False,
    stt_seconds: float | None = None,
) -> None:
    speech_text = _voice_speech_text(response) if optimize_for_voice else response
    if optimize_for_voice:
        _log("speech_text", text=speech_text)
        _log("speech_text_length", words=len(speech_text.split()))
    if lifecycle_events:
        _log("speaking_started", force=True)
    tts_started = time.perf_counter()
    spoken = _speak_response(speech_text)
    tts_seconds = round(time.perf_counter() - tts_started, 3)
    if not spoken.get("success"):
        _log("error", error=spoken.get("error", "Response speech failed."))
        if lifecycle_events:
            _log("speaking_finished", force=True, success=False)
        return

    output_path = Path(spoken["output_path"])
    file_size = output_path.stat().st_size if output_path.exists() else 0
    playback_started = time.perf_counter()
    playback = _play_audio(str(output_path))
    playback_seconds = round(time.perf_counter() - playback_started, 3)
    _log(
        "response_spoken",
        output_path=str(output_path),
        file_size=file_size,
        playback_command=playback.get("command"),
        playback_returncode=playback.get("returncode"),
        played=playback.get("played"),
    )
    if lifecycle_events:
        _log(
            "latency",
            force=True,
            stt_seconds=stt_seconds,
            llm_seconds=_last_llm_seconds,
            tts_seconds=tts_seconds,
            playback_seconds=playback_seconds,
        )
        _log("speaking_finished", force=True, success=bool(playback.get("played")), **playback)


@app.command("memory")
def memory_command(
    action: str = typer.Argument("list", help="list, remember, search, or clear"),
    value: str | None = typer.Argument(None, help="Statement to remember or search query"),
) -> None:
    """List, remember, search, or clear persistent local memories."""
    try:
        from anvil.memory.store import clear_memories, list_memories, remember_statement, search_memories
        if action == "list":
            result = list_memories()
        elif action == "remember" and value:
            result = remember_statement(value)
        elif action == "search" and value:
            result = search_memories(value)
        elif action == "clear":
            result = clear_memories()
        else:
            result = {"success": False, "error": "Usage: memory list | memory remember <statement> | memory search <query> | memory clear"}
        typer.echo(json.dumps(result, indent=2, default=str))
    except Exception as exc:
        _log("error", error=str(exc))


@app.command("tts-test")
def tts_test_command() -> None:
    """Validate local TTS synthesis and playback without LLM or STT calls."""
    try:
        tts.initialize()
        for text in ("Hello Gourab.", "Hello Gourab. I am Anvil."):
            started = time.perf_counter()
            result = _speak_response(text)
            elapsed = time.perf_counter() - started
            if not result.get("success"):
                _log("error", error=result.get("error", "TTS synthesis failed."), text=text)
                continue
            output_path = Path(result["output_path"])
            playback = _play_audio(str(output_path))
            _log(
                "tts_test",
                force=True,
                text=text,
                output_path=str(output_path),
                file_size=output_path.stat().st_size if output_path.exists() else 0,
                synthesis_seconds=round(elapsed, 3),
                audio_duration_seconds=result.get("duration"),
                playback_command=playback.get("command"),
                playback_returncode=playback.get("returncode"),
                played=playback.get("played"),
                playback_error=playback.get("error"),
            )
    except Exception as exc:
        _log("error", error=str(exc))


@app.command("run")
def run_command(
    command: str = typer.Argument(..., help="The developer command to send to Anvil."),
) -> None:
    """Send a single text command to Anvil's shared brain router."""
    response = _route_command(command)
    if response:
        typer.echo(response)


@app.command("chat")
def chat_command(
    tts_enabled: bool = typer.Option(False, "--tts", help="Synthesize and play each response."),
) -> None:
    """Start an interactive text chat using the shared router and tools."""
    if tts_enabled:
        try:
            tts.initialize()
        except Exception as exc:
            _log("error", error=str(exc))
    typer.echo("Anvil chat. Type help for commands.")
    while True:
        try:
            command = typer.prompt("You", prompt_suffix=" > ")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            return

        normalized = command.strip().casefold()
        if normalized == "exit":
            return
        if normalized == "help":
            _print_help()
            continue
        if normalized == "clear":
            typer.clear()
            continue
        if not command.strip():
            continue

        response = _route_command(command)
        if response:
            typer.echo("Anvil >")
            typer.echo(response)
            if tts_enabled:
                _speak_generated_response(response)


@app.command("voice")
def voice_command() -> None:
    """Start direct voice mode; press Enter to record each command."""
    try:
        tts.initialize()
    except Exception as exc:
        _log("error", error=str(exc))
    typer.echo("Anvil voice mode. Press Enter to record, or type help, clear, or exit.")
    try:
        while True:
            action = typer.prompt("Press Enter to record", default="")
            normalized = action.strip().casefold()
            if normalized == "exit":
                return
            if normalized == "help":
                _print_help()
                continue
            if normalized == "clear":
                typer.clear()
                continue
            if normalized:
                typer.echo("Press Enter without text to record a command.")
                continue

            command = _record_command()
            if not command.get("success"):
                if command.get("error") != "no_speech_detected":
                    _log("error", error=command.get("error", "Command transcription failed."))
                continue
            text = str(command.get("text", "")).strip()
            _log("command_transcribed", text=text)
            if not text:
                _log("error", error="No speech was transcribed.")
                continue
            response = _route_command(text)
            if response:
                _log("response_generated", text=response)
                _speak_generated_response(
                    response,
                    optimize_for_voice=True,
                    lifecycle_events=True,
                    stt_seconds=command.get("stt_seconds"),
                )
    except KeyboardInterrupt:
        _log("shutdown")


@app.command("listen")
def listen_command() -> None:
    """Wait for Jarvis, then process voice commands continuously."""
    try:
        tts.initialize()
    except Exception as exc:
        _log("error", error=str(exc))
    stop_event = threading.Event()
    try:
        while not stop_event.is_set():
            wake_events = wake_word.listen(stop_event=stop_event)
            try:
                for event in wake_events:
                    if event.get("event") == "wake_word_detected":
                        _log("wake_word_detected", **{key: value for key, value in event.items() if key != "event"})
                        wake_events.close()
                        command = _record_command()
                        if not command.get("success"):
                            if command.get("error") != "no_speech_detected":
                                _log("error", error=command.get("error", "Command transcription failed."))
                            break
                        text = str(command.get("text", "")).strip()
                        _log("command_transcribed", text=text)
                        response = _route_command(text)
                        if response:
                            _log("response_generated", text=response)
                            _speak_generated_response(
                                response,
                                optimize_for_voice=True,
                                lifecycle_events=True,
                                stt_seconds=command.get("stt_seconds"),
                            )
                        break
                    elif event.get("event") == "error":
                        _log("error", error=event.get("error", "Wake-word listener failed."))
                    elif event.get("event") == "transcript":
                        _log("transcript", text=event.get("text", ""))
            finally:
                wake_events.close()
    except KeyboardInterrupt:
        stop_event.set()
        _log("shutdown")
    except Exception as exc:
        _log("error", error=str(exc))
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
