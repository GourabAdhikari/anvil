"""Typer entry point for Anvil's text, chat, and voice interfaces."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
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


@app.callback()
def _main() -> None:
    """Anvil command-line interface."""


def _log(event: str, **details: Any) -> None:
    typer.echo(json.dumps({"event": event, **details}, default=str))


def _print_help() -> None:
    typer.echo("Commands: help, clear, exit")
    typer.echo("Use chat mode to type commands or voice mode to press Enter and speak.")


def _record_command() -> dict[str, Any]:
    audio_path: Path | None = None
    try:
        frames = int(_COMMAND_SECONDS * 16_000)
        audio = wake_word._record_chunk(16_000, frames)
        if not audio:
            return {"success": False, "error": "No command audio was recorded."}
        audio_path = wake_word._write_wav(audio, 16_000)
        return stt.transcribe_audio(str(audio_path))
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


def _play_audio(path: str) -> bool:
    players = {
        "Darwin": ["afplay"],
        "Linux": ["paplay"],
        "Windows": ["powershell", "-NoProfile", "-Command", "(New-Object Media.SoundPlayer $args[0]).PlaySync()", path],
    }
    command = players.get(__import__("platform").system())
    if command is None:
        return False
    if command[-1] != path:
        command = [*command, path]
    if shutil.which(command[0]) is None:
        return False
    try:
        return subprocess.run(command, check=False).returncode == 0
    except OSError:
        return False


def _route_command(command: str) -> str | None:
    try:
        return router.run(command)
    except Exception as exc:
        _log("error", error=str(exc))
        return None


def _speak_generated_response(response: str) -> None:
    spoken = _speak_response(response)
    if not spoken.get("success"):
        _log("error", error=spoken.get("error", "Response speech failed."))
    else:
        played = _play_audio(spoken["output_path"])
        _log("response_spoken", output_path=spoken.get("output_path"), played=played)


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
    typer.echo("Anvil chat. Type help for commands.")
    while True:
        try:
            command = typer.prompt("You", prompt_suffix=" >")
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
                _speak_generated_response(response)
    except KeyboardInterrupt:
        _log("shutdown")


@app.command("listen")
def listen_command() -> None:
    """Wait for Jarvis, then process voice commands continuously."""
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
                            _log("error", error=command.get("error", "Command transcription failed."))
                            break
                        text = str(command.get("text", "")).strip()
                        _log("command_transcribed", text=text)
                        response = _route_command(text)
                        if response:
                            _log("response_generated", text=response)
                            _speak_generated_response(response)
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
