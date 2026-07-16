"""Typer entry point for Anvil's text interface."""

from __future__ import annotations

import typer

from anvil.brain import router

app = typer.Typer(
    name="anvil",
    help="Anvil, a text-controlled developer agent.",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """Anvil command-line interface."""


@app.command("run")
def run_command(
    command: str = typer.Argument(..., help="The developer command to send to Anvil."),
) -> None:
    """Send a text command to Anvil's shared brain router."""
    try:
        reply = router.run(command)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if reply:
        typer.echo(reply)


if __name__ == "__main__":
    app()
