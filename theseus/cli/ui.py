"""Interactive Theseus interface command."""

from pathlib import Path
from typing import Annotated

import typer
from nicegui import ui as nicegui

from theseus.cli.interface.driver import TheseusInterface


ui = typer.Typer(
    invoke_without_command=True,
    context_settings={"allow_interspersed_args": True},
)


@ui.callback()
def main(
    context: typer.Context,
    root: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            resolve_path=True,
            metavar="root",
        ),
    ],
    logs: Annotated[
        Path | None,
        typer.Option(
            "--logs",
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="Directory containing new-execute bootstrap logs.",
        ),
    ] = None,
    serve: Annotated[
        bool,
        typer.Option("--serve", help="Serve without opening a browser."),
    ] = False,
    bind: Annotated[
        str,
        typer.Option("--bind", help="Address for the web server."),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option("--port", help="Port for the web server."),
    ] = 8000,
) -> None:
    """Open the interactive Theseus interface."""
    if context.invoked_subcommand is not None:
        return
    nicegui.run(
        root=lambda: TheseusInterface(root, logs),
        host=bind,
        port=port,
        title="Theseus",
        show=not serve,
        reload=False,
        favicon="▸",
    )
