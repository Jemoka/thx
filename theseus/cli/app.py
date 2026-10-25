"""Theseus command-line application."""

import sys
import importlib
import importlib.util
from pathlib import Path
from hashlib import sha256

from loguru import logger
import typer

from theseus.cli.ui import ui
from theseus.cli.configure import configure
from theseus.cli.submit import submit
from theseus.cli.run import run
from theseus.cli.jobs import jobs


app = typer.Typer(name="theseus", no_args_is_help=True)
app.add_typer(ui, name="ui")
app.command()(configure)
app.command()(submit)
app.command()(run)
app.command()(jobs)


@app.callback()
def main(
    imports: list[str] | None = typer.Option(
        None,
        "--import",
        help="Import a module name or .py path to register jobs. Repeatable.",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Enable DEBUG logs (default: INFO). May be repeated.",
    ),
) -> None:
    """Run and inspect Theseus workloads."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        backtrace=False,
        diagnose=False,
    )
    for source in imports or []:
        try:
            if source.endswith(".py") or Path(source).is_file():
                path = Path(source).resolve()
                name = "_theseus_import_" + sha256(str(path).encode()).hexdigest()
                if name in sys.modules:
                    continue
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot import Python file {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(name, None)
                    raise
            else:
                # Console scripts do not necessarily put the working directory on sys.path.
                if str(Path.cwd()) not in sys.path:
                    sys.path.insert(0, str(Path.cwd()))
                importlib.import_module(source)
        except (ImportError, OSError, ValueError, SyntaxError) as error:
            raise typer.BadParameter(str(error), param_hint="--import") from error


if __name__ == "__main__":
    app()
