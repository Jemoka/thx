"""Run a configured execution on the current machine."""

from pathlib import Path
from uuid import uuid4

from loguru import logger
import typer
from omegaconf.errors import OmegaConfBaseException

from theseus.cli.configure import load_configuration
from theseus.execute.combobulator import Combobulation
from theseus.execute.dispatch import DispatchSpec


def run(
    name: str = typer.Argument(...),
    yaml_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    root_dir: Path = typer.Argument(..., file_okay=False),
    overrides: list[str] | None = typer.Argument(None),
    project: str = typer.Option("general", "--project", "-p"),
    group: str = typer.Option("default", "--group", "-g"),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Generate a fresh run nonce to prevent job resumption even if available.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run the YAML's job chain locally, storing its artifacts under ROOT_DIR."""
    try:
        document = load_configuration(yaml_path, overrides)
        execution = Combobulation.deserialize(document)
        if fresh:
            execution = execution.model_copy(update={"nonce": uuid4().hex[:6]})
        if not execution.healthcheck():
            raise ValueError(
                "Execution needs at least one job and complete configuration"
            )
        dispatch = DispatchSpec(
            name=name,
            project=project,
            group=group,
            job=execution,
        )
        if dry_run:
            typer.echo(dispatch.model_dump_json(indent=2))
        else:
            logger.info("RUN | starting {} in {}", name, root_dir)
            if not dispatch.run(root_dir):
                raise typer.Exit(1)
            logger.info("RUN | completed {}", name)
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        ImportError,
        OmegaConfBaseException,
    ) as error:
        raise typer.BadParameter(str(error)) from error
