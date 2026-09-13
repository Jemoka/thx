"""Submit a configured execution through the existing solver and providers."""

from pathlib import Path
from uuid import uuid4
import re

from loguru import logger
import typer
from omegaconf.errors import OmegaConfBaseException

from theseus.cli.configure import load_configuration
from theseus.execute.combobulator import Combobulation
from theseus.execute.config import DispatchConfig
from theseus.execute.dispatch import DispatchSpec
from theseus.execute.solve import solve


def submit(
    name: str = typer.Argument(...),
    yaml_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    overrides: list[str] | None = typer.Argument(None),
    dispatch_config: Path | None = typer.Option(None, "--dispatch-config", "-d"),
    project: str = typer.Option("general", "--project", "-p"),
    group: str = typer.Option("default", "--group", "-g"),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Generate a fresh run nonce to prevent job resumption even if available.",
    ),
    chip: str | None = typer.Option(None, "--chip"),
    n_chips: int | None = typer.Option(None, "--n_chips", "--n-chips", "-n", min=0),
    mem: str | None = typer.Option(None, "--mem"),
    cpu: int | None = typer.Option(None, "--cpu", min=1),
    cluster: str | None = typer.Option(None, "--cluster"),
    exclude_cluster: str | None = typer.Option(None, "--exclude-cluster"),
    extras: list[str] | None = typer.Option(
        None, "--extras", help="Add a worker dependency group. Repeatable."
    ),
    env: list[str] | None = typer.Option(
        None,
        "--env",
        help="Override a worker environment variable: KEY=VALUE. Repeatable.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run every step of a combobulation within one allocation."""
    try:
        environment = {}
        for assignment in env or []:
            key, separator, value = assignment.partition("=")
            if (
                not separator
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                or "\x00" in value
            ):
                raise ValueError(
                    "--env requires KEY=VALUE with a valid environment variable name"
                )
            environment[key] = value
        logger.debug("SUBMIT | loading configuration from {}", yaml_path)
        document = load_configuration(
            yaml_path,
            overrides,
            chip=chip,
            n_chips=n_chips,
            cluster=cluster,
            exclude_cluster=exclude_cluster,
            mem=mem,
            cpu=cpu,
        )
        execution = Combobulation.deserialize(document)
        if fresh:
            execution = execution.model_copy(update={"nonce": uuid4().hex[:6]})
        if not execution.healthcheck():
            raise ValueError(
                "Execution needs at least one job and complete configuration"
            )
        inventory = DispatchConfig.load(dispatch_config)
        logger.info("SUBMIT | resolving hardware")
        allocated = solve(execution, inventory)
        if allocated.result is None:
            raise ValueError(
                "No configured provider satisfies this execution and request"
            )
        for host in allocated.result.hosts:
            host.uv_groups = list(dict.fromkeys([*host.uv_groups, *(extras or [])]))
            host.env.update(environment)
        dispatch = DispatchSpec(
            name=name,
            project=project,
            group=group,
            hardware=allocated.result,
            job=execution,
        )
        if dry_run:
            preview = dispatch.model_copy(deep=True)
            assert preview.hardware is not None
            preview.hardware = preview.hardware.redacted()
            typer.echo(preview.model_dump_json(indent=2))
            return
        logger.debug("SUBMIT | publishing and launching {}", name)
        shipped = dispatch.launch(inventory)
        if not shipped.ok:
            raise ValueError(shipped.remote.stderr or "Provider rejected the execution")
        logger.info("SUBMIT | submitted {}: {}", name, ", ".join(shipped.job_ids))
        for path in shipped.logs:
            logger.info("SUBMIT | log: {}", path)
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        ImportError,
        OmegaConfBaseException,
    ) as error:
        raise typer.BadParameter(str(error)) from error
