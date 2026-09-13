"""Load, edit, and save an execution configuration."""

from pathlib import Path

from loguru import logger
import typer
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from theseus.base import SUPPORTED_CHIPS
from theseus.execute.combobulator import Combobulation, Step


def load_configuration(
    source: str | Path | None,
    overrides: list[str] | None = None,
    *,
    chip: str | None = None,
    n_chips: int | None = None,
    cluster: str | None = None,
    exclude_cluster: str | None = None,
    mem: str | None = None,
    cpu: int | None = None,
) -> DictConfig:
    """Hydrate a job or YAML, applying CLI resources and component overrides."""
    document = (
        OmegaConf.load(source)
        if source and Path(source).is_file()
        else OmegaConf.create({})
    )
    if not isinstance(document, DictConfig):
        raise ValueError("Execution YAML must be a mapping")
    if not document:
        document = Combobulation(
            steps=(Step.model_validate({"job": source}),) if source else ()
        ).serialize()
    updates = {
        "execution.gpus": n_chips,
        "execution.cpus": cpu,
        "execution.minimum_memory": mem,
    }
    for key, value in updates.items():
        if value is not None:
            OmegaConf.update(document, key, value, merge=False)
    if chip is not None:
        document.execution.chips = [SUPPORTED_CHIPS[chip].name]
    if n_chips == 0:
        document.execution.chips = []
    for key, names in (
        ("preferred_clusters", cluster),
        ("forbidden_clusters", exclude_cluster),
    ):
        if names is not None:
            OmegaConf.update(
                document,
                f"execution.{key}",
                [v.strip() for v in names.split(",") if v.strip()],
                merge=False,
            )
    execution = Combobulation.deserialize(document)
    if any("=" not in value for value in overrides or []):
        raise ValueError("Configuration overrides must be key=value")
    execution.config.merge_with_dotlist(overrides or [])
    document.config = execution.config
    return document


def save_configuration(document: DictConfig, path: Path) -> None:
    """Save Combobulation YAML with readable registered-job shorthand.

    Built-in tagged identities become their registry names; Step accepts both
    forms. External cloudpickle payloads stay intact so their code travels with
    the configuration. The Combobulation nonce is preserved in execution.nonce.
    """
    execution = Combobulation.deserialize(document)
    if not execution.steps:
        raise ValueError("Add at least one job before saving")
    output = OmegaConf.merge(document, execution.serialize())
    for index, step in enumerate(execution.steps):
        if output.execution.steps[index].job.type == "registered":
            output.execution.steps[index].job = step.job_name
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(output, path)


def configure(
    source: str | None = typer.Argument(None, help="Registered job or execution YAML."),
    out_yaml: Path | None = typer.Argument(None, help="Output YAML."),
    overrides: list[str] | None = typer.Argument(
        None, help="Component key=value overrides."
    ),
    chip: str | None = typer.Option(None, "--chip"),
    n_chips: int | None = typer.Option(None, "--n_chips", "--n-chips", "-n", min=0),
    cluster: str | None = typer.Option(
        None, "--cluster", help="Allowed clusters, comma-separated."
    ),
    exclude_cluster: str | None = typer.Option(None, "--exclude-cluster"),
) -> None:
    """Configure a job/execution, or open the prompt-driven editor with no arguments."""
    if (source is None) != (out_yaml is None):
        raise typer.BadParameter("Supply both JOB_OR_YAML and OUT_YAML, or neither")
    try:
        document = load_configuration(
            source,
            overrides,
            chip=chip,
            n_chips=n_chips,
            cluster=cluster,
            exclude_cluster=exclude_cluster,
        )
        if out_yaml is None:
            from theseus.cli.configurator import Configurator

            Configurator(document).run()
        else:
            save_configuration(document, out_yaml)
            logger.info("CONFIGURE | saved configuration to {}", out_yaml)
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        ImportError,
        OmegaConfBaseException,
    ) as error:
        raise typer.BadParameter(str(error)) from error
