"""Field selection and typed YAML value editing."""

from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, OmegaConf

from theseus.base import SUPPORTED_CHIPS
from theseus.execute.combobulator import Combobulation

if TYPE_CHECKING:
    from theseus.cli.configurator import Configurator


def edit_fields(editor: "Configurator", resources: bool = False) -> None:
    fields: dict[str, Any]
    if resources:
        fields = {
            key: OmegaConf.select(editor.document, key, default=default)
            for key, default in cast(
                dict[str, Any],
                {
                    "execution.gpus": None,
                    "execution.chips": [],
                    "execution.cpus": None,
                    "execution.minimum_memory": None,
                    "execution.sharding.tp": 1,
                    "execution.sharding.fsdp": False,
                    "execution.sharding.zero": True,
                    "execution.sharding.activation_checkpointing": False,
                    "execution.preferred_clusters": [],
                    "execution.forbidden_clusters": [],
                },
            ).items()
        }
    else:
        fields = {}
        pending = [
            ("config", OmegaConf.to_container(editor.document.config, resolve=False))
        ]
        while pending:
            prefix, value = pending.pop()
            if isinstance(value, dict):
                pending.extend(
                    (f"{prefix}.{str(key)}", item) for key, item in value.items()
                )
            else:
                fields[prefix] = value
    query = editor.ask("Filter fields (Enter shows all)")
    options = [
        f"{key} = {value}"
        for key, value in sorted(fields.items())
        if query.lower() in key.lower()
    ]
    if not options:
        raise ValueError("No fields matched")
    selected = editor.ask("Choose a field", options)
    key = selected.split(" = ", 1)[0]
    value = editor.ask("New value (YAML; use [a, b] for lists)")
    candidate = cast(
        DictConfig,
        OmegaConf.merge(editor.document, OmegaConf.from_dotlist([f"{key}={value}"])),
    )
    if key == "execution.chips":
        candidate.execution.chips = [
            SUPPORTED_CHIPS[chip].name for chip in candidate.execution.chips
        ]
    if key in ("execution.gpus", "execution.cpus", "execution.sharding.tp"):
        minimum = 0 if key == "execution.gpus" else 1
        number = OmegaConf.select(candidate, key)
        if not isinstance(number, int) or isinstance(number, bool) or number < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        if key == "execution.gpus" and number == 0:
            candidate.execution.chips = []
    candidate.config = Combobulation.deserialize(candidate).config
    editor.document = candidate
