"""Prompts for editing execution steps and checkpoint lineage."""

from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from theseus.execute.combobulator import Combobulation, Step
from theseus.job import BasicJob
from theseus.registry import JOBS
from theseus.store import _SerializedQueryBuilder

if TYPE_CHECKING:
    from theseus.cli.configurator import Configurator


def select_step(editor: "Configurator", step: Step, index: int) -> Step:
    mode = editor.ask("How should this job start?", ["Fresh", "Branch", "Resume"])
    base: Any = None
    if mode != "Fresh":
        source = editor.ask(
            "Checkpoint source",
            (["Previous step"] if index else []) + ["Exact node", "Query"],
        )
        if source == "Previous step":
            base = "previous"
        else:
            query = _SerializedQueryBuilder().checkpoint()
            if source == "Exact node":
                query.node(editor.ask("Serialized checkpoint node"))
            else:
                for field in ("project", "group", "run"):
                    value = editor.ask(f"Match {field} (Enter leaves unrestricted)")
                    if value:
                        query.spec(**{field: value})
                while (
                    editor.ask("Add a metric condition?", ["No", "Yes"], "1") == "Yes"
                ):
                    key = editor.ask("Metric name")
                    operator = editor.ask(
                        "Comparison", ["=", "!=", "<", "<=", ">", ">="]
                    )
                    value = OmegaConf.from_dotlist(
                        [f"value={editor.ask('Value (YAML scalar)')}"]
                    ).value
                    query.where(key, operator, value)
                key = editor.ask("Sort by metric (Enter uses checkpoint write order)")
                if key:
                    ascending = (
                        editor.ask("Order", ["Ascending", "Descending"], "1")
                        == "Ascending"
                    )
                    query.sort(key, ascending)
            base = query.serialize()
    return Step(job=step.job, base=base, resume=mode == "Resume")


def edit_steps(editor: "Configurator", action: str) -> None:
    execution = Combobulation.deserialize(editor.document)
    steps = list(execution.steps)
    if action == "Add job":
        query = editor.ask("Filter composable job names (Enter shows all)")
        names = sorted(
            name
            for name, job in JOBS.items()
            if issubclass(job, BasicJob) and query.lower() in name.lower()
        )
        if not names:
            raise ValueError("No composable jobs matched")
        name = editor.ask("Choose a job", names)
        step = Step(job=JOBS[name])
        steps.append(select_step(editor, step, len(steps)) if steps else step)
    else:
        if not steps:
            raise ValueError("Add a job first")
        selected = editor.ask(
            "Choose a step",
            [f"{i + 1}. {step.job_name}" for i, step in enumerate(steps)],
        )
        index = int(selected.split(".", 1)[0]) - 1
        if action == "Edit step":
            steps[index] = select_step(editor, steps[index], index)
        else:
            if index + 1 < len(steps) and steps[index + 1].base == "previous":
                raise ValueError(
                    "The next step depends on this one; edit its base before removing it"
                )
            del steps[index]
    updated = Combobulation(
        **{
            **{name: getattr(execution, name) for name in Combobulation.model_fields},
            "steps": tuple(steps),
        }
    )
    # Carry over only fields still owned by the remaining jobs.
    pending = [
        (
            "",
            OmegaConf.to_container(updated.config, resolve=False),
            OmegaConf.to_container(execution.config, resolve=False),
        )
    ]
    while pending:
        prefix, schema, previous = pending.pop()
        if isinstance(schema, dict) and isinstance(previous, dict):
            pending.extend(
                (f"{prefix}.{str(key)}" if prefix else str(key), value, previous[key])
                for key, value in schema.items()
                if key in previous
            )
        elif prefix:
            OmegaConf.update(updated.config, prefix, previous, merge=False)
    editor.document.execution = updated.serialize().execution
    editor.document.config = updated.config
