"""Prompt-driven Rich editor for an execution document."""

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig
from omegaconf.errors import OmegaConfBaseException
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from theseus.cli.configure import load_configuration, save_configuration
from theseus.cli.configurator.fields import edit_fields
from theseus.cli.configurator.steps import edit_steps
from theseus.execute.combobulator import Combobulation


@dataclass
class Configurator:
    """Edit a configuration through ordinary terminal prompts."""

    document: DictConfig
    console: Console = field(default_factory=Console)

    def ask(
        self, question: str, choices: list[str] | None = None, default: str = ""
    ) -> str:
        for index, choice in enumerate(choices or [], 1):
            self.console.print(f"  {index}) {choice}", markup=False)
        numbers = [str(index) for index in range(1, len(choices or []) + 1)]
        answer = Prompt.ask(
            Text(question),
            console=self.console,
            default=default,
            choices=[*numbers, *choices, ":cancel"] if choices else None,
            show_choices=False,
            show_default=bool(default),
        )
        if answer == ":cancel":
            raise InterruptedError
        return choices[int(answer) - 1] if choices and answer in numbers else answer

    def run(self) -> Path | None:
        choices = [
            "Add job",
            "Edit step",
            "Remove step",
            "Edit configuration",
            "Edit resources and clusters",
            "Load configuration",
            "Save",
            "Quit",
        ]
        while True:
            flow = Text()
            for index, step in enumerate(
                Combobulation.deserialize(self.document).steps, 1
            ):
                mode = (
                    "resume"
                    if step.resume
                    else "branch"
                    if step.base is not None
                    else "fresh"
                )
                if step.base is not None:
                    mode += (
                        " from previous"
                        if step.base == "previous"
                        else " from checkpoint query"
                    )
                if index > 1:
                    flow.append("      | next\n      v\n", style="dim")
                flow.append(f"[{index}. {step.job_name}]", style="bold cyan")
                flow.append(f" ({mode})\n")
            if not flow:
                flow.append("No jobs yet. Choose Add job to begin.", style="dim")
            self.console.print()
            self.console.print(
                Panel(flow, title="Execution", box=box.ASCII, padding=(1, 2))
            )
            self.console.print()
            self.console.rule("Configuration", characters="-", style="dim")
            self.console.print(
                "Type :cancel at any prompt to return here.", style="dim"
            )
            try:
                selected = self.ask("Configuration menu", choices)
                if selected == "Quit":
                    return None
                if selected in ("Add job", "Edit step", "Remove step"):
                    edit_steps(self, selected)
                elif selected in ("Edit configuration", "Edit resources and clusters"):
                    edit_fields(
                        self, resources=selected == "Edit resources and clusters"
                    )
                elif selected == "Load configuration":
                    path = Path(self.ask("Existing configuration path"))
                    if not path.is_file():
                        raise ValueError("Configuration file does not exist")
                    self.document = load_configuration(path)
                elif selected == "Save":
                    path = Path(
                        self.ask("Save configuration to", default="config.yaml")
                    )
                    if (
                        path.exists()
                        and self.ask("Replace existing file?", ["No", "Yes"], "1")
                        != "Yes"
                    ):
                        continue
                    save_configuration(self.document, path)
                    return path
            except InterruptedError:
                self.console.print("Edit cancelled.", style="dim")
            except (EOFError, KeyboardInterrupt):
                return None
            except (
                ValueError,
                TypeError,
                KeyError,
                OSError,
                ImportError,
                OmegaConfBaseException,
            ) as error:
                self.console.print(f"Error: {error}", markup=False)
