"""List registered jobs and their descriptions."""

from inspect import getdoc
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from theseus.registry import JOBS


def jobs(
    prefix: Annotated[str, typer.Argument(help="Filter job names by prefix.")] = "",
) -> None:
    """List registered job types. Use ui to browse recorded runs."""
    table = Table("Job", "Description")
    for name, job in sorted(JOBS.items()):
        if not name.startswith(prefix):
            continue
        label = Text(name)
        if prefix:
            label.stylize("bold cyan", 0, len(prefix))
        doc = getdoc(job) or ""
        description = " ".join(doc.split("\n\n", 1)[0].split())
        table.add_row(label, Text(description or "—"))
    Console().print(table)
