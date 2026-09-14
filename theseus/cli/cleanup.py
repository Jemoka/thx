"""Compact and vacuum a Theseus root's Delta value table."""

from pathlib import Path

from deltalake import DeltaTable
from loguru import logger
from rich.console import Console
from rich.table import Table
import typer

from theseus.store import ObjectStore


def cleanup(
    root: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Compact ROOT/objects/values and vacuum using the table's retention policy."""
    values = root / "objects" / "values"
    try:
        logger.info("CLEANUP | compacting {}", values)
        metrics = ObjectStore.compact(root)
        if metrics is None:
            logger.info("CLEANUP | no Delta value table at {}", values)
            return
        logger.info("CLEANUP | vacuuming {} with table retention", values)
        removed = DeltaTable(values).vacuum(dry_run=False, full=True)
        summary = Table("Statistic", "Files", title="Value table cleanup")
        summary.add_row("Compacted inputs", str(metrics["numFilesRemoved"]))
        summary.add_row("Compacted outputs", str(metrics["numFilesAdded"]))
        summary.add_row("Expired files deleted", str(len(removed)))
        Console().print(summary)
    except Exception as error:
        logger.error("CLEANUP | {}", error)
        raise typer.Exit(1) from error
