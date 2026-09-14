"""Compact and vacuum a Theseus root's Delta value table."""

from pathlib import Path
from time import sleep

from deltalake import DeltaTable
from loguru import logger
from rich.console import Console
from rich.table import Table
import typer

from theseus.store import ObjectStore


def cleanup(
    root: Path = typer.Argument(..., exists=True, file_okay=False),
    expire: bool = typer.Option(
        False,
        "--expire",
        help="Immediately delete obsolete files. Can break active readers and writers; waits 10 seconds before starting.",
    ),
) -> None:
    """Compact ROOT/objects/values and vacuum; --expire bypasses retention."""
    values = root / "objects" / "values"
    if expire:
        logger.warning(
            "CLEANUP | --expire PERMANENTLY DELETES obsolete and unreferenced files "
            "without retention. Active readers may fail; concurrent writers may lose "
            "uncommitted files. Old table snapshots may become unreadable. "
            "Active readers/writers are not detected. Press Ctrl-C within 10 seconds "
            "to cancel if any are running."
        )
        sleep(10)
    try:
        logger.info("CLEANUP | compacting {}", values)
        metrics = ObjectStore.compact(root)
        if metrics is None:
            logger.info("CLEANUP | no Delta value table at {}", values)
            return
        logger.info(
            "CLEANUP | vacuuming {} with {}",
            values,
            "zero retention" if expire else "table retention",
        )
        removed = DeltaTable(values).vacuum(
            dry_run=False,
            full=True,
            retention_hours=0 if expire else None,
            enforce_retention_duration=not expire,
        )
        summary = Table("Statistic", "Files", title="Value table cleanup")
        summary.add_row("Compacted inputs", str(metrics["numFilesRemoved"]))
        summary.add_row("Compacted outputs", str(metrics["numFilesAdded"]))
        summary.add_row("Expired files deleted", str(len(removed)))
        Console().print(summary)
    except Exception as error:
        logger.error("CLEANUP | {}", error)
        raise typer.Exit(1) from error
