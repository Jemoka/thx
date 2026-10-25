#!/usr/bin/env -S uv run --group migrate python
"""Migrate ROOT/objects/values into a new persistent chDB or ClickHouse table.

Usage: ./scripts/values_migrate_v1_to_v2.py ROOT CONNECTION_STRING

Stop legacy writers and cleanup before migrating. The committed Delta snapshot
is copied in bounded Arrow batches; blobs stay in ROOT/objects/blobs. An existing
root table is never overwritten. Failed imports retain their staging tables for
inspection. Rerunning creates a separate table and uploads again; it does not
resume. Delta is provided by the migrate dependency group or package extra.
"""

import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import pyarrow as pa
from deltalake import DeltaTable
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from theseus.store import ObjectReader, QueryBuilder, _METADATA_TYPES


def migrate(root: Path, metadata: str, batch_size: int = 8192) -> int:
    """Copy one committed legacy snapshot, publishing only a verified import."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    root = root.expanduser().resolve()
    console = Console()
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Reading Delta snapshot", total=None)
        source = DeltaTable(root / "objects" / "values")
        dataset = source.to_pyarrow_dataset()
        required = {
            "_x_name",
            "_x_nonce",
            "_x_seq",
            "_x_prc_idx",
            "_x_write",
            "_x_event",
        }
        if missing := required - set(dataset.schema.names):
            raise ValueError(
                f"legacy table is missing required metadata: {sorted(missing)}"
            )
        column_types = {
            pa.bool_(): "Bool",
            pa.int64(): "Int64",
            pa.float64(): "Float64",
            pa.string(): "String",
        }
        types = dict(_METADATA_TYPES)
        for field in dataset.schema:
            if field.type not in column_types:
                raise TypeError(
                    f"unsupported legacy column {field.name!r}: {field.type}"
                )
            if field.name in types and types[field.name] != column_types[field.type]:
                raise TypeError(
                    f"invalid legacy metadata type for {field.name!r}: {field.type}"
                )
            types[field.name] = column_types[field.type]
        total = dataset.count_rows()
        progress.update(
            task, description=f"Delta v{source.version()}: copying rows", total=total
        )

        identity = root / ".id"
        if not identity.exists():
            with NamedTemporaryFile(dir=root, prefix=".id.") as pending:
                pending.write(uuid4().hex.encode())
                pending.flush()
                os.fsync(pending.fileno())
                try:
                    os.link(pending.name, identity)
                except FileExistsError:
                    pass
                directory = os.open(root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)

        reader = ObjectReader.local(root, metadata=metadata)
        client = reader._client
        assert client is not None
        staging = f"{reader._table}_migration_{uuid4().hex}"
        published = False
        try:
            if client.command(f"EXISTS TABLE {reader._table}"):
                raise ValueError(
                    f"destination table {reader._table} already exists; use a new database"
                )
            columns = ", ".join(
                f"{QueryBuilder._identifier(name)} Nullable({kind})"
                for name, kind in types.items()
            )
            settings = {"max_query_size": max(262144, len(columns.encode()) * 2 + 4096)}
            console.print(f"Staging table: {staging}", soft_wrap=True)
            console.print(f"Destination table: {reader._table}", soft_wrap=True)
            client.command(
                f"CREATE TABLE {staging} ({columns}) ENGINE = MergeTree ORDER BY "
                "(ifNull(_x_name, ''), ifNull(_x_nonce, ''), ifNull(_x_seq, 0))",
                settings=settings,
            )
            copied = 0
            for batch in dataset.to_batches(
                batch_size=batch_size, batch_readahead=1, fragment_readahead=1
            ):
                if not batch.num_rows:
                    continue
                client.insert_arrow(
                    staging,
                    pa.Table.from_batches([batch]),
                    settings={**settings, "async_insert": 0},
                )
                copied += batch.num_rows
                progress.advance(task, batch.num_rows)
            progress.update(task, description="Verifying imported rows")
            stored = int(client.command(f"SELECT count() FROM {staging}"))
            if copied != total or stored != total:
                raise RuntimeError(
                    f"row count mismatch: source={total}, copied={copied}, stored={stored}"
                )
            # A single rename refuses a destination created by another writer.
            client.command(f"RENAME TABLE {staging} TO {reader._table}")
            published = True
            progress.update(task, description="Migration complete")
        finally:
            try:
                if not published:
                    console.print(
                        f"Migration did not complete. No tables were deleted. "
                        f"If created, staging table {staging} remains for inspection. "
                        "Rerunning uploads again into a new staging table.",
                        soft_wrap=True,
                    )
            finally:
                reader.close()
    console.print(
        f"Migrated {copied:,} rows. Blob files remain in {root / 'objects' / 'blobs'}."
    )
    return copied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="./scripts/values_migrate_v1_to_v2.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root", type=Path, help="Existing Theseus root containing objects/values"
    )
    parser.add_argument(
        "metadata", help="New chDB path/URI or HTTP(S) ClickHouse database DSN"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8192, help="Rows per insert (default: 8192)"
    )
    args = parser.parse_args()
    migrate(args.root, args.metadata, args.batch_size)
