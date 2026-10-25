# Database Migration

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

The old store kept scalar metadata in Delta Lake under `ROOT/objects/values`.
The new store uses persistent ClickHouse tables, with embedded chDB for local
use or a ClickHouse server for concurrent processes. This removes Theseus's
Delta snapshot, retry, and compaction machinery. Old roots need an explicit
migration before the new store can read their history.

## Copy the metadata

Stop jobs writing to the old root and any legacy cleanup processes. From a
Theseus checkout, run:

```bash
./scripts/values_migrate_v1_to_v2.py /path/to/root "chdb:///path/to/new.chdb"
```

The script enables the `migrate` dependency group, which supplies `deltalake`.
It is also included in `all` and available as the `libthx[migrate]` package extra.
The progress bar reports copied rows. Use `--batch-size 8192` to set rows per
insert; 8,192 is the default.

For a shared server:

```bash
./scripts/values_migrate_v1_to_v2.py /path/to/root \
  "https://USER:PASSWORD@HOST:8443/theseus"
```

Here `theseus` is an existing database. The account needs permission to create,
insert, select, and rename tables. Percent-encode special characters in
the username and password. Embedded chDB uses filesystem permissions and
allows only one open connection to its directory at a time.

## What moves

The script copies the committed Delta snapshot, preserving event rows, metric
columns, write order, and lineage. It creates `ROOT/.id` atomically if missing
and preserves an existing ID. The destination is `values_<root-id>`: one table
per root within the selected database.

Rows first enter `values_<root-id>_migration_<uuid>`. After checking the row
count, the script renames that table to its final name. An existing destination
table is never overwritten. The script prints the staging and destination names
before uploading. Failed or interrupted imports keep their staging tables for
inspection; the script never drops them. A rerun uploads again into a fresh
table, so inspect the retained table before deciding to rerun. There is no
automatic resume.

Blob files stay in `ROOT/objects/blobs`. The old Delta directory stays intact.
This copies metadata; it does not move checkpoints or delete the source.

## Use the new database

Set the cluster's `metadata` in `~/.theseus.yaml` to the same destination:

```yaml
clusters:
  my-cluster:
    # Keep the existing root and other cluster settings.
    metadata: "chdb:///path/to/new.chdb"
```

Use the server URL instead for concurrent jobs. Direct Python callers pass the
same value to `ObjectStore.local(root, metadata=...)` or
`ObjectReader.local(root, metadata=...)`. Omitting it selects `<root>/objects/values/v2`;
the root's `.id` identifies its table, but does not record the connection string.

Keep `.id` with the root and verify the migrated runs before resuming work.
See [Experiment DAG and Storage](../Design/dag.md) for the storage layout and
scalar conflict rules.
