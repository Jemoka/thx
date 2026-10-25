# Metadata Store

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

## 1. Pick local or remote storage

Use a local chDB directory for one process, or a ClickHouse server for concurrent
jobs. Set `metadata` under your cluster in `~/.theseus.yaml`:

```yaml
# Local: Theseus creates this directory on the worker.
metadata: "chdb:///absolute/path/to/metadata.chdb"
```

```yaml
# Remote: use your server's connection details.
metadata: "https://USER:PASSWORD@HOST:8443/theseus"
```

Choose one. Omit `metadata` to use `<root>/objects/values/v2`. Local storage allows only
one open connection at a time.

## 2. Make a ClickHouse DB (remote only)

[Install ClickHouse locally](https://clickhouse.com/docs/get-started/setup/install)
or [create a ClickHouse Cloud service](https://clickhouse.com/docs/get-started/setup/cloud).
In its SQL console or `clickhouse-client`, run:

```sql
CREATE DATABASE IF NOT EXISTS theseus;
```

## 3. Connect your cluster

Add your chosen `metadata` value to the existing cluster. Keep the other settings:

```yaml title="~/.theseus.yaml"
clusters:
  lab:
    root: /shared/theseus
    work: /scratch/theseus-work
    metadata: "chdb:///absolute/path/to/metadata.chdb"
    # Or: metadata: "https://USER:PASSWORD@HOST:8443/theseus"
```

For a server, `/theseus` selects the database. Workers must be able to reach it.
Percent-encode special characters in credentials, such as `@` → `%40`.

## 4. Run a job

With your [provider](providers.md) and training YAML configured, submit normally:

```bash
uv run theseus submit my-run train.yaml --cluster lab
```

Workers receive the connection automatically. Theseus creates the tables;
checkpoint files stay under your existing root.

For a Python session, pass the connection directly:

```python
from theseus.quick import quick

with quick(root_dir="/path/to/theseus", metadata_cnx="https://USER:PASSWORD@HOST:8443/theseus") as q:
    nodes = q.find().all()
```

`init(root_dir=..., metadata_cnx=...)` accepts the same arguments; call `close()`
when finished. Omit `metadata_cnx` for the root's local database.

Did you use Theseus before [7e732b84](https://review.gerrithub.io/c/Jemoka/theseus/+/1256356)?
[Migrate your data](../database-migration.md) before submitting.
