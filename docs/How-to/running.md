# Running Experiments

The typical workflow is two steps: **configure a run → run (or submit) it**.

```bash
uv run theseus configure <job> config.yaml             # make config
```

```bash
uv run theseus run <name> config.yaml <output_dir>     # run locally
uv run theseus submit <name> config.yaml               # or send to remote infra
```

Before you get started, you can run:

```bash
uv run theseus jobs
```

This prints a table of every registered experiment. Pick one — for the examples below we'll use `gpt/train/pretrain`.

Choose a workflow:

- **[Local](running-local.md)** — run on your current machine
- **[Remote Configuration](remote-configuration/providers.md)** — configure provider access, storage, and worker environments
- **[Remote (SSH & SLURM)](running-remote.md)** — submit to a plain SSH host or SLURM cluster
- **[Remote (Volcano / K8s)](volcano.md)** — submit to a Kubernetes cluster with Volcano
- **[Interactive inspection](running-repl.md)** — use Jupyter in an allocated environment
