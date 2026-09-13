# Remote Dispatch (SSH & SLURM)

Remote dispatch lets you send a job to an SSH host or a SLURM cluster from your local machine without manually SSHing in.

## Prerequisites — `~/.theseus.yaml`

You need a dispatch config that describes your infrastructure. Copy `examples/dispatch.yaml` from the repo as a starting point:

```bash
cp examples/dispatch.yaml ~/.theseus.yaml
```

Then edit it to match your clusters. A minimal plain-SSH example:

```yaml
clusters:
  mybox:
    root: /data/theseus  # this is the output folder
    work: /tmp/theseus   # this is a temporary directory where code is copied

hosts:
  mybox:
    ssh: mybox          # alias in ~/.ssh/config
    cluster: mybox
    type: plain
    chips:
      h100: 4
    uv_groups: [cuda13]

priority:
  - mybox
```

A minimal SLURM example:

```yaml
clusters:
  hpc:
    root: /mnt/data/theseus  # this is the output folder
    work: /scratch/theseus   # this is a temporary directory where code is copied

hosts:
  hpc-login:
    ssh: hpc            # alias in ~/.ssh/config
    cluster: hpc
    type: slurm
    partitions: [gpu]
    account: myproject
    uv_groups: [cuda13]

priority:
  - hpc-login
```

## Step 1 — Generate a config (same as local)

```bash
theseus configure gpt/train/pretrain run.yaml --chip h100 -n 4
```

## Step 2 — Submit

```bash
theseus submit my-gpt-run run.yaml
```

Theseus reads `~/.theseus.yaml`, finds the first host in `priority` that can satisfy the hardware request, ships your code, and either SSHes in to run it directly (plain host) or submits an `sbatch` job (SLURM host).

Override hardware at submit time if you didn't bake it into the config:

```bash
theseus submit my-gpt-run run.yaml --chip h100 -n 8
```

Pin to a specific cluster, or exclude one:

```bash
theseus submit my-gpt-run run.yaml --cluster hpc-login
theseus submit my-gpt-run run.yaml --exclude-cluster cloud
```

## Monitoring jobs

Submission prints the provider job IDs and log locations. Use those paths
rather than constructing a filename: each dispatch has its own nonce and
work directory.

For SLURM, inspect the allocation with `squeue` or `sacct`, follow the printed
log with `tail -f`, and cancel with `scancel JOB_ID`. For SSH, connect to the
configured host and follow the printed log path.

The current CLI does not accept `--dirty`, `--clean`, stacked `-s` configuration
files, or filesystem `--restore` paths. Put the complete configuration and
checkpoint node queries in one execution document.
