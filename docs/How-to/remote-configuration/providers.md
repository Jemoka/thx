# Remote Configuration

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Use this guide to configure the machines and schedulers that `theseus submit`
can use. Start with one provider, verify it, then add more providers or optional
settings as you need them. The field tables cover the complete current dispatch
configuration, including fields that are accepted but have no effect today.

You will need access to an existing SSH machine, SLURM cluster, or Kubernetes
cluster with Volcano. Have the relevant filesystem paths, hardware capacity,
and account or queue information ready. Run the local commands below from your
theseus checkout.

The result is an **infrastructure configuration**: where jobs may run, how they
reach their files, and which execution environment to prepare. Your experiment's
model, training parameters, checkpoint selection, and sharding belong in its
separate execution YAML. See [Configure and run an experiment](../running.md) for
that file.

## 1. Choose the configuration file

This guide uses `~/.theseus.yaml`. theseus also supports an XDG location, and
that location takes precedence when you do not pass a filename:

1. `--dispatch-config PATH`, when supplied, selects exactly that file.
2. Otherwise, theseus checks `$XDG_CONFIG_HOME/theseus/config.yaml`, defaulting
   to `~/.config/theseus/config.yaml`.
3. If the XDG file does not exist, it checks `~/.theseus.yaml`.

These files are alternatives; their contents are not merged. During setup, pass
`--dispatch-config ~/.theseus.yaml` explicitly so you know which file is used.

If you already have the file, edit it in place. For a new file, begin with:

```yaml
clusters: {}
hosts: {}
priority: []
gres_mapping: {}
```

Keep each top-level key only once as you add the examples below. Add entries
inside the existing `clusters` and `hosts` mappings instead of appending another
copy of either mapping.

| Top-level field | Default | What to put here |
| --- | --- | --- |
| `clusters` | `{}` | Named filesystem configurations. Each host refers to one of these names. |
| `hosts` | `{}` | Named compute providers: one SSH machine, one SLURM login endpoint, or one Volcano pool per entry. |
| `priority` | `[]` | Host entry names in the order you want the solver to try them. Unlisted hosts are tried afterward in YAML insertion order. |
| `gres_mapping` | `{}` | theseus chip keys mapped to your SLURM GPU type strings. Required for the GPU types you want to allocate through SLURM. |
| `mount` | `null` | Accepted legacy top-level field; the current providers do not use it. Configure JuiceFS under `clusters.<name>.mount` instead. |
| `proxy` | `null` | Accepted legacy top-level field; the current providers do not use it as an SSH proxy. Configure jump hosts in SSH configuration instead. |

Do not set both top-level `mount` and `proxy` to nonempty values: the loader
rejects that combination even though the current providers do not use them.
For a new configuration, omit both.

## 2. Define where a cluster keeps files

A **cluster** here names a filesystem layout. A **host** names a way to allocate
compute that uses that layout. They can have different names, and several hosts
can refer to the same cluster.

For example, add a cluster called `lab`:

```yaml
clusters:
  lab:
    root: /shared/theseus
    work: /scratch/theseus-work
    log: /shared/theseus-logs
    share: /shared/theseus-dispatch
```

Replace these paths with paths on the remote machines. Use absolute paths:
theseus does not generally expand `~` or shell variables inside these fields.

Choose `root` for persistent run data. Choose `work` for extracted code and
per-run Python environments. Choose `share` for the bootstrap and dispatch
files that must be reachable before a worker starts. They are separate because
a fast local work disk and a shared filesystem serve different purposes.

For SSH, all paths must be usable by the remote account. For SLURM, `share`
must be visible at the same path from the login host and compute nodes; use a
shared log directory if you want to inspect logs from the login host. `work`
may be local to each compute node when `share` and `log` are set separately.
For Volcano, paths refer to the container's filesystem; the publication
location must be inside its mounted PVC.

| Cluster field | Default | How to choose it |
| --- | --- | --- |
| `root` | Required | Persistent theseus root on the worker. Also the mount point if this cluster requests JuiceFS. |
| `work` | Required | Base directory for unpacked repositories and run environments. Workers get separate subdirectories beneath project/group/name/nonce. |
| `log` | `<work>/logs` | Directory for durable process logs. Set explicitly when `work` is temporary or only visible on compute nodes. |
| `data` | `<root>/data` | Override the cluster's dataset directory. |
| `checkpoints` | `<root>/checkpoints` | Override the cluster's checkpoint-directory property. Object-store checkpoints can instead live as blobs under `objects`; changing this does not relocate existing blobs. |
| `results` | `<root>/results` | Override the cluster's results directory. |
| `objects` | `<root>/objects` | Object-store directory containing scalar metadata and blobs, including saved checkpoint configuration. Keep it accessible wherever you query or restore runs. |
| `status` | `<root>/status` | Override the cluster's status directory. |
| `share` | `<work>/.dispatch` | Base directory where the provider publishes dispatch files. For Volcano, `pvc_codedrop_path` can override it. |
| `mount` | `null` | JuiceFS metadata URL. Omit it when storage is already provided without a bootstrap-managed JuiceFS mount. |
| `cache_size` | `null` | String passed to JuiceFS `--cache-size`; measured in MiB. For example, `"102400"` is 100 GiB. Omission leaves the JuiceFS client default. |
| `cache_dir` | `null` | String passed to JuiceFS `--cache-dir`. Point it at suitable worker-local cache storage; omission leaves the client default. |
| `all_squash` | `null` | String passed to JuiceFS `--all-squash`, such as `"23296:100"`, mapping requests to that numeric UID:GID. |

Leave directory overrides unset unless you need a different layout. The paths
are locations, not migration instructions: changing them will not copy existing
runs into the new directories.

### If you use JuiceFS

Add mount settings to the relevant cluster, keeping publication and early logs
accessible before the mount is established:

```yaml
clusters:
  lab:
    root: /mnt/theseus
    work: /scratch/theseus-work
    log: /shared/theseus-logs
    share: /shared/theseus-dispatch
    mount: redis://juicefs-metadata.example/1
    cache_size: "102400"
    cache_dir: /scratch/juicefs-cache
    all_squash: "23296:100"
```

Replace the metadata URL and numeric IDs with those used by your filesystem.
theseus mounts at `root`, passes the three optional settings above, and requests
`--umask=0000` for newly created files. That does not change permissions on
existing files. JuiceFS defines the cache units and identity mapping behavior;
see its [mount option reference](https://juicefs.com/docs/community/command_reference/).

If `root` is already a mount point, the bootstrap reuses it. Editing this YAML
therefore does **not** update an existing mount's cache or squash settings.
Check the active mount when troubleshooting permissions; a configured
`all_squash` value alone does not prove it is active on that machine. Coordinate
any remount with the filesystem's users.

## 3. Add the provider you actually use

Choose one of the following sections first. You can add the others later.
Every `cluster` value must name an entry you created in step 2.

### Option A: one machine over SSH

Use `type: plain` for a machine where you can run directly without a scheduler:

```yaml
hosts:
  workstation:
    type: plain
    ssh: gpu-workstation
    cluster: lab
    chips: {h100: 4}
    uv_groups: [cuda13]
    env:
      HF_HOME: /scratch/huggingface
```

`workstation` is the theseus provider name. `gpu-workstation` is the SSH target,
which may be an alias in your local `~/.ssh/config`. Verify that login works
without an interactive password prompt:

```bash
ssh -o BatchMode=yes gpu-workstation true
```

SSH configuration controls the username, key, port, and any `ProxyJump`. Those
are not separate theseus YAML fields.

| SSH host field | Default | What it controls |
| --- | --- | --- |
| `type` | `plain` when omitted | Selects direct SSH execution. Use the literal `plain`, not `ssh`. |
| `ssh` | Required | SSH target reached from the submitting machine. |
| `cluster` | Required | Filesystem cluster name. |
| `chips` | `{}` | Map of supported chip keys to maximum usable counts on this machine. GPU selection also checks live `nvidia-smi` output. |
| `uv_groups` | `[]` | Dependency groups installed in the remote environment; see step 4. |
| `env` | `{}` | String-valued worker environment variables; see step 4. |

Use the keys from [Available Chips](../../Design/chips.md). For example,
`chips: {h100: 4}` permits up to four H100s on this host. It does not request
four for every job. The execution or `submit -n` supplies the request.

Direct SSH selects one machine and checks GPU memory use as an availability
heuristic. It is not a reservation scheduler, and it does not enforce CPU or
RAM requests. Its GPU inspection uses `nvidia-smi`; a chip appearing in the
registry does not imply this provider can discover it through another
accelerator's tooling.

### Option B: a SLURM cluster

First map the chip names you will request to the GPU types reported by your
SLURM installation. For example, if the site advertises `gpu:h100_80gb:8`, use:

```yaml
gres_mapping:
  h100: h100_80gb
```

The value is the GPU **type component**, without `gpu:` or the count. theseus
adds those when building `--gres`. Use your site's actual spelling.

Then add a login endpoint and the partitions it may use:

```yaml
hosts:
  hpc-login:
    type: slurm
    ssh: hpc
    cluster: lab
    partitions:
      - name: gpu
        default: true
    cpu_partitions: [cpu]
    account: myproject
    time: "1-00:00:00"
    uv_groups: [cuda13]
```

Verify the SSH alias reaches a login host with SLURM commands available. theseus
runs availability queries and submits `sbatch` there; the training process runs
on the allocated compute nodes.

| SLURM host field | Default | What it controls |
| --- | --- | --- |
| `type` | Must be `slurm` | Selects the SLURM provider. Omitting it selects plain SSH instead. |
| `ssh` | Required | Login-host SSH target. |
| `cluster` | Required | Filesystem cluster name shared by the allocation. |
| `partitions` | `[]` | GPU partitions, written as names or partition mappings described below. |
| `cpu_partitions` | `[]` | Partition names for CPU-only jobs. If empty, the provider falls back to the names in `partitions`. |
| `account` | `null` | Optional `sbatch --account` value. |
| `qos` | `null` | Optional `sbatch --qos` value. |
| `time` | `"14-0"` | `sbatch --time` value, defaulting to 14 days. Set a quoted value within your site's allowed limit. |
| `exclude` | `[]` | Node names passed to `sbatch --exclude`. The current availability scan does not filter them out itself. |
| `chips` | `null` | Optional per-chip cap on the **total allocation through this provider**. `null` adds no configured cap; `{}` allows no GPU types. Missing keys in a supplied map have a cap of zero. |
| `uv_groups` | `[]` | Remote dependency groups. |
| `env` | `{}` | Worker environment variables. |
| `mem` | `null` | Accepted but not used by the current SLURM submission code. Put RAM requests in the execution or use `submit --mem`. |
| `annotations` | `{}` | Accepted but not used by the current SLURM provider. It does not append arbitrary `sbatch` directives. |

A short partition entry, `partitions: [gpu]`, is equivalent to an entry with
`name: gpu` and the defaults below:

| Partition field | Default | What it controls |
| --- | --- | --- |
| `name` | Required | Actual SLURM partition name. |
| `default` | `false` | Prefer this entry ahead of entries without the flag. This is theseus's preference, not a change to the scheduler's default partition. |
| `constraint` | `null` | Optional feature expression passed to `sbatch --constraint` if this partition is selected. It is not applied during the availability scan. |

For example, use this when a partition requires a site-specific feature:

```yaml
partitions:
  - name: gpu
    default: true
    constraint: "gpu80g"
  - name: gpu-long
```

The solver checks advertised GPU types and available or queueable capacity.
A compatible allocation can still wait in SLURM. The generated job uses one
process per node and a homogeneous GPU count across its nodes. Requests that
need several nodes may be rounded up to obtain an equal per-node count.

CPU and memory requests apply per node: the execution's CPU request becomes
`--cpus-per-task`, defaulting to 2, and its memory request becomes `--mem`,
defaulting to `64G`. The configured `mem` field does not change that default.
See [Submit to SSH or SLURM](../running-remote.md) for the submission workflow.

### Option C: Kubernetes with Volcano

Use a namespace with an existing Volcano queue, a shared PVC, and an image
that can run the theseus bootstrap. You also need a working local `kubectl`.
The provider talks to Kubernetes from the submitting machine; it does not use
an SSH login host.

For example, add a container-visible filesystem layout and a provider:

```yaml
clusters:
  k8s:
    root: /workspace/theseus
    work: /workspace/work

hosts:
  batch:
    type: volcano
    cluster: k8s
    image: registry.example/theseus-training:latest
    pvc_name: shared-workspace
    pvc_mount_path: /workspace
    namespace: training
    queue: training
    chips: {h100: 8}
    gpus_per_node: 8
    num_nodes: 2
    uv_groups: [cuda13]
```

Replace the image and PVC with existing resources. All workers must be able to
mount the PVC and reach one another. Their processes need the bootstrap's tools,
including Bash, and access to install the configured dependencies. theseus does
not create the queue or provision GPU nodes from this file.

#### Select the Kubernetes destination and image

| Volcano field | Default | What it controls |
| --- | --- | --- |
| `type` | Must be `volcano` | Selects the Volcano provider. |
| `cluster` | Required | theseus filesystem cluster name, not a kubeconfig context. |
| `image` | Required | Worker container image. The upload helper uses its own `busybox:1.37` image. |
| `namespace` | `default` | Namespace passed to `kubectl` and used for jobs and the PVC. |
| `queue` | `default` | Existing Volcano queue. The solver checks that its state is `Open`. |
| `kubeconfig` | `null` | Local filename passed to `kubectl --kubeconfig`; omission uses kubectl's normal configuration. |
| `context` | `null` | Context passed to `kubectl --context`; omission leaves kubectl's normal context selection. |

An open queue and a matching inventory are not a guarantee of an immediate
start. The scheduler decides when the submitted workers can run.

#### Make the dispatch files visible inside the pods

| Volcano field | Default | What it controls |
| --- | --- | --- |
| `pvc_name` | Required | Existing PVC mounted into the upload helper and workers. |
| `pvc_mount_path` | `/workspace` | Absolute mount path inside the containers. |
| `pvc_codedrop_path` | `null` | Override the directory used to publish bootstrap and dispatch files. Otherwise use the cluster's `share`, then `<work>/.dispatch`. |

The resolved publication directory must be inside `pvc_mount_path` and cannot
contain `..`. If you choose worker-local `work`, set `pvc_codedrop_path` to a
location on the PVC. For example, with a PVC at `/workspace` and work at
`/tmp/theseus-work`, use `pvc_codedrop_path: /workspace/dispatch`.

#### Describe capacity and per-worker resource defaults

| Volcano field | Default | What it controls |
| --- | --- | --- |
| `chips` | `{}` | Chip counts **per node** in this homogeneous pool. These are declared capacities, not live device discovery. |
| `num_nodes` | `1` | Maximum number of nodes the solver may use from this pool. |
| `gpus_per_node` | `0` | Additional per-node GPU cap. Zero means use the count in `chips` without this extra cap. |
| `gpu_resource_key` | `nvidia.com/gpu` | Kubernetes resource name used to request the allocated accelerators. |
| `cpu` | `null` | CPU quantity per GPU worker when the execution has no CPU request; final fallback is `"1"`. |
| `memory` | `null` | RAM quantity per GPU worker when the execution has no memory request; final fallback is `"64Gi"`. |
| `cpu_cpu` | `null` | CPU quantity for a CPU-only worker; final fallback is `"1"`. This does not inherit `cpu`. |
| `cpu_memory` | `null` | RAM quantity for a CPU-only worker; final fallback is `"64Gi"`. This does not inherit `memory`. |
| `shm_size` | `null` | Optional size of a memory-backed volume mounted at `/dev/shm` in workers. Omission leaves the image/runtime's usual setup. |

In the example above, `{h100: 8}` and `num_nodes: 2` describe up to 16 H100s.
The provider chooses enough nodes for the execution's request, with the same GPU
count per worker. It does not launch two nodes for every job. Multiple workers
are required to land on different physical hosts.

Quote resource quantities: `cpu: "8"`, `cpu_cpu: "500m"`, or
`memory: "64Gi"`. Kubernetes uses `1000m` for one CPU, and `Gi` for binary memory
units. theseus sets matching requests and limits for worker CPU, memory, and
accelerators. See [Kubernetes resource quantities](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/).

An execution-level `--cpu` or `--mem` overrides the corresponding provider
default. An integer execution memory value is interpreted as MiB and rendered
with a `Mi` suffix for Kubernetes. The inventory's quantity fields above should
remain strings.

#### Add scheduling, identity, and environment settings when needed

| Volcano field | Default | What it controls |
| --- | --- | --- |
| `service_account` | `null` | Optional pod `serviceAccountName`. Omission lets Kubernetes choose its default. |
| `priority_class` | `null` | Optional pod `priorityClassName`. This is separate from the top-level provider search order. |
| `node_selector` | `{}` | String label selectors copied into pod `nodeSelector`. |
| `tolerations` | `[]` | List of toleration mappings copied into the pod specification. Use the fields required by your cluster's taints. |
| `labels` | `{}` | Labels applied to the Volcano job and worker pods. theseus also sets its own `theseus.dev/dispatch` pod label. |
| `env` | `{}` | Worker environment variables applied by the bootstrap. They are not Kubernetes `valueFrom` or Secret-reference objects. |
| `uv_groups` | `[]` | Dependency groups installed by the worker bootstrap. |
| `rdma` | `false` | Add a request and limit for the fixed resource key `rdma/rdma_shared_device_a`. Enable only for a pool exposing that resource. |
| `rdma_per_node` | `8` | RDMA count used for a CPU-only allocation when `rdma=True`. GPU workers use their allocated GPU count instead. |
| `helper_resources` | CPU `"1"` and memory `"1Gi"`, for both requests and limits | Resource quantities for the temporary upload helper. Keys use the `requests.<resource>` or `limits.<resource>` form below. |

For example, an administrator might ask you to add:

```yaml
node_selector:
  accelerator: h100
tolerations:
  - key: dedicated
    operator: Equal
    value: training
    effect: NoSchedule
service_account: training
labels:
  team: research
shm_size: "8Gi"
helper_resources:
  requests.cpu: "1"
  requests.memory: "1Gi"
  limits.cpu: "1"
  limits.memory: "1Gi"
```

Use labels and taints that exist on your cluster. Node selectors, tolerations,
service account, and priority class also apply to the upload helper, so overly
restrictive settings can prevent publication before the training job is created.
The helper uses the same PVC and queue, runs without the worker's GPU/RDMA
requests, and is removed after publication. Supplying `helper_resources`
replaces that mapping; include each request and limit you want to retain.

theseus passes toleration objects through to Kubernetes rather than validating
their internal schema. See the [Kubernetes tolerations guide](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/)
for fields such as `key`, `operator`, `value`, `effect`, and optional
`tolerationSeconds`. See [Submit to Volcano / Kubernetes](../volcano.md) for running
and inspecting an allocation.

## 4. Choose the worker environment

Each provider has its own `uv_groups` and `env`. They belong under the host
entry, not the cluster entry:

```yaml
uv_groups: [cuda13]
env:
  HF_HOME: /scratch/huggingface
  TOKENIZERS_PARALLELISM: "false"
  XLA_PYTHON_CLIENT_MEM_FRACTION: "0.85"
```

`uv_groups` contains dependency-group names from the repository's
`pyproject.toml`. The bootstrap runs `uv sync --no-default-groups`, so it does
not inherit your local default group selection. Pick the hardware group needed
by the worker, such as `cuda12` or `cuda13`, and any additional job dependencies.
The hardware groups `cpu`, `cuda12`, `cuda13`, and `tpu` are mutually exclusive.
A group installs dependencies; it does not provision that accelerator or make a
provider capable of discovering it.

For a CPU-only request (`submit -n 0`), the providers replace any hardware group
with `cpu`, retain other groups, hide CUDA devices, and set `JAX_PLATFORMS=cpu`.
This happens even if the host normally runs GPU jobs.

Environment values must be strings. Quote numeric and boolean-looking values
as above. They are copied into the resolved dispatch and exported by the worker
bootstrap before dependency installation and job execution. Values are passed
literally; use actual paths rather than expecting `$HOME` to expand inside a
value. This mapping is not a shell startup script.

For a single submission, `--extras GROUP` appends a dependency group and
`--env KEY=VALUE` overrides a worker environment variable. Both flags are
repeatable. Use them for a run-specific change; edit the inventory for a default
you want future submissions to share.

## 5. Choose the provider search order

After adding providers, put their **host entry names** into `priority`:

```yaml
priority: [workstation, hpc-login, batch]
```

theseus tries each in order and uses the first provider that returns a compatible
allocation. Unlisted hosts remain eligible afterward. Unknown names in
`priority` are skipped. A compatible queued SLURM allocation can be selected
before a later provider is tried, so order providers by your actual preference.

For a particular run, `--cluster` and `--exclude-cluster` filter by **cluster
names**, not host names. In this guide, that means `--cluster lab` or
`--cluster k8s`, not `--cluster hpc-login` or `--cluster batch`.

## 6. Check the file before submitting work

First load the file without querying remote machines or launching anything:

```bash
uv run python - <<'PY'
from pathlib import Path
from theseus.execute.config import DispatchConfig

config = DispatchConfig.load(Path.home() / ".theseus.yaml")
for name, host in config.hosts.items():
    if host.cluster not in config.clusters:
        raise ValueError(f"{name}: unknown cluster {host.cluster!r}")
    print(name, host.type, "->", host.cluster)
PY
```

Check that every provider you intended appears in the output. The loader
ignores unsupported host types and unrecognized fields, so a successful load
alone does not catch every typo. Compare your spelling and indentation with the
field tables, and avoid putting settings under a different provider type.

Next, use a complete execution YAML you have already configured, here called
`train.yaml`, to check selection:

```bash
uv run theseus submit configuration-check train.yaml \
  --dispatch-config ~/.theseus.yaml \
  --cluster lab --chip h100 -n 2 --cpu 8 --mem 64Gi \
  --dry-run
```

Change the cluster and chip to match your inventory. This validates the execution
and queries provider availability, then prints the resolved dispatch. It does
not publish files or launch the job. Check the selected provider, cluster paths,
chip count, worker count, dependency groups, and environment in the output.
Scheduler-specific defaults such as SLURM account or Volcano service account
remain in the inventory and are applied when the provider constructs the actual
submission; they are not all shown in the dry-run dispatch JSON.

If selection fails, check these in order:

1. Confirm the command names the file you edited and that its host entries loaded.
2. Confirm `host.cluster` and any `--cluster` filter use an existing cluster key.
3. Confirm chip keys match [Available Chips](../../Design/chips.md); for SLURM, also
   check `gres_mapping` against the types advertised by the partitions.
4. Check noninteractive SSH access or the configured kubectl context and queue.
5. Check the requested count against host caps, node counts, and per-node capacity.

Once the result matches your intended allocation, use the appropriate
[SSH/SLURM](../running-remote.md) or [Volcano](../volcano.md) submission guide. Remove
`--dry-run` only when you are ready for that command to launch work.
