# JuiceFS Integration

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Use JuiceFS when several machines should see the same theseus data, checkpoints,
and object store without copying a root directory between runs. theseus can
mount an existing volume during remote bootstrap. It does not provision the
metadata service, create the backing object store, or format the volume.

## Prepare the shared volume

Start with an existing JuiceFS volume and its metadata URL. Every worker needs
network access and credentials for its metadata service and backing storage,
plus an environment that permits FUSE mounts. For volume creation, follow the
[JuiceFS setup documentation](https://juicefs.com/docs/community/getting_started/).

Choose which theseus directories belong on shared storage. By default, `data`,
`checkpoints`, `objects`, `results`, and `status` are under the cluster's `root`.
Directory overrides can move individual locations elsewhere. The source-code
publication directory `work` and early bootstrap logs need to be accessible
before theseus mounts the root.

## Configure the cluster mount

Add the mount settings to a cluster in your dispatch configuration. This is a
cluster fragment; retain your existing `hosts` entries pointing to `shared`:

```yaml
clusters:
  shared:
    root: /home/research/theseus-root
    work: /home/research/theseus-work
    log: /home/research/theseus-logs
    mount: redis://metadata.example/1
    cache_size: "102400"
    cache_dir: /scratch/juicefs-cache
    all_squash: "23296:100"
```

Replace the paths, URL, and numeric identity with your deployment's values.
Use `clusters.<name>.mount`; the legacy top-level `mount` field is not used by
current providers. See [Remote Configuration](providers.md) for the
complete inventory and provider setup.

| Field | What theseus does with it |
| --- | --- |
| `root` | Uses this worker path as the mount point. |
| `mount` | Passes this metadata URL to `juicefs mount`. Omission means no bootstrap-managed mount. |
| `cache_size` | Passes the string to `--cache-size`; `"102400"` requests 100 GiB in MiB units. |
| `cache_dir` | Passes the path to `--cache-dir`; use suitable worker-local storage. |
| `all_squash` | Passes the string to `--all-squash`, mapping client users to the specified numeric UID:GID. |

Cache and identity options are optional. The bootstrap also passes `--umask=0000`
and `-d`. JuiceFS documents `all-squash` and this mount umask option as introduced
in v1.3; an already installed older client is not upgraded by theseus. Check
`juicefs mount --help` on the actual worker. See the
[JuiceFS mount reference](https://juicefs.com/docs/community/command_reference/#juicefs-mount)
for option semantics and client defaults.

## Understand what happens at launch

The dispatcher includes the cluster settings in the worker specification. The
bootstrap reads that specification, installs a missing JuiceFS client on Linux
using its user-local installer, and checks whether `root` is already a mount
point. If it isn't, it creates the directory and mounts there before execution.
If it is, it reuses the mount without changing its options or verifying that it
is the intended volume.

Changing YAML therefore does not update an existing mount's cache, identity, or
volume selection. Check the active mount before concluding that a new setting
took effect. At cleanup, bootstrap attempts to unmount the mount it created;
it leaves a pre-existing mount alone. Coordinate shared mount ownership so one
job's cleanup does not disrupt another job using the same mount point.

SSH, SLURM, and Volcano dispatch carry these cluster fields. On SLURM, the
compute nodes need the mount prerequisites, not just the login host. On Volcano,
the container and cluster security configuration must permit FUSE; declaring
`mount` does not grant those permissions. Code publication still uses Volcano's
configured PVC and its publication path must be inside that PVC.

## Open the same root from your notebook

A local `quick` session does not invoke remote bootstrap or automatically mount
from `~/.theseus.yaml`. Mount the volume on the notebook machine separately,
then pass its local mount point:

```python
from theseus.quick import quick

with quick("/home/research/theseus-root") as q:
    rows = q.find().spec(run="train-gpt").select(keys=["train/loss"])
    print(rows)
```

The absolute mount path can differ between machines, but it must expose the
same intended data and object-store tree. Merely giving two clusters the same
root path does not make their disks shared. Likewise, mounting an empty volume
over an existing directory does not migrate the directory's old contents into
that volume. Prepare or copy data into the mounted volume before dispatching.

## Diagnose a permission failure

Run these read-only checks on the machine and under the account that actually
failed to write. On Linux:

```bash
id
namei -l /home/research/theseus-root/objects/values
getfacl /home/research/theseus-root/objects/values
findmnt -T /home/research/theseus-root/objects/values
juicefs version
```

Check traversal permissions on every parent directory and write permission on
the destination. Numeric ownership can differ from the notebook user's UID.
For example, a directory owned by `23296:100` with mode `775` does not give an
unrelated user write access through the “other” permission bits.

`all_squash: "23296:100"` is a mount request, not a recursive ownership repair.
It neither changes an existing mount nor rewrites existing directory permissions.
Check the mount's effective settings, filesystem ACLs, and any kernel permission
checks as well as the configured mapping. The bootstrap's umask affects new
entries; it does not repair existing ones. Fix the specific ownership, ACL, or
mount mismatch with whoever manages the volume. Retrying a permanent permission
failure will not grant access.

Keep local cache space and the shared persistent root distinct when diagnosing
capacity failures. Inspect bootstrap storage messages to distinguish client
installation, mount, and later application writes. Once the mount is working,
normal [remote submission](../../Tutorials/remote.md) and
[checkpoint inspection](../../Tutorials/inspecting-runs.md) use the shared files.
