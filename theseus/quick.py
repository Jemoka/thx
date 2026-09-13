"""
Interactive job sessions for fresh runs, branching, and resumption.

Fresh job:
    with quick("/path/to/theseus") as q:
        q.build(MyTrainer, "experiment")  # registered job names also work
        q.config.training.per_device_batch_size = 16
        job = q.create()  # construct and set up state once; ready to inspect
        job()  # run the prepared job

Continue a saved job:
    with quick("/path/to/theseus") as q:
        q.find().spec(run="source").checkpoint().resume()
        # resume() loads the selected checkpoint's config immediately.
        q.config.optimization.lr = 1e-5
        q.build()  # from_node: saved job class + edited config; constructs now
        job = q.create()
        job()  # continues with the prepared resume state

Branch from a saved job:
    with quick("/path/to/theseus") as q:
        q.find().spec(run="source").checkpoint().branch()
        q.config.optimization.lr = 3e-5
        q.build(name="new-experiment")  # constructs the saved class with a new name
        job = q.create()
        job()  # runs the new branch

Time travel debugging (a checkpoint from a base trainer):
    import jax
    import matplotlib.pyplot as plt
    import seaborn as sns
    from theseus.model.attention.base import SelfAttention

    with quick("/path/to/theseus") as q:
        q.find().spec(run="source").checkpoint().resume()
        q.build()
        job = q.create()

        paths = job.find(SelfAttention)
        with job.debug(paths[0]) as (layer, inputs):
            x = inputs.x
            print("Captured input:", x.shape, x[0, 0, :8])
            projected = layer.c_attn(x)  # GPT SelfAttention's existing projection
            ax = sns.heatmap(jax.device_get(projected[0]))
            ax.set(xlabel="QKV feature", ylabel="Token", title="Checkpoint projection")
            plt.show()
        # Debugger closes here, restoring JIT/config and discarding exploration.
        # find()/debug() replay the saved batch at the checkpoint sequence.
        # Training advances once before consuming the next batch.
        job()  # continue training after inspection

Selection and lifecycle:
    Each find() returns an independent query. Chain filters on that builder;
    separate find() calls never accumulate filters. build() leaves held query
    builders intact. all()/select() consume and reset their builder's filters.

    Queries default to ascending sequence order; branch()/resume() take the last
    match (highest seq). Use latest() for the most recently written checkpoint,
    or sort(...) to override the order. branch()/resume()
    replace q.config with its saved configuration. Missing saved config raises
    immediately, leaving the previous selection/config intact. Select before
    create(), and make config edits AFTER selection, before job construction.

    build(job, name) configures a fresh session and clears base/resume selection.
    To use an explicit class with a checkpoint, build it first, then select.
    build(job=None) requires a selected node and calls RestoreableJob.from_node
    with q.config plus any explicit config overrides. Resume restores saved job
    identity; branch accepts a new name/project/group (default name: "local").
    create() sets up that instance's saved state without running training.
    Call job() on the returned instance to run with the prepared branch/resume
    mode and any interactive state edits. q() is shorthand for creating and
    running the cached job with the selected mode. setup() is idempotent:
    subsequent setup calls preserve prepared state.
    Config edits after construction do
    not reconfigure the instance. Repeating identical build() arguments preserves
    the session; changed arguments replace it.

    q.shard(tp=2, fsdp=False, zero=True) configures the next job before create().

    q.spec() snapshots a built session as a Combobulation without creating or
    running a job. Extend it with branch()/resume(), set resources with gpu()
    or cpu(), and pass it as DispatchSpec(job=...). DispatchSpec supplies the
    execution name/project/group and hardware; checkpoint bases must be
    accessible from the dispatch's root. Live in-memory job state is not exported.

    To submit through the CLI's hardware solver and configured providers:
        OmegaConf.save(q.spec().gpu(2).serialize(), "train.yaml")
        # uv run theseus submit experiment train.yaml

    init(root) opens the same session without a context manager; call close()
    afterward. close() finishes the live job and restores the prior config
    context. Root defaults to $THESEUS_ROOT or ".". Queries need no active job.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, TYPE_CHECKING, cast

from omegaconf import DictConfig, OmegaConf, open_dict

from theseus.base import ExecutionSpec, Node, ShardingPolicy
from theseus.config import build, _current_config
from theseus.execute.combobulator import Combobulation, Step
from theseus.job import RestoreableJob
from theseus.registry import JOBS
from theseus.store import ObjectReader, QueryBuilder


if TYPE_CHECKING:
    from contextvars import Token


class QuickJob:
    """Query a root and optionally configure and run one active job."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root_dir = str(root_dir or os.environ.get("THESEUS_ROOT") or ".")
        self._store = ObjectReader.local(self._root_dir)
        self._instance: Any = None
        self._config: DictConfig | None = None
        self._config_token: Token[Any] | None = None
        self._build_args: tuple[Any, ...] | None = None
        self.base: Node | None = None
        self._resume = False
        self._sharding = ShardingPolicy()

    #### session lifecycle ####

    def build(
        self,
        job: Any | str = None,
        name: str | None = None,
        project: str | None = None,
        group: str | None = None,
        config: DictConfig | None = None,
    ) -> QuickJob:
        """Configure a job; identical arguments preserve the active session.

        New builds clear branch/resume selection but leave existing queries intact. Select a base
        after building. Argument comparison uses a detached snapshot of supplied
        overrides, so editing ``q.config`` does not rebuild the job.

        Omit job after branch()/resume() to construct the saved job immediately.
        Pass config overrides here; create() returns the restored instance.
        """
        args: tuple[Any, ...]
        if job is None:
            if self.base is None:
                raise RuntimeError(
                    "Select a node with branch() or resume() before build()."
                )
            args = (
                None,
                self._sharding,
                self.base,
                self._resume,
                name,
                project,
                group,
                OmegaConf.to_container(config, resolve=False)
                if config is not None
                else None,
            )
            if self._build_args == args:
                return self
            spec = ExecutionSpec.local(
                self._root_dir,
                name=name or "local",
                project=project,
                group=group,
                shard=self._sharding,
            )
            runtime_cfg = OmegaConf.merge(
                self.config, config if config is not None else {}
            )
            instance: RestoreableJob[Any]
            instance, cfg = RestoreableJob.from_node(
                self.base, spec, runtime_cfg=runtime_cfg, resume=self._resume
            )
            self.close()
            self._instance = instance
            self._job_cls = type(instance)
            self._config = cfg
            self._config_token = _current_config.set(cast(OmegaConf, cfg))
            self._build_args = args
            return self

        if name is None:
            raise ValueError("Provide a name when building an explicit job.")
        job_cls: Any
        if isinstance(job, str):
            if job not in JOBS:
                raise ValueError(f"Job '{job}' not found in registry.")
            job_cls, job_name = JOBS[job], job
        else:
            job_cls = job
            job_name = next((n for n, cls in JOBS.items() if cls is job), "unknown")

        overrides = (
            OmegaConf.to_container(config, resolve=False)
            if config is not None
            else None
        )
        args = (job_cls, job_name, name, project, group, overrides, self._sharding)
        if self._build_args == args:
            return self

        # Validate the replacement before closing a working session.
        components = job_cls.config()
        cfg = cast(
            "DictConfig",
            build(
                *(components if isinstance(components, (list, tuple)) else [components])
            ),
        )
        if config is not None:
            with open_dict(cfg):
                cfg = cast("DictConfig", OmegaConf.merge(cfg, config))
            OmegaConf.set_struct(cfg, True)

        self.close()
        self._job_cls = job_cls
        self._name = name
        self._project = project
        self._group = group
        self._instance = None
        self.base = None
        self._resume = False
        self._config = cfg
        self._config_token = _current_config.set(cast(OmegaConf, cfg))
        self._build_args = args
        return self

    @property
    def config(self) -> DictConfig:
        """Editable config from build() or the selected checkpoint."""
        if self._config is None:
            raise RuntimeError(
                "Call build(job, name) or select a checkpoint before accessing configuration."
            )
        return self._config

    def close(self) -> None:
        """Finish the active job's resources and restore the prior config context."""
        if self._config_token is not None:
            try:
                if self._instance is not None:
                    self._instance.finish()
            finally:
                _current_config.reset(self._config_token)
                self._config_token = None
                self._config = None
                self._build_args = None

    #### job execution ####

    def spec(self) -> Combobulation:
        """Snapshot the built job, config, lineage, and sharding for dispatch.

        Call build() first. The returned execution is independent of this
        session and includes config edits made up to this call. It describes
        execution from the selected checkpoint, not the live instance's state.
        """
        if self._config is None or self._build_args is None:
            raise RuntimeError("Call build() before exporting an execution spec.")
        node = self.base
        step = Step.new(
            self._job_cls,
            base=(lambda query: query.node(node)) if node is not None else None,
            resume=self._resume,
        )
        return Combobulation.model_validate(
            {
                "steps": (step,),
                "sharding": self._sharding,
                "config": OmegaConf.to_yaml(self.config, resolve=False),
            }
        )

    def shard(
        self,
        tp: int = 1,
        fsdp: bool = False,
        zero: bool = True,
        activation_checkpointing: bool = False,
    ) -> "QuickJob":
        """Configure parallelism before creating the job."""
        policy = ShardingPolicy(
            tp=tp,
            fsdp=fsdp,
            zero=zero,
            activation_checkpointing=activation_checkpointing,
        )
        if self._instance is not None and policy != self._sharding:
            raise RuntimeError("Configure sharding before creating the job")
        self._sharding = policy
        return self

    def create(self) -> Any:
        """Create and set up the configured job without running it.

        The node and lineage mode selected by :meth:`find` are applied to the
        instance. Repeated calls preserve the prepared state and run node.
        """
        if self._config is None or self._build_args is None:
            raise RuntimeError("Call build(job, name) before creating the job.")
        if self._instance is None:
            self._instance = self._job_cls.local(
                self._root_dir,
                base=self.base,
                name=self._name,
                project=self._project,
                group=self._group,
                shard=self._sharding,
            )
            self._instance._resume_required = self._resume
        self._instance.setup(resume=self._resume)
        return self._instance

    def find(self) -> "QuickQuery":
        """Query the root, with or without an active job.

        The query reads the object store rooted at this quick job's output path.
        Each call returns an independent builder with no accumulated filters.
        Use all()/select() for reads. To select an execution base, call branch()
        or resume() before create(). Selection loads checkpoint configuration;
        call build() without a job to construct the saved job after editing it.
        """
        return QuickQuery(self, self._store)

    def __call__(self) -> Any:
        """Create and run the job with the selected resume mode."""
        return self.create()(resume=self._resume)


class QuickQuery(QueryBuilder):
    """Select a checkpoint, ordered by sequence unless explicitly overridden."""

    def __init__(self, quick_job: QuickJob, store: ObjectReader) -> None:
        super().__init__(store)
        self.sort("_x_seq")
        self.quick_job = quick_job

    def branch(self) -> QuickJob:
        """Select a branch base and load its saved configuration."""
        return self._select(resume=False)

    def resume(self) -> QuickJob:
        """Select a resume base and load its saved configuration."""
        return self._select(resume=True)

    def _select(self, resume: bool) -> QuickJob:
        if self.quick_job._instance is not None:
            raise RuntimeError("Select a base before creating the job.")
        nodes = self.all()
        if not nodes:
            raise LookupError("No nodes matched the quick job query.")
        node = nodes[-1]
        values = self.store.query().node(node).select(keys=["blob"])
        blob = values[0].get("blob") if values else None
        if not isinstance(blob, Path) or not (blob / "config.yaml").is_file():
            raise ValueError(f"No saved configuration for node={node.serialize()}")
        cfg = OmegaConf.load(blob / "config.yaml")
        if not isinstance(cfg, DictConfig):
            raise ValueError("Checkpoint configuration must be a mapping")
        OmegaConf.set_struct(cfg, True)

        # Validate the checkpoint before replacing the active configuration.
        if self.quick_job._config_token is not None:
            _current_config.reset(self.quick_job._config_token)
        self.quick_job._config = cfg
        self.quick_job._config_token = _current_config.set(cast(OmegaConf, cfg))
        self.quick_job.base = node
        self.quick_job._resume = resume
        return self.quick_job


@contextmanager
def quick(root_dir: str | Path | None = None) -> Generator[QuickJob, None, None]:
    """Open a query/run session and close its active job on context exit."""
    session = init(root_dir)
    try:
        yield session
    finally:
        session.close()


def init(root_dir: str | Path | None = None) -> QuickJob:
    """Open a session at root_dir, defaulting to $THESEUS_ROOT or ".".

    Queries are available immediately. Call build(job, name, ...) before editing
    config or executing a job, and close() when finished.
    """
    return QuickJob(root_dir)
