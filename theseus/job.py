"""
There are two types of nodes and one type of edge.

Nodes:
- TrainingNode (takes a checkpoint or nothing, and continues training it)
- EvaluationNode (takes a checkpoint, and labels it with evaluation)

Edges:
- Apply

Examples:
For [T]raining and [E]valuation nodes.

Imagine I'm doing a "standard" ML training recipe. This may look like:

[T] pretrain -> [T] midtrain -> [T] RL
 |               |               |
 v               v               v
[E] blimp       [E] smolchat    [E] gsm8k
[E] hellaswag

In fact each of these is one training step:

[T] -> [T] -> [T] -> [T]
        |
        v
       [E]

I could also be doing some topology transfer. For instance, you
may wonder if you can thoughtbubblesify a normal transformer. So
this may look like:

[T] load (contrib.Qwen) -> [T] midtrain (forking.Thoughtbubbles)
 |                          |
 v                          v
[E] gsm8k                  [E] gsm8k

The edges are actually a bit of a simplification, in the sense that
each "checkpoint" is actually an edge, some of which has evaluation.
"""

import json
import random
from dataclasses import dataclass
from abc import abstractmethod
from pathlib import Path
from typing import (
    Generic,
    TypeVar,
    Any,
    Type,
    Self,
    List,
    Union,
    Optional,
    Mapping,
    TYPE_CHECKING,
    cast,
)

import numpy as np
from loguru import logger
from omegaconf import OmegaConf

import jax
from jax.experimental import multihost_utils

from theseus.store import ObjectStore, RecordStore, ValueRow
from theseus.checkpoint import CheckpointManager
from theseus.base import _BaseJob, ExecutionSpec, Node, PyTree, JobSpec, ShardingPolicy
from theseus.config import current_config, configure, configuration, field

if TYPE_CHECKING:
    from comet_ml import CometExperiment


C = TypeVar("C")


class BasicJob(_BaseJob, Generic[C]):
    JOB_NAME: str
    STORE: Type[ObjectStore] = ObjectStore

    @classmethod
    def config(cls) -> Union[Type[C], List[Type[Any]]]:
        raise NotImplementedError()

    def __init__(self, spec: ExecutionSpec, base: Optional[Node] = None) -> None:
        #### configure the job ####
        cfg_type = self.config()
        self.args: C
        if isinstance(cfg_type, list):
            self.args = configure(cfg_type[0])
        else:
            self.args = configure(cfg_type)

        #### initialize storage details ####
        self.spec = spec
        self.store = self.STORE(spec.hardware, spec.execution_id, spec.tag)
        self.base = base
        self.node = Node(name=self.node_name(self.spec))
        self._resume_required = False
        self._node_written = False
        self._setup_complete = False

    #### you can call these ####
    @classmethod
    def local(
        cls,
        root_dir: str,
        base: Optional[Node] = None,
        name: str = "local",
        project: str | None = None,
        group: str | None = None,
        shard: ShardingPolicy | None = None,
    ) -> Self:
        spec = ExecutionSpec.local(
            root_dir,
            name=name,
            project=project,
            group=group,
            shard=shard,
        )
        return cls(spec, base=base)

    def tick(self) -> None:
        """Record the current node and advance to its logical successor."""
        if not getattr(self, "_setup_complete", False):
            raise RuntimeError("You can't tick a job without kicking it off.")

        if not getattr(self, "_node_written", False):
            self.store.value(self.node, {})
        self.node.update(self.node.next())
        self._node_written = False

    #### you need to implement these ####
    @abstractmethod
    def state_restore(self, state: ValueRow) -> None:
        """"""

        raise NotImplementedError()

    @abstractmethod
    def state_init(self) -> None:
        """Initialize the job as the beginning of a branch.

        Notes:
            Specifically, calling this function should allocate hardware
            memory, and start initialization etc., and start the job from
            scratch. A node has been created
        """

        raise NotImplementedError()

    def state_reset(self) -> None:
        """Reset state hook. We'll call this before every non-resume call."""

        # by default, we don't do anything to reset
        pass

    @abstractmethod
    def run(self) -> None:
        """Run the job, assuming all hosts have setup"""
        raise NotImplementedError()

    #### metadata ####
    @staticmethod
    def node_name(spec: JobSpec) -> str:
        """Return a unique name for the node, based on the spec"""
        return f"{spec.project or 'general'}.{spec.group or 'default'}.{spec.name}"

    def main_process(self) -> bool:
        res: bool = jax.process_index() == 0
        return res

    def _synchronize_node(self, node: Node) -> Node:
        is_source = self.main_process()
        payload = node.model_dump_json().encode() if is_source else b""
        payload_size = multihost_utils.broadcast_one_to_all(
            np.asarray(len(payload), dtype=np.int32),
            is_source=is_source,
        )
        size = int(np.asarray(payload_size).item())
        buffer = (
            np.frombuffer(payload, dtype=np.uint8)
            if is_source
            else np.zeros(size, dtype=np.uint8)
        )
        synchronized = multihost_utils.broadcast_one_to_all(
            buffer,
            is_source=is_source,
        )
        return Node.model_validate_json(
            np.asarray(synchronized, dtype=np.uint8).tobytes()
        )

    #### our implementations of things ####
    def finish(self) -> None:
        self.store.close()

    def setup(self, resume: bool = False) -> None:
        """Prepare state and the run node once, without running or synchronizing.

        The first successful setup fixes the branch/resume mode. Later calls
        preserve the existing state and node, including interactive edits.
        Resuming selects the saved node; advancing belongs to the running job.
        """
        if self._setup_complete:
            return
        if self._resume_required and not resume:
            logger.warning(
                "JOB {} | RESUME MISMATCH: from_node(resume=True) but "
                "setup(resume=False); honoring the restore-time resume",
                self.spec.name,
            )
        resume = resume or self._resume_required

        logger.debug(f"JOB {self.spec.name} | building state")

        # initialize the node
        if self.base is not None:
            logger.debug(
                f"JOB {self.spec.name} | restoring state baseof={self.base.serialize()}"
            )
            # restore the state
            values = self.store.query().node(self.base).select()
            self.state_restore(values[0] if values else {})
            if not resume:
                # construct the next node from the DAG
                # which is a CHILD of the node from which its restored
                self.node.update(
                    Node(
                        name=self.node_name(self.spec),
                        parent=self.base.serialize(),
                    )
                )
            else:
                self.node.update(self.base)
        else:
            logger.debug(f"JOB {self.spec.name} | starting anew")
            self.state_init()

        # reset, if needed
        if not resume:
            self.state_reset()
        self._setup_complete = True

    def __call__(self, resume: bool = False) -> None:
        logger.info(f"JOB {self.spec.name} | starting")
        logger.debug(f"JOB {self.spec.name} | pre-start sync")
        multihost_utils.sync_global_devices(f"{self.spec.name}:start")
        self.setup(resume=resume)

        # Serialize the node and synchronize it across all hosts.
        self.node.update(self._synchronize_node(self.node))
        self.store.value(self.node, {})
        self._node_written = True

        # print node info
        logger.info(
            f"JOB {self.spec.name} | synchronized and starting with nodeid={self.node.serialize()}"
        )

        self.run()

        #### footer: sync nd wait for everyone
        logger.debug(f"JOB {self.spec.name} | finished, waiting for everyone...")
        multihost_utils.sync_global_devices(f"{self.spec.name}:finish")
        self.store.value(self.node, {"_x_finished": True})
        self._node_written = True
        self.finish()
        logger.info(f"JOB {self.spec.name} | finished")


class CheckpointedJob(BasicJob[C], Generic[C]):
    def __init__(self, spec: ExecutionSpec, base: Optional[Node] = None) -> None:
        super().__init__(spec, base=base)

        self.chkpt_manager = CheckpointManager()
        self.key = jax.random.PRNGKey(0)

    #### you need to implement these ####
    @property
    @abstractmethod
    def template(self) -> PyTree[Any]:
        """ "Template PyTree with which to do loading/saving; could be abstract shapes."""

        raise NotImplementedError()

    @abstractmethod
    def initialize(self) -> None:
        """Initialize and install a complete state."""

        raise NotImplementedError()

    @abstractmethod
    def surgery(self, partial: PyTree[bool]) -> PyTree[Any]:
        """Initialize marked leaves and return ``None`` for all other leaves."""

        raise NotImplementedError()

    @abstractmethod
    def apply(self, state: PyTree[Any], metadata: ValueRow) -> None:
        """ "Load state to be consistent with the above.

        Notes:
            Could probably just be self.state=state + step accounting.
            But usually involves a bit more wedding planning.
        """

        raise NotImplementedError()

    def reset(self) -> None:
        """Reset state hook. For instance for midtraining."""

        super().state_reset()

    #### our implementations of things ####
    def finish(self) -> None:
        try:
            self.chkpt_manager.close()
        finally:
            super().finish()

    def save(
        self,
        state: PyTree[Any],
        metadata: dict[str, float | int | str],
    ) -> None:
        """ "Save the state into the current node."""

        # perform quick input validation and ignore checkpoint if the metadata has
        # non-scalar values, which will blow up PyArrow
        if not all(isinstance(v, (int, float, str, bool)) for v in metadata.values()):
            logger.error(
                "JOB | Checkpoint metadata contains non-scalar values. "
                "Your checkpoint manager maybe poisoned by this, so we "
                "will ignore your checkpoint.",
            )
            logger.debug("JOB | bad metadata={}", metadata)
            return

        # seralize the random states and store them with the checkpoint
        randomness = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "jax_random": int(self.key[0]),
        }
        checkpoint_metadata = dict(metadata)
        checkpoint_metadata["_x_checkpoint"] = True

        with self.store.blob(self.node, checkpoint_metadata) as path:
            logger.debug("CHECKPOINT | starting save")
            self.chkpt_manager.save(state, path / "checkpoint")
            logger.debug("CHECKPOINT | saved training state")
            if self.main_process():
                np.save(path / "rng.npy", np.array(randomness, dtype=object))
                logger.debug("CHECKPOINT | saved randomness")
                cfg = current_config()
                if cfg is not None:
                    with open(path / "config.yaml", "w") as df:
                        df.write(OmegaConf.to_yaml(cfg))
                logger.debug("CHECKPOINT | saved configuration")
                job_spec_data = {
                    field_name: getattr(self.spec, field_name)
                    for field_name in JobSpec.model_fields
                }
                with open(path / "job.json", "w") as df:
                    json.dump(job_spec_data, df)
                logger.debug("CHECKPOINT | saved job spec")
        self._node_written = True

    def state_restore(self, state: ValueRow) -> None:
        """Restore and apply the checkpoint referenced by a stored node view."""

        blob = state.get("blob")
        if not isinstance(blob, (str, Path)):
            node = self.base.serialize() if self.base is not None else "unknown"
            raise ValueError(f"No restorable object is associated with node={node}")

        path = Path(blob)
        try:
            randomness = np.load(path / "rng.npy", allow_pickle=True).item()
            random.setstate(randomness["python_random"])
            np.random.set_state(randomness["numpy_random"])
            self.key = jax.random.PRNGKey(randomness["jax_random"])
        except (EOFError, FileNotFoundError):
            self.key = jax.random.PRNGKey(0)

        restored, partial = self.chkpt_manager.restore(
            path / "checkpoint",
            self.template,
            allow_partial=True,
        )
        if partial is not None:
            initialized = self.surgery(partial)
            restored = jax.tree.map(
                lambda old, new, needed: new if needed else old,
                restored,
                initialized,
                partial,
                is_leaf=lambda value: value is None,
            )
        self.apply(restored, state)

    def state_init(self) -> None:
        # yes, that indeed is an alias, but it means
        # that downstream users only has to care about one
        # set of abstractions
        self.initialize()

    def state_reset(self) -> None:
        # ditto the above
        self.reset()


class LoggingJob(BasicJob[C], Generic[C]):
    """Job that reads and records scalar logs and PyTree payloads.

    This class deliberately does not implement checkpointing. PyTree records use
    the simple :class:`theseus.store.RecordStore` path: they are accepted only on
    one host and inefficiently serialized as MessagePack by JAX process 0. Stateful
    jobs should also inherit :class:`CheckpointedJob`, which owns distributed Orbax
    checkpoints, synchronization, randomness, configuration, and job metadata.
    """

    STORE = RecordStore
    store: RecordStore

    #### you can call these ####

    def get(self, node: Node | str) -> dict[str, Any]:
        """Return a node's folded values and decoded MessagePack records.

        Args:
            node: A node or its serialized identity.

        Returns:
            The object-store values. When the node's blob directory contains
            MessagePack records, they are returned as ``payload`` in filename order.
            The original blob path remains available as ``blob``.
        """
        return self.store.get_record(node)

    def log(self, payload: Mapping[str, Any]) -> None:
        """Record scalar values on the job's current node.

        Args:
            payload: Named scalar values. NumPy and JAX scalars are converted to
                their host Python representations.

        Raises:
            RuntimeError: If the job has not started or the value writer failed.
            ValueError: If a value is not scalar.
            TypeError: If a scalar has an unsupported type.
        """
        if not hasattr(self, "node"):
            raise RuntimeError("You can't log something without kicking it off.")
        self.store.value(self.node, self._normalize_log(payload))
        self._node_written = True

    def artifact(
        self,
        suffix: str,
        description: Mapping[str, Any],
        payload: PyTree[Any],
    ) -> None:
        """Store a PyTree artifact on the job's current node.

        The artifact is written only by JAX process 0. Use log() for scalar logs
        and artifact() for analysis data and other payloads. Model checkpoints
        use save().

        Args:
            suffix: Unique filename stem within the node. Reusing it replaces the
                prior artifact.
            description: Scalar metadata used to discover the artifact.
            payload: PyTree serialized as MessagePack by the record store.

        Raises:
            RuntimeError: If the job has not started or a writer failed.
            ValueError: If ``suffix`` is invalid or description values are not
                scalar.
            TypeError: If a description scalar has an unsupported type.
        """
        if not hasattr(self, "node"):
            raise RuntimeError("You can't store an artifact without kicking it off.")
        self.store.artifact(
            self.node,
            suffix,
            self._normalize_log(description),
            payload,
        )
        if self.main_process():
            self._node_written = True

    #### our things ####

    @staticmethod
    def _normalize_log(
        payload: Mapping[str, Any],
    ) -> dict[str, float | int | str]:
        """Convert scalar log values to their host Python representations."""
        normalized: dict[str, float | int | str] = {}
        for name, value in payload.items():
            host_value = np.asarray(jax.device_get(value))
            if host_value.ndim != 0:
                raise ValueError(
                    f"Log value {name!r} must be scalar, got shape {host_value.shape}"
                )

            scalar = host_value.item()
            if not isinstance(scalar, (float, int, str)):
                raise TypeError(
                    f"Log value {name!r} has unsupported type {type(scalar)!r}"
                )
            normalized[name] = scalar

        return normalized


@dataclass
class CometLoggingConfig:
    remote: bool = field("logging/remote", default=True)


class CometLoggingJob(LoggingJob[C], Generic[C]):
    """Mirror scalar logs to Comet when logging/remote is enabled.

    Include config() in the concrete job's schemas and call super().run() before
    its workload. One experiment follows a nonce across resumes;
    sequence numbers are metric steps. PyTree records remain in RecordStore.
    """

    _comet: "CometExperiment | None" = None

    @classmethod
    def config(cls) -> List[Type[Any]]:
        return [CometLoggingConfig]

    def run(self) -> None:
        """Open remote logging on process 0 before the concrete workload."""
        if not self.main_process() or not configure(CometLoggingConfig).remote:
            return
        if self._comet is not None:
            return

        import comet_ml

        key = comet_ml.get_experiment_key(self.node.nonce)
        self._comet = comet_ml.start(
            experiment_key=key,
            mode="get_or_create",
            project_name=self.spec.project,
            experiment_config=comet_ml.ExperimentConfig(auto_output_logging="simple"),
        )
        self._comet.set_name(self.spec.name)
        if self.spec.group is not None:
            self._comet.add_tags([self.spec.group])
        cfg = current_config()
        if cfg is not None:
            self._comet.log_parameters(
                cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            )

    def log(self, payload: Mapping[str, Any]) -> None:
        """Keep local scalar records and mirror them at the current sequence."""
        super().log(payload)
        if self._comet is not None:
            normalized = self._normalize_log(payload)
            normalized["node"] = self.node.serialize()
            self._comet.log_metrics(normalized, step=self.node.seq)

    def finish(self) -> None:
        """Flush Comet once and always close the inherited job resources."""
        experiment, self._comet = self._comet, None
        try:
            if experiment is not None:
                experiment.end()
        finally:
            super().finish()


class RestoreableJob(CheckpointedJob[C], Generic[C]):
    """Checkpointed job reconstructable from a stored node."""

    @classmethod
    def from_node(
        cls,
        node: Node,
        spec: ExecutionSpec,
        runtime_cfg: Any | None = None,
        resume: bool = False,
    ) -> tuple[Self, Any]:
        """Return the concrete job and configuration recorded for a node."""
        store = ObjectStore(spec.hardware)
        try:
            values = store.query().node(node).select(raw=True)
            state = values[0] if values else {}
        finally:
            store.close()

        blob = state.get("blob")
        if not isinstance(blob, Path):
            raise ValueError(
                f"No restorable object is associated with node={node.serialize()}"
            )

        # save() writes only JobSpec.model_fields to job.json, so restoring these
        # values preserves the caller's ExecutionSpec hardware and topology.
        if resume or runtime_cfg is None:
            job_json = blob / "job.json"
            if job_json.exists():
                with open(job_json, "r") as df:
                    job_spec_data = json.load(df)
                for key, value in job_spec_data.items():
                    setattr(spec, key, value)
                logger.debug("CHECKPOINT | restored job spec from checkpoint")

        cfg = OmegaConf.load(blob / "config.yaml")
        if runtime_cfg is not None:
            cfg = OmegaConf.merge(cfg, runtime_cfg)

        from theseus.registry import JOBS

        job_key = state.get("_x_job")
        job_cls = JOBS.get(job_key) if isinstance(job_key, str) else None
        if job_cls is None:
            raise ValueError(
                f"Unknown checkpoint job {job_key!r}. Import the job's module "
                "before calling from_node(), or supply a job class explicitly "
                "when building the execution."
            )
        elif not issubclass(job_cls, cls):
            logger.warning(
                "Configured job type {!r} is not a subclass of {}, defaulting to {}",
                job_key,
                cls.__name__,
                cls.__name__,
            )
            job_cls = cls

        with configuration(cfg):
            job: Self = job_cls(spec, base=node)
        job._resume_required = resume
        return job, cfg

    def save(
        self,
        state: PyTree[Any],
        metadata: dict[str, float | int | str],
    ) -> None:
        """Save a checkpoint labeled with its registered job type."""
        values = dict(metadata)
        job_name = type(self).__dict__.get("JOB_NAME")
        if isinstance(job_name, str):
            values["_x_job"] = job_name
        super().save(state, values)
