"""Declarative execution chains."""

import base64
import sys
from uuid import uuid4
from typing import Any, Callable, Literal, Mapping, Self, TypeAlias, cast

import cloudpickle
from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from theseus.base import Chip, SUPPORTED_CHIPS, ShardingPolicy
from theseus.config import build
from theseus.job import BasicJob
from theseus.registry import JOBS, is_builtin_job
from theseus.store import QueryBuilder, SerializedQuery, _SerializedQueryBuilder

BaseQuery: TypeAlias = Callable[[QueryBuilder], QueryBuilder | None]
BaseOf: TypeAlias = SerializedQuery | Literal["previous"] | None


class Step(BaseModel):
    """One job and the lineage from which it should start."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    job: type[BasicJob]  # type: ignore[type-arg]
    base: BaseOf = None
    resume: bool = False
    _job_payload: tuple[type[BasicJob], dict[str, str]] | None = PrivateAttr(  # type: ignore[type-arg]
        default=None
    )

    @property
    def job_name(self) -> str:
        """Return the registered name or portable class identity for this job."""
        registered_name = self.job.__dict__.get("JOB_NAME")
        if isinstance(registered_name, str):
            return registered_name
        return f"{self.job.__module__}.{self.job.__qualname__}"

    @field_serializer("job")
    def _serialize_job(
        self,
        job: type[BasicJob],  # type: ignore[type-arg]
    ) -> dict[str, str]:
        if self._job_payload is not None and self._job_payload[0] is job:
            return dict(self._job_payload[1])
        job_name = job.__dict__.get("JOB_NAME")
        builtin = job.__dict__.get("JOB_BUILTIN")
        if builtin is None:
            builtin = is_builtin_job(job)
        if builtin and isinstance(job_name, str):
            return {"type": "registered", "payload": job_name}
        module = sys.modules.get(job.__module__)
        register_by_value = (
            not builtin
            and module is not None
            and module.__name__ not in cloudpickle.list_registry_pickle_by_value()
        )
        try:
            if register_by_value:
                cloudpickle.register_pickle_by_value(module)
            payload = cloudpickle.dumps(job)
        except Exception as error:
            raise ValueError(
                f"Job {job.__qualname__!r} could not be cloudpickled"
            ) from error
        finally:
            if register_by_value:
                cloudpickle.unregister_pickle_by_value(module)
        return {
            "type": "cloudpickle",
            "payload": base64.b64encode(payload).decode("ascii"),
        }

    @field_validator("job", mode="before")
    @classmethod
    def _deserialize_job(cls, job: Any) -> Any:
        # Configuration YAML names registered jobs directly; portable dispatches
        # use tagged registered or cloudpickled identities.
        if isinstance(job, str):
            registered = JOBS.get(job)
            if registered is None:
                raise ValueError(f"{job!r}: this job is not located on this instance")
            return registered
        if not isinstance(job, Mapping):
            return job

        kind = job.get("type")
        payload = job.get("payload")
        if not isinstance(kind, str) or not isinstance(payload, str):
            raise ValueError("Serialized jobs require string type and payload fields")
        if kind == "registered":
            registered = JOBS.get(payload)
            if registered is None:
                raise ValueError(
                    f"{payload!r}: this job is not located on this instance"
                )
            return registered
        if kind != "cloudpickle":
            raise ValueError(f"Unknown serialized job type {kind!r}")

        # A cloudpickled DispatchSpec is executable code and therefore shares
        # the same trust boundary as the repository bundle that accompanies it.
        try:
            restored = cloudpickle.loads(base64.b64decode(payload, validate=True))
        except Exception as error:
            raise ValueError("Cloudpickled job payload could not be loaded") from error
        if not isinstance(restored, type) or not issubclass(restored, BasicJob):
            raise ValueError("Cloudpickled job is not a BasicJob subclass")
        return restored

    @model_validator(mode="wrap")
    @classmethod
    def _preserve_job_payload(
        cls,
        value: Any,
        handler: ModelWrapValidatorHandler[Self],
    ) -> Self:
        """Keep validated code bytes stable across transport round trips."""
        step = handler(value)
        if not isinstance(value, Mapping):
            return step
        serialized = value.get("job")
        if not isinstance(serialized, Mapping):
            return step
        kind = serialized.get("type")
        payload = serialized.get("payload")
        if kind == "cloudpickle" and isinstance(payload, str):
            step._job_payload = (step.job, {"type": kind, "payload": payload})
        return step

    #### construction ####

    @classmethod
    def new(
        cls,
        job: type[BasicJob[Any]],
        base: BaseQuery | None = None,
        resume: bool = False,
        implicit_base: Literal["previous"] | None = None,
    ) -> Self:
        """Build an execution step and immediately serialize its base query."""
        if base is None:
            return cls(job=job, base=implicit_base, resume=resume)
        builder = _SerializedQueryBuilder()
        selected = base(builder)
        if selected is None:
            return cls(job=job, resume=resume)
        if selected is not builder:
            raise TypeError("base queries must return the supplied builder or None")
        return cls(job=job, base=builder.serialize(), resume=resume)


class Combobulation(BaseModel):
    """Ordered jobs and resource constraints for one execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    steps: tuple[Step, ...]
    nonce: str = Field(default_factory=lambda: uuid4().hex[:6])
    minimum_memory: int | str | None = None
    cpus: int | None = None
    gpus: int | None = None
    chips: tuple[str, ...] = ()
    preferred_clusters: tuple[str, ...] = ()
    forbidden_clusters: tuple[str, ...] = ()
    sharding: ShardingPolicy = Field(default_factory=ShardingPolicy)
    _configuration: DictConfig = PrivateAttr()

    @model_serializer(mode="wrap")
    def _serialize_configuration(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        serialized = cast(dict[str, Any], handler(self))
        serialized["config"] = OmegaConf.to_yaml(self.config, resolve=False)
        return serialized

    @model_validator(mode="wrap")
    @classmethod
    def _deserialize_configuration(
        cls,
        value: Any,
        handler: ModelWrapValidatorHandler[Self],
    ) -> Self:
        serialized_config = None
        if isinstance(value, dict) and "config" in value:
            value = dict(value)
            serialized_config = value.pop("config")
        execution = handler(value)
        if serialized_config is None:
            return execution
        if not isinstance(serialized_config, str):
            raise ValueError("Combobulation config must be an OmegaConf YAML string")
        configuration = OmegaConf.merge(
            execution.config,
            OmegaConf.create(serialized_config),
        )
        OmegaConf.set_struct(configuration, True)
        execution._configuration = cast(DictConfig, configuration)
        return execution

    def model_post_init(self, __context: Any) -> None:
        schemas: list[type[Any]] = []
        for job in self.jobs:
            declared = job.config()
            schemas.extend(
                declared if isinstance(declared, (list, tuple)) else [declared]
            )
        configuration = cast(DictConfig, build(*schemas))
        OmegaConf.set_struct(configuration, True)
        self._configuration = configuration

    @property
    def jobs(self) -> tuple[type[BasicJob[Any]], ...]:
        """Return the job classes in execution order."""
        return tuple(step.job for step in self.steps)

    @property
    def config(self) -> DictConfig:
        """Return the mutable, structured configuration shared by all steps."""
        return self._configuration

    def healthcheck(self) -> bool:
        """Return whether the execution has a job and no missing config values."""
        return bool(self.steps) and not OmegaConf.missing_keys(self.config)

    def run(self, job: type[BasicJob[Any]]) -> Self:
        """Append a fresh job without restoring a checkpoint."""
        return self._replace(steps=(*self.steps, Step.new(job)))

    def branch(
        self,
        job: type[BasicJob[Any]],
        base: BaseQuery | None = None,
    ) -> Self:
        """Append a job that starts a new lineage from a checkpoint.

        Args:
            job: Job class to run after the current final step.
            base: Optional query selecting an explicit base. Without one, the
                execution uses the preceding step's last checkpoint.

        Returns:
            A new execution ending with the branching job.
        """
        return self._replace(
            steps=(
                *self.steps,
                Step.new(job, base, implicit_base="previous"),
            ),
        )

    def resume(
        self,
        job: type[BasicJob[Any]],
        base: BaseQuery | None = None,
    ) -> Self:
        """Append a job that continues an existing checkpoint lineage.

        Args:
            job: Job class to run after the current final step.
            base: Optional query selecting an explicit base. Without one, the
                execution uses the preceding step's last checkpoint.

        Returns:
            A new execution ending with the resumed job.
        """
        return self._replace(
            steps=(
                *self.steps,
                Step.new(
                    job,
                    base,
                    resume=True,
                    implicit_base="previous",
                ),
            ),
        )

    def memory(self, minimum: int | str) -> Self:
        """Request memory per node: integer MiB or a size such as "64Gi"."""
        return self._replace(minimum_memory=minimum)

    def shard(
        self,
        tp: int = 1,
        fsdp: bool = False,
        zero: bool = True,
        activation_checkpointing: bool = False,
    ) -> Self:
        """Set parallelism for every job in this execution."""
        return self._replace(
            sharding=ShardingPolicy(
                tp=tp,
                fsdp=fsdp,
                zero=zero,
                activation_checkpointing=activation_checkpointing,
            )
        )

    def cpu(self, minimum: int) -> Self:
        """Return a new execution with a minimum CPU request."""
        return self._replace(cpus=minimum)

    def gpu(self, minimum: int) -> Self:
        """Return a new execution with a minimum GPU request."""
        return self._replace(gpus=minimum)

    def chip(self, chip: str | Chip) -> Self:
        """Return a new execution accepting one additional chip type."""
        resolved = chip if isinstance(chip, Chip) else SUPPORTED_CHIPS[chip]
        return self._replace(chips=(*self.chips, resolved.name))

    #### serialization ####

    def serialize(self) -> DictConfig:
        """Serialize the complete execution setup.

        Returns:
            An OmegaConf containing the execution graph, resource constraints,
            and hydrated component configuration.
        """
        serialized = self.model_dump()
        configuration = serialized.pop("config")
        return OmegaConf.create(
            {
                "execution": serialized,
                "config": OmegaConf.create(configuration),
            }
        )

    @classmethod
    def deserialize(cls, serialized: DictConfig) -> Self:
        """Restore an execution from :meth:`serialize` output.

        Args:
            serialized: Complete execution setup.

        Returns:
            The restored execution with its embedded component configuration.

        Raises:
            ValueError: If a serialized job cannot be resolved or restored.
        """
        payload = OmegaConf.to_container(serialized.execution, resolve=True)
        if not isinstance(payload, dict):
            raise ValueError("Serialized execution must be a mapping")
        payload["config"] = OmegaConf.to_yaml(
            serialized.get("config", {}), resolve=False
        )
        return cls.model_validate(payload)

    #### private methods ####

    def _replace(self, **changes: Any) -> Self:
        values = {name: getattr(self, name) for name in type(self).model_fields}
        updated = type(self).model_validate({**values, **changes})
        configuration = OmegaConf.merge(updated.config, self.config)
        OmegaConf.set_struct(configuration, True)
        updated._configuration = cast(DictConfig, configuration)
        return updated


class Combobulator:
    """Assemble declarative execution chains without running them."""

    def run(
        self,
        job: type[BasicJob[Any]],
        base: BaseQuery | None = None,
    ) -> Combobulation:
        """Start an execution with one job.

        Args:
            job: First job class in the execution.
            base: Optional query selecting the node from which to branch. An
                empty query result causes execution to stop without running it.

        Returns:
            A new execution containing the initial job.
        """
        return Combobulation(steps=(Step.new(job, base),))
