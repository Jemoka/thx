import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from pydantic import BaseModel, ValidationError

from theseus.base import ShardingPolicy, Node, SUPPORTED_CHIPS
from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult
from theseus.config import field
from theseus.execute.combobulator import Combobulation, Combobulator, Step
from theseus.execute.dispatch import DispatchSpec
from theseus.job import BasicJob
from theseus.registry import JOBS, is_builtin_job, job
from theseus.store import ObjectStore, QueryBuilder


@dataclass
class FirstConfig:
    first: int = field("first/value", default=1)


@dataclass
class SecondConfig:
    second: str = field("second/value", default="two")


@dataclass
class MissingConfig:
    required: int = field("required/value")


@job("tests/execute/first")
class FirstJob(BasicJob[FirstConfig]):
    @classmethod
    def config(cls) -> type[FirstConfig]:
        return FirstConfig


@job("tests/execute/second")
class SecondJob(BasicJob[SecondConfig]):
    @classmethod
    def config(cls) -> list[type[SecondConfig]]:
        return [SecondConfig]


@job("tests/execute/missing")
class MissingJob(BasicJob[MissingConfig]):
    @classmethod
    def config(cls) -> type[MissingConfig]:
        return MissingConfig


def test_nonce_survives_serialization_builders_and_dispatch():
    chain = Combobulator().run(FirstJob)
    nonce = chain.nonce
    changed = chain.gpu(2).branch(SecondJob)
    assert changed.nonce == nonce
    restored = Combobulation.deserialize(changed.serialize())
    assert restored.nonce == nonce
    dispatch = DispatchSpec(name="test", project="test", group="test", job=restored)
    assert dispatch.nonce == nonce
    assert DispatchSpec.model_validate_json(dispatch.model_dump_json()).nonce == nonce
    assert Combobulator().run(FirstJob).nonce != nonce


class UnregisteredJob(BasicJob[FirstConfig]):
    @classmethod
    def config(cls) -> type[FirstConfig]:
        return FirstConfig


def select_first(query: QueryBuilder) -> QueryBuilder:
    return query.spec(project="first")


def select_final(query: QueryBuilder) -> QueryBuilder:
    return query.spec(project="final")


def test_execution_records_branch_and_resume_semantics(tmp_path: Path) -> None:
    chain = (
        Combobulator()
        .run(FirstJob, base=select_first)
        .branch(SecondJob)
        .resume(FirstJob, base=select_final)
    )

    first = Node(name="first.default.run", nonce="aaaaaa")
    final = Node(name="final.default.run", nonce="bbbbbb")
    store = ObjectStore.local(str(tmp_path))
    store.value(first, {})
    store.value(final, {})
    store.close()

    assert chain.jobs == (FirstJob, SecondJob, FirstJob)
    assert chain.steps[0].base is not None
    assert store.query(chain.steps[0].base).all() == [first]
    assert chain.steps[0].resume is False
    assert chain.steps[1].base == "previous"
    assert chain.steps[1].resume is False
    assert chain.steps[2].base is not None
    assert store.query(chain.steps[2].base).all() == [final]
    assert chain.steps[2].resume is True

    restored = Combobulation.deserialize(chain.serialize())
    assert store.query(restored.steps[0].base).all() == [first]
    assert store.query(restored.steps[2].base).all() == [final]


def test_execution_appends_fresh_jobs_without_mutating_the_original() -> None:
    original = Combobulator().run(FirstJob).resume(FirstJob).gpu(2).memory("64Gi")
    original.config.first.value = 9
    extended = original.run(SecondJob)

    assert original.jobs == (FirstJob, FirstJob)
    assert extended.jobs == (FirstJob, FirstJob, SecondJob)
    assert extended.steps[-1].base is None
    assert extended.steps[-1].resume is False
    assert extended.gpus == 2
    assert extended.minimum_memory == "64Gi"
    assert extended.config.first.value == 9
    assert extended.config.second.value == "two"
    extended.config.first.value = 10
    assert original.config.first.value == 9

    restored = Combobulation.deserialize(extended.serialize())
    assert restored.steps[-1].base is None
    assert restored.steps[-1].resume is False
    assert restored.config.first.value == 10


def test_execution_serializes_and_deserializes_complete_setup(tmp_path: Path) -> None:
    chain = (
        Combobulator()
        .run(FirstJob)
        .branch(SecondJob)
        .memory("64Gi")
        .cpu(16)
        .gpu(4)
        .chip("a6000")
        .chip(SUPPORTED_CHIPS["h100"])
        .shard(fsdp=True, activation_checkpointing=True)
    )
    assert OmegaConf.is_struct(chain.config)
    assert chain.config is chain.config
    chain.config.first.value = 7
    with pytest.raises(ConfigAttributeError):
        chain.config.missing = True

    serialized = chain.serialize()

    assert serialized.config.first.value == 7
    assert serialized.config.second.value == "two"
    assert serialized.execution.sharding.activation_checkpointing is True
    assert all(step.job.type == "cloudpickle" for step in serialized.execution.steps)
    assert serialized.execution.steps[0].base is None
    assert serialized.execution.steps[1].base == "previous"
    assert serialized.execution.minimum_memory == "64Gi"
    assert serialized.execution.cpus == 16
    assert serialized.execution.gpus == 4
    assert tuple(serialized.execution.chips) == ("a6000", "h100")
    assert serialized.config.first.value == 7
    path = tmp_path / "setup.yaml"
    OmegaConf.save(serialized, path)

    restored = Combobulation.deserialize(OmegaConf.load(path))

    assert restored.jobs == chain.jobs
    assert restored.serialize() == OmegaConf.load(path)


def test_execution_model_dump_round_trips_registered_jobs() -> None:
    execution = Combobulator().run(FirstJob).branch(SecondJob)
    execution.config.first.value = 17
    chip = SUPPORTED_CHIPS["a100"]
    machine = ClusterMachine(
        name="test",
        cluster=Cluster(
            name="test",
            root="/test",
            work="/test/work",
            mount="redis://juicefs.example/1",
            cache_size="1024",
        ),
        resources={chip: 2},
        uv_groups=["cuda"],
        env={"UV_CACHE_DIR": "/cache/uv"},
    )
    dispatch = DispatchSpec(
        name="test",
        project="tests",
        group="tests",
        hardware=HardwareResult(chip=chip, hosts=[machine], total_chips=2),
        job=execution,
    )

    dumped_json = dispatch.model_dump_json()
    dumped = json.loads(dumped_json)

    assert all(step["job"]["type"] == "cloudpickle" for step in dumped["job"]["steps"])
    assert dumped["hardware"]["hosts"][0]["resources"] == {"a100-sxm4-80gb": 2}
    assert isinstance(dumped["job"]["config"], str)
    restored = DispatchSpec.model_validate_json(dumped_json)
    assert restored.job.jobs == (FirstJob, SecondJob)
    assert restored.job.config.first.value == 17
    assert restored.hardware.hosts[0].resources == {chip: 2}
    assert restored.hardware.hosts[0].cluster.mount == "redis://juicefs.example/1"
    assert restored.hardware.hosts[0].cluster.cache_size == "1024"
    assert restored.hardware.hosts[0].uv_groups == ["cuda"]
    assert restored.hardware.hosts[0].env == {"UV_CACHE_DIR": "/cache/uv"}


def test_execution_model_dump_cloudpickles_unregistered_jobs() -> None:
    execution = Combobulator().run(UnregisteredJob)

    serialized = execution.model_dump_json()
    dumped = json.loads(serialized)
    assert dumped["steps"][0]["job"]["type"] == "cloudpickle"

    restored = Combobulation.model_validate_json(serialized)
    assert restored.jobs == (UnregisteredJob,)
    assert restored.jobs[0].config() is FirstConfig
    assert restored.steps[0].job_name == (
        f"{UnregisteredJob.__module__}.{UnregisteredJob.__qualname__}"
    )
    assert restored.model_dump_json() == serialized

    replaced = restored.steps[0].model_copy(update={"job": FirstJob})
    assert replaced.model_dump()["job"]["type"] == "cloudpickle"
    assert Step.model_validate(replaced.model_dump()).job is FirstJob


def test_execution_model_dump_cloudpickles_registered_main_module_jobs() -> None:
    job_class = job("tests/execute/main-module")(
        type("MainModuleJob", (FirstJob,), {"__module__": "__main__"})
    )

    dumped = Combobulator().run(job_class).model_dump()

    assert dumped["steps"][0]["job"]["type"] == "cloudpickle"
    restored = Combobulation.model_validate(dumped)
    assert restored.steps[0].job_name == "tests/execute/main-module"


def test_execution_model_dump_cloudpickles_main_module_jobs_without_name() -> None:
    job_class = type(
        "NotebookJob",
        (UnregisteredJob,),
        {"__module__": "__main__"},
    )
    execution = Combobulator().run(job_class)

    restored = Combobulation.model_validate_json(execution.model_dump_json())

    assert restored.jobs[0].__name__ == "NotebookJob"
    assert restored.jobs[0].__module__ == "__main__"
    assert restored.jobs[0].config() is FirstConfig


def test_builtin_jobs_keep_name_serialization_and_subclasses_do_not():
    builtin = JOBS["gpt/train/pretrain"]
    assert builtin.__dict__["JOB_BUILTIN"] is True
    assert Step(job=builtin).model_dump()["job"] == {
        "type": "registered",
        "payload": "gpt/train/pretrain",
    }
    assert Step.model_validate({"job": "gpt/train/pretrain"}).job is builtin

    external = type("ExternalSubclass", (builtin,), {"__module__": __name__})
    assert not is_builtin_job(external)
    assert Step(job=external).model_dump()["job"]["type"] == "cloudpickle"
    assert Step(job=external).job_name.endswith(".ExternalSubclass")
    named = job("tests/execute/external-subclass")(external)
    assert named.__dict__["JOB_BUILTIN"] is False
    assert Step(job=named).model_dump()["job"]["type"] == "cloudpickle"


@pytest.mark.parametrize("pre_registered", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_pickling_restores_module_registration(monkeypatch, pre_registered, fails):
    module = sys.modules[__name__]
    if pre_registered:
        cloudpickle.register_pickle_by_value(module)
    before = cloudpickle.list_registry_pickle_by_value()
    try:
        if fails:
            from unittest.mock import Mock

            monkeypatch.setattr(
                cloudpickle, "dumps", Mock(side_effect=RuntimeError("failure"))
            )
            with pytest.raises(ValueError, match="could not be cloudpickled"):
                Step(job=FirstJob).model_dump()
        else:
            assert Step(job=FirstJob).model_dump()["job"]["type"] == "cloudpickle"
        assert cloudpickle.list_registry_pickle_by_value() == before
    finally:
        if pre_registered:
            cloudpickle.unregister_pickle_by_value(module)


def test_execution_model_validate_rejects_jobs_missing_from_instance() -> None:
    with pytest.raises(
        ValidationError,
        match="this job is not located on this instance",
    ):
        Step.model_validate(
            {
                "job": {
                    "type": "registered",
                    "payload": "tests/execute/not-installed",
                }
            }
        )


def test_execution_model_validate_rejects_invalid_cloudpickle_payload() -> None:
    with pytest.raises(ValidationError, match="could not be loaded"):
        Step.model_validate({"job": {"type": "cloudpickle", "payload": "not base64"}})


def test_execution_model_validate_rejects_cloudpickled_non_job() -> None:
    payload = base64.b64encode(cloudpickle.dumps(dict)).decode("ascii")

    with pytest.raises(ValidationError, match="not a BasicJob subclass"):
        Step.model_validate({"job": {"type": "cloudpickle", "payload": payload}})


def test_combobulator_starts_independent_chains() -> None:
    combobulator = Combobulator()

    assert combobulator.run(FirstJob).jobs == (FirstJob,)
    assert combobulator.run(SecondJob).jobs == (SecondJob,)
    independent = combobulator.run(FirstJob).branch(SecondJob, base=lambda _: None)
    assert independent.steps[1].base is None


def test_execution_canonicalizes_chip_aliases() -> None:
    execution = Combobulator().run(FirstJob).chip("a100")

    assert execution.chips == ("a100-sxm4-80gb",)
    assert tuple(execution.serialize().execution.chips) == ("a100-sxm4-80gb",)


def test_execution_healthcheck_requires_a_job_and_complete_config() -> None:
    assert issubclass(Combobulation, BaseModel)
    assert issubclass(Step, BaseModel)
    assert not Combobulation(steps=()).healthcheck()
    assert Combobulator().run(FirstJob).healthcheck()

    execution = Combobulator().run(MissingJob)
    assert not execution.healthcheck()
    execution.config.required.value = 1
    assert execution.healthcheck()


def test_sharding_policy_round_trip_and_immutable_chain():
    original = Combobulator().run(FirstJob)
    chain = original.shard(tp=2, fsdp=True).branch(SecondJob)
    restored = Combobulation.deserialize(chain.serialize())
    assert original.sharding == ShardingPolicy()
    assert restored.sharding == ShardingPolicy(tp=2, fsdp=True)
    dispatch = DispatchSpec(name="sharded", project="test", group="test", job=restored)
    assert (
        DispatchSpec.model_validate_json(dispatch.model_dump_json()).job.sharding
        == restored.sharding
    )


@pytest.mark.parametrize(
    "options", [{"tp": 0}, {"tp": True}, {"fsdp": True, "zero": False}]
)
def test_sharding_policy_rejects_invalid_choices(options):
    with pytest.raises(ValidationError):
        Combobulator().run(FirstJob).shard(**options)
