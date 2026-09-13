#!/usr/bin/env python3
"""Launch live checkpoint, topology, and redispatch transition probes."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from theseus.base import (
    ShardingPlan,
    Axis,
    ExecutionSpec,
    Node,
    PyTree,
    SUPPORTED_CHIPS,
)
from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult
from theseus.config import field
from theseus.execute.combobulator import Combobulation, Combobulator
from theseus.execute.config import ClusterConfig, DispatchConfig
from theseus.execute.dispatch import DispatchSpec
from theseus.execute.provider import SSHConfig, SlurmConfig
from theseus.model.module import Module
from theseus.store import ValueRow
from theseus.training.base import BaseTrainer, BaseTrainerConfig


class TransitionModel(Module):
    """Tiny sharded model that keeps live probes cheap."""

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(tp=[("in", Axis.SHARD)])

    @classmethod
    def components(cls) -> list[type[Any]]:
        return []

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        y: jax.Array | None = None,
        padding_mask: jax.Array | None = None,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        del deterministic
        kernel_init = nn.with_logical_partitioning(
            nn.initializers.lecun_normal(),
            ("in", "out"),
        )
        logits = nn.Dense(
            8,
            kernel_init=kernel_init,
            param_dtype=self._param_dtype,
        )(x)
        if y is None:
            return logits, jnp.zeros((), dtype=logits.dtype)
        errors = jnp.square(logits - y)
        mask = (
            jnp.ones_like(errors)
            if padding_mask is None
            else padding_mask.astype(errors.dtype)
        )
        return logits, (errors * mask).sum() / mask.sum()


@dataclass
class TransitionConfig(BaseTrainerConfig):
    batch_size: int = field("training/batch_size", default=4)
    per_device_batch_size: int = field("training/per_device_batch_size", default=1)
    total_tokens: int = field("training/tokens", default=32)
    validate: bool = field("training/validation", default=False)
    evaluate: bool = field("training/evaluate", default=False)
    block_size: int = field("architecture/block_size", default=8)
    mode: str = field("e2e/mode", default="handoff")
    expected_processes: int = field("e2e/expected_processes", default=1)
    expected_tp: int = field("e2e/expected_tp", default=1)
    pause_seconds: int = field("e2e/pause_seconds", default=600)


class TransitionTrainer(BaseTrainer[TransitionConfig, TransitionModel]):
    """Train once, checkpoint, and prove restoration through the real runner."""

    MODEL = TransitionModel
    CONFIG = TransitionConfig
    DATASET = []
    EVALUATION = []

    def __init__(self, spec: ExecutionSpec, base: Node | None = None) -> None:
        self.restored_metadata: ValueRow | None = None
        super().__init__(spec, base=base)

    def _init_data(self, spec: ExecutionSpec) -> None:
        del spec

    def evaluator(self) -> None:
        return None

    def apply(self, state: PyTree[Any], metadata: ValueRow) -> None:
        super().apply(state, metadata)
        self.restored_metadata = metadata

    def run(self) -> None:
        """Exercise a handoff or pause at a durable checkpoint."""
        self._assert_topology()
        step_index = int((self.spec.tag or "0:").split(":", 1)[0])
        restored_step = (
            self.restored_metadata.get("e2e/step_index")
            if self.restored_metadata is not None
            else None
        )

        if isinstance(restored_step, (int, float)) and int(restored_step) == step_index:
            self._assert_restored(step_index)
            return

        if step_index and restored_step != step_index - 1:
            raise AssertionError(
                f"step {step_index} expected checkpoint {step_index - 1}, "
                f"found {restored_step!r}"
            )

        self._train_once()
        self.checkpoint()
        checksum = self._checksum()
        self.log(
            {
                "e2e/checksum": checksum,
                "e2e/processes": jax.process_count(),
                "e2e/shards": self.args.expected_tp,
                "e2e/step_index": step_index,
            }
        )
        self.chkpt_manager.checkpointer.wait_until_finished()
        logger_message = (
            f"E2E | checkpoint ready step_index={step_index} "
            f"checksum={checksum} processes={jax.process_count()} "
            f"shards={self.args.expected_tp}"
        )
        print(logger_message, flush=True)

        if self.args.mode == "handoff":
            raise RuntimeError("intentional checkpoint handoff")
        if self.args.mode in {"preempt", "chain"}:
            if self.args.mode == "chain" and step_index == 0:
                return
            self.store.close()
            print(
                "E2E | checkpoint metadata flushed; waiting for cancellation",
                flush=True,
            )
            time.sleep(self.args.pause_seconds)
            raise RuntimeError("checkpoint probe was not cancelled")
        raise ValueError(f"unknown e2e mode {self.args.mode!r}")

    def _assert_topology(self) -> None:
        expected_mesh = (
            jax.device_count() // self.args.expected_tp,
            self.args.expected_tp,
        )
        if jax.process_count() != self.args.expected_processes:
            raise AssertionError(
                f"expected {self.args.expected_processes} processes, "
                f"found {jax.process_count()}"
            )
        if tuple(self.mesh.devices.shape) != expected_mesh:
            raise AssertionError(
                f"expected mesh {expected_mesh}, found {self.mesh.devices.shape}"
            )
        kernel = self.state.params["Dense_0"]["kernel"].value
        if kernel.sharding.spec != P(Axis.SHARD, None):  # type: ignore[no-untyped-call]
            raise AssertionError(f"unexpected kernel sharding {kernel.sharding.spec}")

    def _assert_restored(self, step_index: int) -> None:
        assert self.restored_metadata is not None
        stored_checksum = self.restored_metadata["e2e/checksum"]
        if not isinstance(stored_checksum, (int, float)):
            raise AssertionError(f"invalid stored checksum {stored_checksum!r}")
        expected = float(stored_checksum)
        actual = self._checksum()
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5):
            raise AssertionError(f"checkpoint checksum changed: {expected} -> {actual}")
        if self.base is None or self.node != self.base.next():
            raise AssertionError("runner did not continue from the discovered node")
        if int(jax.device_get(self.state.step)) != step_index + 1:
            raise AssertionError("restored optimizer step is incorrect")
        self.log(
            {
                "e2e/restored": 1,
                "e2e/restored_from_seq": self.base.seq,
                "e2e/continued_as_seq": self.node.seq,
            }
        )
        print(
            f"E2E | restore PASS step_index={step_index} checksum={actual} "
            f"base_seq={self.base.seq} node_seq={self.node.seq} "
            f"processes={jax.process_count()} shards={self.args.expected_tp}",
            flush=True,
        )

    def _train_once(self) -> None:
        local_batch = (
            self.per_device_batch_size * self.local_replicas * self.accumulate_steps
        )
        batch = self._to_global(
            self._reshape_batch(
                {
                    "x": np.ones((local_batch, 8), dtype=np.float32),
                    "y": np.zeros((local_batch, 8), dtype=np.float32),
                    "padding_mask": np.ones((local_batch, 8), dtype=np.bool_),
                }
            )
        )
        self.state, loss, _, _ = self._make_train_step()(
            self.state,
            batch,
            self.key,
            self.accumulate_steps,
        )
        if not math.isfinite(float(loss)):
            raise AssertionError(f"training produced invalid loss {loss}")

    def _checksum(self) -> float:
        return sum(
            float(jax.device_get(jnp.sum(leaf))) for leaf in jax.tree.leaves(self.state)
        )


def new_cluster(name: str, configured: ClusterConfig) -> Cluster:
    """Convert dispatch inventory into the portable cluster wire model."""
    return Cluster(
        name=name,
        root=configured.root,
        work=configured.work,
        log=configured.log,
        data=configured.data,
        checkpoints=configured.checkpoints,
        results=configured.results,
        status=configured.status,
        mount=configured.mount,
        cache_size=configured.cache_size,
        cache_dir=configured.cache_dir,
        all_squash=configured.all_squash,
    )


def new_hardware(
    config: DispatchConfig,
    provider: str,
    chip_name: str,
    layout: tuple[int, ...],
    visible_devices: str | None,
) -> HardwareResult:
    """Build an exact, homogeneous layout for a bounded live probe."""
    host = config.hosts[provider]
    if not isinstance(host, (SSHConfig, SlurmConfig)):
        raise ValueError(f"unsupported provider {provider!r}")
    if isinstance(host, SSHConfig) and len(layout) != 1:
        raise ValueError("SSH transition probes require one host")
    chip = SUPPORTED_CHIPS[chip_name]
    cluster = new_cluster(host.cluster, config.clusters[host.cluster])
    environment = dict(host.env)
    if isinstance(host, SlurmConfig) and len(layout) > 1:
        environment.update(
            {
                "NCCL_IB_DISABLE": "1",
                "NCCL_SOCKET_IFNAME": "^lo,docker,ib",
            }
        )
    if visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    machines = [
        ClusterMachine(
            name=provider,
            cluster=cluster,
            resources={chip: count},
            # The live probe needs the accelerator runtime, not the cluster's
            # kitchen-sink development group.
            uv_groups=[group for group in host.uv_groups if group != "all"],
            env=environment,
        )
        for count in layout
    ]
    return HardwareResult(chip=chip, hosts=machines, total_chips=sum(layout))


def new_execution(mode: str, chain: bool) -> Combobulation:
    """Build the single-step or chained live probe."""
    execution = Combobulator().run(TransitionTrainer)
    if chain:
        execution = execution.branch(TransitionTrainer)
    execution.config.e2e.mode = mode
    return execution


def launch(arguments: argparse.Namespace, config: DispatchConfig) -> None:
    """Create, persist, and remotely launch one probe dispatch."""
    layout = tuple(int(count) for count in arguments.layout.split(","))
    execution = new_execution(arguments.mode, arguments.chain)
    execution.config.e2e.expected_processes = len(layout)
    execution.config.e2e.expected_tp = arguments.tp
    execution.config.e2e.pause_seconds = arguments.pause_seconds
    execution.config.architecture.dtype.param = "float32"
    execution.config.architecture.dtype.activation = "float32"
    dispatch = DispatchSpec(
        name=arguments.name,
        project="execute-e2e",
        group=arguments.group,
        nonce=arguments.nonce,
        hardware=new_hardware(
            config,
            arguments.provider,
            arguments.chip,
            layout,
            arguments.visible_devices,
        ),
        job=execution.shard(tp=arguments.tp),
    )
    arguments.output.write_text(dispatch.model_dump_json(indent=2))
    result = dispatch.launch(config)
    print(json.dumps(asdict(result), indent=2))
    if not result.ok:
        raise RuntimeError(result.remote.stderr)


def relaunch(arguments: argparse.Namespace, config: DispatchConfig) -> None:
    """Relaunch the exact serialized identity for an idempotency probe."""
    dispatch = DispatchSpec.model_validate_json(arguments.dispatch.read_text())
    result = dispatch.launch(config)
    print(json.dumps(asdict(result), indent=2))
    if not result.ok:
        raise RuntimeError(result.remote.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--provider", required=True)
    launch_parser.add_argument("--chip", required=True)
    launch_parser.add_argument("--layout", required=True, help="GPUs per host, CSV")
    launch_parser.add_argument("--tp", type=int, required=True)
    launch_parser.add_argument(
        "--mode", choices=("handoff", "preempt", "chain"), required=True
    )
    launch_parser.add_argument("--chain", action="store_true")
    launch_parser.add_argument("--pause-seconds", type=int, default=600)
    launch_parser.add_argument("--visible-devices")
    launch_parser.add_argument("--name", required=True)
    launch_parser.add_argument("--group", required=True)
    launch_parser.add_argument("--nonce", required=True)
    launch_parser.add_argument("--output", type=Path, required=True)

    replay_parser = commands.add_parser("relaunch")
    replay_parser.add_argument("dispatch", type=Path)

    arguments = parser.parse_args()
    config = DispatchConfig.load()
    if arguments.command == "launch":
        launch(arguments, config)
    else:
        relaunch(arguments, config)


if __name__ == "__main__":
    main()
