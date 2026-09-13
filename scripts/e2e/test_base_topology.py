#!/usr/bin/env python3
"""Exercise real loader batches and checkpoint restore across CPU host topologies.

Run once with --dataset padded and once with --dataset pmd.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path


@dataclass(frozen=True)
class Topology:
    processes: int
    devices_per_process: int
    tp: int


@dataclass(frozen=True)
class Scenario:
    name: str
    saved: Topology
    restored: Topology


SCENARIOS = (
    Scenario(
        "tp-single-to-ddp-multi",
        Topology(1, 2, 2),
        Topology(2, 1, 1),
    ),
    Scenario(
        "tp-multi-to-ddp-single",
        Topology(2, 2, 2),
        Topology(1, 2, 1),
    ),
    Scenario(
        "ddp-single-to-tp-multi",
        Topology(1, 2, 1),
        Topology(2, 2, 2),
    ),
    Scenario(
        "ddp-multi-to-tp-single",
        Topology(2, 1, 1),
        Topology(1, 2, 2),
    ),
)


def tree_signature(tree):
    import jax
    import jax.numpy as jnp

    signature = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        signature.append(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(leaf.shape),
                "dtype": str(leaf.dtype),
                "sum": float(jax.device_get(jnp.sum(leaf))),
                "square_sum": float(jax.device_get(jnp.sum(jnp.square(leaf)))),
            }
        )
    return signature


def assert_signature(actual, expected):
    assert len(actual) == len(expected)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        assert actual_leaf["path"] == expected_leaf["path"]
        assert actual_leaf["shape"] == expected_leaf["shape"]
        assert actual_leaf["dtype"] == expected_leaf["dtype"]
        assert math.isclose(
            actual_leaf["sum"],
            expected_leaf["sum"],
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
        assert math.isclose(
            actual_leaf["square_sum"],
            expected_leaf["square_sum"],
            rel_tol=1e-5,
            abs_tol=1e-5,
        )


def run_worker(args):
    from dataclasses import dataclass

    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from omegaconf import OmegaConf

    from theseus.base import ShardingPlan, ShardingPolicy, Axis, ExecutionSpec, Node
    from theseus.config import build, configuration
    from theseus.data.datasets import DatasetComponent
    from theseus.training.flywheel.strategy import Sampling
    from jax.experimental import multihost_utils
    from theseus.inference.base import InferenceJob
    from theseus.job import RestoreableJob
    from theseus.model.module import Module
    from theseus.registry import job
    from theseus.training.base import BaseTrainer, BaseTrainerConfig

    if args.processes > 1:
        jax.distributed.initialize(
            coordinator_address=args.coordinator,
            num_processes=args.processes,
            process_id=args.process_id,
            local_device_ids=range(args.devices),
            initialization_timeout=60,
            heartbeat_timeout_seconds=20,
            shutdown_timeout_seconds=60,
        )
        from jax._src import distributed, xla_bridge
        from jaxlib import xla_client

        collectives = xla_client._xla.make_gloo_tcp_collectives(
            distributed_client=distributed.global_state.client,
            interface=next(
                name for _, name in socket.if_nameindex() if name.startswith("lo")
            ),
        )
        registration = xla_bridge._backend_factories["cpu"]
        xla_bridge._backend_factories["cpu"] = replace(
            registration,
            factory=partial(xla_bridge.make_cpu_client, collectives=collectives),
        )

    class TinyModel(Module):
        @property
        def sharding(self):
            return ShardingPlan(tp=[("in", Axis.SHARD)])

        @classmethod
        def components(cls):
            return []

        @nn.compact
        def __call__(self, x, y=None, padding_mask=None, deterministic=False):
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
    class TinyTrainerConfig(BaseTrainerConfig):
        pass

    class LocalData(DatasetComponent):
        DATASET_KEY = "topology"

    def gather_batch(trainer):
        local_batch = trainer.batch()
        assert trainer.batch() is local_batch
        gathered = jax.tree.map(
            lambda value: np.asarray(
                multihost_utils.process_allgather(value, tiled=True)
            ),
            local_batch,
        )
        if jax.process_count() > 1:
            rows = local_batch["x"].shape[0]
            assert not np.array_equal(
                gathered["x"][:rows], gathered["x"][rows : 2 * rows]
            )
        global_batch = trainer._to_global(trainer._reshape_batch(local_batch))
        placed = jax.tree.map(
            lambda value: np.asarray(
                multihost_utils.process_allgather(value, tiled=True)
            ).reshape(-1, 8),
            global_batch,
        )
        # Accumulation can reorder rows; placement must preserve every sample.
        source_order = np.argsort(gathered["x"][:, 0], kind="stable")
        placed_order = np.argsort(placed["x"][:, 0], kind="stable")
        for key in gathered:
            np.testing.assert_array_equal(
                placed[key][placed_order], gathered[key][source_order]
            )
        return gathered, global_batch

    @job("tests/base-topology")
    class TinyTrainer(BaseTrainer[TinyTrainerConfig, TinyModel]):
        MODEL = TinyModel
        CONFIG = TinyTrainerConfig
        DATASET = [Sampling(LocalData, 1.0, args.dataset)]
        EVALUATION = []

        def evaluator(self):
            return None

        def run(self):
            kernel = self.state.params["Dense_0"]["kernel"].value
            assert kernel.sharding.spec == P(Axis.SHARD, None)
            assert len(kernel.sharding.device_set) == jax.device_count()

            if args.action == "save":
                saved_batch, batch = gather_batch(self)
                self.state, loss, _, _ = self._make_train_step()(
                    self.state,
                    batch,
                    self.key,
                    self.accumulate_steps,
                )
                assert math.isfinite(float(loss))
                assert int(jax.device_get(self.state.step)) == 1
                expected = tree_signature(self.state)
                expected_params = tree_signature(self.state.params)
                saved_node = self.node
                self.checkpoint()
                if self.main_process():
                    np.savez(Path(args.root) / "saved-batch.npz", **saved_batch)
                    (Path(args.root) / "expected.json").write_text(json.dumps(expected))
                    (Path(args.root) / "expected-params.json").write_text(
                        json.dumps(expected_params)
                    )
                    (Path(args.root) / "node.txt").write_text(saved_node.serialize())
                self.tick()
                following, _ = gather_batch(self)
                if self.main_process():
                    np.savez(Path(args.root) / "next-batch.npz", **following)
                return

            template = self.template
            assert not any(
                isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(template)
            )
            for restored_leaf, template_leaf in zip(
                jax.tree.leaves(self.state),
                jax.tree.leaves(template),
                strict=True,
            ):
                assert restored_leaf.sharding.is_equivalent_to(
                    template_leaf.sharding,
                    restored_leaf.ndim,
                )
                assert {
                    shard.device: shard.index
                    for shard in restored_leaf.addressable_shards
                } == template_leaf.sharding.addressable_devices_indices_map(
                    template_leaf.shape
                )

            expected = json.loads((Path(args.root) / "expected.json").read_text())
            assert_signature(tree_signature(self.state), expected)
            assert int(jax.device_get(self.state.step)) == 1
            assert self.node.seq == 0
            saved_batch, _ = gather_batch(self)
            with np.load(Path(args.root) / "saved-batch.npz") as expected_batch:
                for key, value in saved_batch.items():
                    np.testing.assert_array_equal(value, expected_batch[key])
            self.tick()
            assert self.node.seq == 1
            following, _ = gather_batch(self)
            with np.load(Path(args.root) / "next-batch.npz") as expected_batch:
                for key, value in following.items():
                    np.testing.assert_array_equal(value, expected_batch[key])

    @job("tests/base-topology-inference")
    class TinyInference(InferenceJob[TinyTrainerConfig, TinyModel]):
        MODEL = TinyModel

        @classmethod
        def config(cls):
            return [TinyTrainerConfig, TinyModel]

        def run(self):
            kernel = self.state.params["Dense_0"]["kernel"].value
            assert kernel.sharding.spec == P(Axis.SHARD, None)
            assert len(kernel.sharding.device_set) == jax.device_count()
            expected = json.loads(
                (Path(args.root) / "expected-params.json").read_text()
            )
            assert_signature(tree_signature(self.state.params), expected)

    try:
        assert jax.process_count() == args.processes
        assert jax.local_device_count() == args.devices
        cfg = build(*TinyTrainer.config())
        OmegaConf.set_struct(cfg, False)
        cfg.architecture.block_size = 8
        cfg.architecture.dtype.param = "float32"
        cfg.architecture.dtype.activation = "float32"
        cfg.training.batch_size = 4
        cfg.training.per_device_batch_size = 1
        cfg.training.tokens = 32
        cfg.training.validation = False
        cfg.training.evaluate = False
        cfg.logging.remote = False
        OmegaConf.set_struct(cfg, True)

        root = Path(args.root)
        spec = ExecutionSpec.local(
            root_dir=str(root),
            name="base-topology",
            shard=ShardingPolicy(tp=args.tp),
        ).model_copy(update={"distributed": args.processes > 1})
        base = None
        if args.action == "restore":
            base = Node.deserialize((root / "node.txt").read_text())

        if base is None:
            with configuration(cfg):
                TinyTrainer(spec)()
        else:
            restored, restored_cfg = RestoreableJob.from_node(base, spec, resume=True)
            with configuration(restored_cfg):
                restored()
                inference_spec = spec.model_copy(
                    update={"name": "base-topology-inference"}
                )
                TinyInference(inference_spec, base=base)()
    finally:
        if args.processes > 1:
            jax.distributed.shutdown()


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def run_phase(root: Path, action: str, topology: Topology, dataset: str) -> None:
    coordinator = f"127.0.0.1:{available_port()}"
    env = os.environ.copy()
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "PYTHONUNBUFFERED": "1",
            "TF_CPP_MIN_LOG_LEVEL": "1",
            "XLA_FLAGS": (
                f"--xla_force_host_platform_device_count={topology.devices_per_process}"
            ),
        }
    )
    for proxy in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ):
        env.pop(proxy, None)

    processes = []
    for process_id in range(topology.processes):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--action",
            action,
            "--root",
            str(root),
            "--coordinator",
            coordinator,
            "--processes",
            str(topology.processes),
            "--process-id",
            str(process_id),
            "--devices",
            str(topology.devices_per_process),
            "--tp",
            str(topology.tp),
            "--dataset",
            dataset,
        ]
        processes.append(
            subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    outputs = []
    try:
        for process in processes:
            outputs.append(process.communicate(timeout=300)[0])
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
        raise

    if any(process.returncode for process in processes):
        raise RuntimeError("".join(outputs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=("padded", "pmd"), default="padded")
    parser.add_argument("--action", choices=("save", "restore"))
    parser.add_argument("--root")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--tp", type=int)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    with tempfile.TemporaryDirectory(prefix="theseus-base-topology-") as temp:
        for scenario in SCENARIOS:
            root = Path(temp) / scenario.name
            root.mkdir()
            import numpy as np

            data_dir = root / "data" / "topology"
            data_dir.mkdir(parents=True)
            tokens = np.arange(2048 * 9, dtype=np.uint32).reshape(2048, 9) + 1
            tokens[::7] = 0
            tokens.tofile(data_dir / "train.bin")
            np.ones_like(tokens, dtype=np.bool_).tofile(data_dir / "train.bin.mask")
            (data_dir / "shape.json").write_text(
                json.dumps({"train": list(tokens.shape)})
            )
            print(
                f"{scenario.name}: save {scenario.saved} -> restore {scenario.restored}"
            )
            run_phase(root, "save", scenario.saved, args.dataset)
            run_phase(root, "restore", scenario.restored, args.dataset)
            print(f"{scenario.name}: PASS")


if __name__ == "__main__":
    main()
