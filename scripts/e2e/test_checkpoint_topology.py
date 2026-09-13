#!/usr/bin/env python3
"""Exercise checkpoint restores across local CPU device and host topologies."""

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
    mesh: tuple[int, int]


@dataclass(frozen=True)
class Scenario:
    name: str
    saved: Topology
    restored: Topology


SCENARIOS = (
    Scenario("single-to-single", Topology(1, 1, (1, 1)), Topology(1, 1, (1, 1))),
    Scenario("single-to-multiple", Topology(1, 1, (1, 1)), Topology(1, 4, (1, 4))),
    Scenario("multiple-to-single", Topology(1, 4, (1, 4)), Topology(1, 1, (1, 1))),
    Scenario(
        "multiple-to-different-multiple",
        Topology(1, 4, (1, 4)),
        Topology(1, 4, (2, 2)),
    ),
    Scenario("singlehost-to-multihost", Topology(1, 2, (1, 2)), Topology(2, 1, (1, 2))),
    Scenario("multihost-to-singlehost", Topology(2, 1, (1, 2)), Topology(1, 2, (1, 2))),
)


def apply_model(variables, inputs):
    return inputs @ variables["params"]["dense"]["kernel"]


def make_template(mesh):
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from jax.sharding import NamedSharding, PartitionSpec as P

    param_shapes = {
        "embed": jax.ShapeDtypeStruct((16, 8), jnp.float32),
        "dense": {
            "kernel": jax.ShapeDtypeStruct((8, 8), jnp.float32),
            "bias": jax.ShapeDtypeStruct((8,), jnp.float32),
        },
    }
    optimizer = optax.adamw(learning_rate=1e-3)
    template = jax.eval_shape(
        lambda params: train_state.TrainState.create(
            apply_fn=apply_model,
            params=params,
            tx=optimizer,
        ),
        param_shapes,
    )

    def add_sharding(leaf):
        partitions = P() if not leaf.shape else P("shard", *([None] * (leaf.ndim - 1)))
        return jax.ShapeDtypeStruct(
            leaf.shape,
            leaf.dtype,
            sharding=NamedSharding(mesh, partitions),
            weak_type=leaf.weak_type,
        )

    return jax.tree.map(add_sharding, template)


def make_state(template):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax.training import train_state

    offsets = iter((0, 1000, 2000))

    def materialize(leaf):
        offset = next(offsets)
        values = (np.arange(math.prod(leaf.shape)) + offset).reshape(leaf.shape)
        values = values.astype(leaf.dtype) / 128
        return jax.make_array_from_callback(
            leaf.shape,
            leaf.sharding,
            lambda index: values[index],
        )

    params = jax.tree.map(materialize, template.params)
    shardings = jax.tree.map(lambda leaf: leaf.sharding, template)
    state = jax.jit(
        lambda parameters: train_state.TrainState.create(
            apply_fn=apply_model,
            params=parameters,
            tx=template.tx,
        ),
        out_shardings=shardings,
    )(params)
    gradients = jax.tree.map(jnp.ones_like, state.params)
    return jax.jit(
        lambda current, grads: current.apply_gradients(grads=grads),
        out_shardings=shardings,
    )(state, gradients)


def tree_signature(tree):
    import jax
    import jax.numpy as jnp

    signature = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        total = float(jax.device_get(jnp.sum(leaf)))
        square_total = float(jax.device_get(jnp.sum(jnp.square(leaf))))
        signature.append(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(leaf.shape),
                "dtype": str(leaf.dtype),
                "sum": total,
                "square_sum": square_total,
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
            actual_leaf["sum"], expected_leaf["sum"], rel_tol=1e-5, abs_tol=1e-5
        ), (actual_leaf, expected_leaf)
        assert math.isclose(
            actual_leaf["square_sum"],
            expected_leaf["square_sum"],
            rel_tol=1e-5,
            abs_tol=1e-5,
        ), (actual_leaf, expected_leaf)


def save_checkpoint(root, mesh):
    import jax

    from theseus.checkpoint import CheckpointManager

    template = make_template(mesh)
    state = make_state(template)
    manager = CheckpointManager()
    manager.save(state, root / "state")
    manager.checkpointer.wait_until_finished()
    signature = tree_signature(state)

    if jax.process_index() == 0:
        (root / "expected.json").write_text(json.dumps(signature))


def restore_checkpoint(root, mesh):
    import jax

    from theseus.checkpoint import CheckpointManager

    template = make_template(mesh)
    assert not any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(template))

    restored = CheckpointManager().restore(root / "state", template)
    for restored_leaf, template_leaf in zip(
        jax.tree.leaves(restored), jax.tree.leaves(template), strict=True
    ):
        assert restored_leaf.sharding.is_equivalent_to(
            template_leaf.sharding, restored_leaf.ndim
        )
        assert {
            shard.device: shard.index for shard in restored_leaf.addressable_shards
        } == template_leaf.sharding.addressable_devices_indices_map(template_leaf.shape)

    expected = json.loads((root / "expected.json").read_text())
    assert_signature(tree_signature(restored), expected)


def run_worker(args):
    import numpy as np

    import jax
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh

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

    try:
        assert jax.process_count() == args.processes
        assert jax.local_device_count() == args.devices
        assert jax.device_count() == math.prod(args.mesh)
        mesh = Mesh(np.asarray(jax.devices()).reshape(args.mesh), ("batch", "shard"))
        action = save_checkpoint if args.action == "save" else restore_checkpoint
        action(Path(args.root), mesh)
        multihost_utils.sync_global_devices(f"checkpoint-topology:{args.action}")
    finally:
        if args.processes > 1:
            jax.distributed.shutdown()


def available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def run_phase(root, action, topology):
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
            "--mesh",
            *(str(axis) for axis in topology.mesh),
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
            outputs.append(process.communicate(timeout=180)[0])
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
        raise

    failures = [process.returncode for process in processes if process.returncode]
    if failures:
        raise RuntimeError("".join(outputs))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--action", choices=("save", "restore"))
    parser.add_argument("--root")
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--mesh", nargs=2, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
        return

    with tempfile.TemporaryDirectory(prefix="theseus-checkpoint-topology-") as temp:
        for scenario in SCENARIOS:
            root = Path(temp) / scenario.name
            print(
                f"{scenario.name}: save {scenario.saved} -> restore {scenario.restored}"
            )
            run_phase(root, "save", scenario.saved)
            run_phase(root, "restore", scenario.restored)
            print(f"{scenario.name}: PASS")


if __name__ == "__main__":
    main()
