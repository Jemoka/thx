#!/usr/bin/env python3
"""Tiny GPT: all 32 checkpoint transfers between sharding policies on 2/4 CPUs.

Run with `.venv/bin/python scripts/e2e/test_checkpoint_sharding.py`.
Workers run sequentially. Each source trains twice and checkpoints through
BaseTrainer; each target verifies every restored leaf and a further training
step against uninterrupted execution (float32, rtol=1e-5, atol=1e-6).

On two devices TP+FSDP uses TP=2, so the FSDP mesh dimension is size one;
on four devices it uses TP=2 and FSDP=2. Plain TP uses every device.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

POLICIES = ("tp", "ddp", "fsdp", "tp_fsdp")


def run_worker(args):
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from loguru import logger

    from theseus.base import Axis, ExecutionSpec, ShardingPolicy
    from theseus.config import build, configuration
    from theseus.model.models import GPT
    from theseus.training.base import BaseTrainer, BaseTrainerConfig

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    assert jax.device_count() == args.devices

    class CheckpointTrainer(BaseTrainer):
        MODEL = GPT
        CONFIG = BaseTrainerConfig
        DATASET = []

        def _init_data(self, spec):
            # The fixture supplies a deterministic token batch directly.
            pass

        def evaluator(self):
            return None

    policy = ShardingPolicy(
        tp=args.devices
        if args.policy == "tp"
        else 2
        if args.policy == "tp_fsdp"
        else 1,
        fsdp=args.policy in ("fsdp", "tp_fsdp"),
        zero=args.policy != "tp",
    )
    cfg = build(*CheckpointTrainer.config())
    cfg.architecture.n_layers = 2
    cfg.architecture.n_embd = 16
    cfg.architecture.n_head = 4
    cfg.architecture.vocab_size = 32
    cfg.architecture.block_size = 8
    cfg.architecture.dropout = 0.0
    cfg.architecture.dtype.param = "float32"
    cfg.architecture.dtype.activation = "float32"
    cfg.training.batch_size = 4
    cfg.training.per_device_batch_size = 4 // (args.devices // policy.tp)
    cfg.training.tokens = 128
    cfg.training.validation = False
    cfg.training.evaluate = False
    cfg.training.analyze = False
    root = Path(args.root)
    destination = root / f"{args.devices}-{args.policy}"
    (destination / args.action).mkdir(parents=True, exist_ok=True)
    spec = ExecutionSpec.local(str(destination / args.action), shard=policy)
    with configuration(cfg):
        trainer = CheckpointTrainer(spec)
        try:
            trainer.initialize()
            step = trainer._make_train_step()
            tokens = np.arange(32, dtype=np.int32).reshape(1, 4, 8) % 32
            batch = jax.device_put(
                {
                    "x": tokens,
                    "y": (tokens + 1) % 32,
                    "padding_mask": np.ones_like(tokens),
                },
                NamedSharding(trainer.mesh, P(None, Axis.BATCH, None)),
            )
            key = jax.random.PRNGKey(19)
            if args.action == "save":
                for _ in range(2):
                    trainer.state, *_ = step(trainer.state, batch, key, 1)
                trainer.checkpoint()
                trainer.chkpt_manager.checkpointer.wait_until_finished()
                trainer.store.close()
                row = trainer.store.query().node(trainer.node).select(raw=True)[0]
                (destination / "checkpoint.json").write_text(
                    json.dumps({"blob": str(row["blob"])})
                )
                save_values(destination / "saved.npz", trainer.state)
                trainer.state, loss, _, norm = step(trainer.state, batch, key, 1)
                save_values(destination / "continued.npz", (trainer.state, loss, norm))
            else:
                for source in POLICIES:
                    origin = root / f"{6 - args.devices}-{source}"
                    row = json.loads((origin / "checkpoint.json").read_text())
                    trainer.state_restore(row)
                    assert int(trainer.state.step) == 2
                    assert_values(origin / "saved.npz", trainer.state, exact=True)
                    for value, shape in zip(
                        jax.tree.leaves(trainer.state),
                        jax.tree.leaves(trainer.template),
                        strict=True,
                    ):
                        assert value.sharding.is_equivalent_to(
                            shape.sharding, value.ndim
                        )
                        assert {
                            shard.device: shard.index
                            for shard in value.addressable_shards
                        } == (
                            shape.sharding.addressable_devices_indices_map(shape.shape)
                        )
                    trainer.state, loss, _, norm = step(trainer.state, batch, key, 1)
                    assert_values(origin / "continued.npz", (trainer.state, loss, norm))
                    print(
                        f"PASS {6 - args.devices}:{source} -> {args.devices}:{args.policy}",
                        flush=True,
                    )
        finally:
            trainer.finish()


def save_values(path, tree):
    import jax
    import numpy as np

    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    np.savez(
        path, **{jax.tree_util.keystr(key): np.asarray(value) for key, value in leaves}
    )


def assert_values(path, tree, exact=False):
    import jax
    import numpy as np

    with np.load(path) as expected:
        leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
        assert set(expected.files) == {jax.tree_util.keystr(key) for key, _ in leaves}
        for key, value in leaves:
            name = jax.tree_util.keystr(key)
            reference = expected[name]
            assert value.shape == reference.shape, name
            assert value.dtype == reference.dtype, name
            np.testing.assert_allclose(
                value,
                reference,
                rtol=0 if exact else 1e-5,
                atol=0 if exact else 1e-6,
                err_msg=name,
            )


def run_phase(root, action, devices, policy):
    env = os.environ.copy()
    env.update(
        JAX_PLATFORMS="cpu",
        JAX_NUM_CPU_DEVICES=str(devices),
        XLA_FLAGS=f"--xla_force_host_platform_device_count={devices}",
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--action",
            action,
            "--devices",
            str(devices),
            "--policy",
            policy,
            "--root",
            str(root),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    (root / f"{action}-{devices}-{policy}.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout)
    print(
        result.stdout.strip() if action == "restore" else f"Saved {devices}:{policy}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, help="Keep checkpoints and worker logs here"
    )
    parser.add_argument("--action", choices=("save", "restore"), help=argparse.SUPPRESS)
    parser.add_argument("--devices", type=int, choices=(2, 4), help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=POLICIES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action:
        run_worker(args)
    else:
        from contextlib import nullcontext

        with (
            tempfile.TemporaryDirectory(prefix="theseus-checkpoint-sharding-")
            if args.root is None
            else nullcontext(args.root) as root
        ):
            root = Path(root)
            root.mkdir(parents=True, exist_ok=True)
            for action in ("save", "restore"):
                for devices in (2, 4):
                    for policy in POLICIES:
                        run_phase(root, action, devices, policy)
            print("PASS: all 32 checkpoint transfers, including continued training.")
