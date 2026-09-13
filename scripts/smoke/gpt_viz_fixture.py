#!/usr/bin/env python3
"""Train an instrumented tiny GPT and verify its visualization records.

The model is the GPT used by ``PretrainGPT``, configured with two 128-wide
layers. Synthetic token batches keep the smoke self-contained, while the real
training, validation, checkpoint, ``sow()``, and asynchronous record paths run
unchanged.

Run:
    uv run python scripts/smoke/gpt_viz_fixture.py
"""

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

import theseus.model.models.base as gpt_module
from theseus.base import ExecutionSpec, PyTree
from theseus.config import configure
from theseus.experiments.models.gpt import PretrainGPT
from theseus.model.attention.rope import RopeAttention
from theseus.model.block.block import Block
from theseus.model.layers.layernorm import LayerNorm
from theseus.model.layers.mlp import MLP
from theseus.quick import quick
from theseus.registry import job
from theseus.store import RecordStore
from theseus.training.base import BaseTrainer, BaseTrainerConfig


class InstrumentedAttention(RopeAttention):
    """RoPE attention that records its tensors through Flax ``sow()``."""

    def attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        output = super().attn(q, k, v, mask, **kwargs)
        if self.is_mutable_collection("plots"):
            scores = jnp.einsum(
                "bqhd,bkhd->bhqk",
                q.astype(jnp.float32),
                k.astype(jnp.float32),
            ) / jnp.sqrt(q.shape[-1])
            if mask is not None:
                scores = jnp.where(mask, scores, -jnp.inf)

            self.sow("plots", "queries", q)
            self.sow("plots", "keys", k)
            self.sow("plots", "values", v)
            self.sow("plots", "attention_weights", jax.nn.softmax(scores, axis=-1))
            self.sow("plots", "attention_output", output)
        return output


class InstrumentedBlock(Block):
    """Standard GPT block configured with instrumented attention."""

    def setup(self) -> None:
        self.ln_1 = configure(LayerNorm)
        self.attn = configure(InstrumentedAttention)
        self.ln_2 = configure(LayerNorm)
        self.mlp = configure(MLP)


# GPT resolves Block from this module at setup time. Keep the instrumentation
# local to this smoke-test process while exercising the exact PretrainGPT model.
gpt_module.Block = InstrumentedBlock


@job("fixtures/gpt-viz-quick")
class FixtureTrainer(BaseTrainer[BaseTrainerConfig, Any]):
    """Tiny real trainer backed by deterministic synthetic token batches."""

    MODEL = PretrainGPT.MODEL
    CONFIG = BaseTrainerConfig
    DATASET = []
    EVALUATION = []

    def _init_data(self, spec: ExecutionSpec) -> None:
        del spec
        self._batch_index = 0

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        offset = self._batch_index if slice == "train" else 0
        if slice == "train":
            self._batch_index += 1
        tokens = (np.arange(self.args.block_size) + offset) % self.model.vocab_size
        x = tokens[None, :].astype(np.int32)
        return {
            "x": x,
            "y": np.roll(x, -1, axis=-1),
            "padding_mask": np.ones_like(x, dtype=np.bool_),
        }

    def evaluator(self) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("~/theseus").expanduser(),
        help="Theseus root (default: ~/theseus)",
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--name", default="gpt-viz-quick-fixture")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.root.mkdir(parents=True, exist_ok=True)

    with quick(str(args.root)) as q:
        q.build(FixtureTrainer, args.name, project="fixtures", group="viz")
        q.config.architecture.n_layers = 2
        q.config.architecture.n_embd = 128
        q.config.architecture.n_head = 4
        q.config.architecture.intermediate_size = 256
        q.config.architecture.block_size = 16
        q.config.architecture.vocab_size = 256
        q.config.architecture.dropout = 0.0
        q.config.architecture.dtype.param = "float32"
        q.config.architecture.dtype.activation = "float32"
        q.config.architecture.instrumentation.residual = True
        q.config.training.batch_size = 1
        q.config.training.per_device_batch_size = 1
        q.config.training.tokens = args.steps * q.config.architecture.block_size
        q.config.training.validation = True
        q.config.training.evaluate = False
        q.config.logging.report_interval = 1
        q.config.logging.validation_interval = 1
        q.config.logging.checkpoint_interval = args.steps + 1
        trainer = q.create()
        q()

    store = RecordStore(trainer.spec.hardware)
    try:
        selected = (
            store.query()
            .name(trainer.node_name(trainer.spec))
            .nonce(trainer.node.nonce)
            .has("train/loss")
            .select(return_nodes=True)
        )
        nodes = [node for node, _ in selected]
        rows = [row for _, row in selected]
        checkpoints = (
            store.query()
            .name(trainer.node_name(trainer.spec))
            .nonce(trainer.node.nonce)
            .checkpoint()
            .all()
        )
        artifacts = (
            store.query()
            .name(trainer.node_name(trainer.spec))
            .nonce(trainer.node.nonce)
            .artifact()
            .all()
        )
        records = [
            Path(store.get(node)["blob"]) / "validation.msgpack" for node in nodes
        ]

        if len(nodes) != args.steps:
            raise RuntimeError(
                f"expected {args.steps} logged nodes, found {len(nodes)}"
            )
        if len(checkpoints) != args.steps:
            raise RuntimeError(
                f"expected {args.steps} checkpoints, found {len(checkpoints)}"
            )
        if len(artifacts) != args.steps:
            raise RuntimeError(
                f"expected {args.steps} artifacts, found {len(artifacts)}"
            )
        if artifacts != checkpoints:
            raise RuntimeError("fixture checkpoints and artifacts do not match")
        if not all(record.is_file() for record in records):
            raise RuntimeError("one or more validation records are missing")

        payloads = rows[-1]["blob"]
        if not isinstance(payloads, list) or not payloads:
            raise RuntimeError("validation record did not decode")
        sample = payloads[0]
        plot_paths = sorted(
            "/".join(path) for path in flatten_dict(sample["plots"]).keys()
        )
        expected_paths = {
            f"blocks_{layer}/attn/{value}/0"
            for layer in range(2)
            for value in (
                "queries",
                "keys",
                "values",
                "attention_weights",
                "attention_output",
            )
        }
        if missing := sorted(expected_paths.difference(plot_paths)):
            raise RuntimeError(f"attention records are missing: {missing}")

        summary = {
            "name": trainer.node_name(trainer.spec),
            "nonce": nodes[0].nonce,
            "steps": len(nodes),
            "checkpoints": len(checkpoints),
            "artifacts": len(artifacts),
            "loss": [rows[0]["train/loss"], rows[-1]["train/loss"]],
            "sample": str(records[-1]),
            "plot_paths": plot_paths,
        }
    finally:
        store.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
