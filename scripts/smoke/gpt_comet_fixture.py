#!/usr/bin/env python3
"""Run a laptop-sized GPT through quick, logging to the configured Comet account.

Run: uv run python scripts/smoke/gpt_comet_fixture.py
Use --no-remote to exercise the same training path without Comet.
"""

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from theseus.base import ExecutionSpec, PyTree
from theseus.model.models.base import GPT
from theseus.quick import quick
from theseus.training.base import BaseTrainer, BaseTrainerConfig


class TinyGPT(BaseTrainer[BaseTrainerConfig, GPT]):
    """One tiny GPT trained on a repeating, synthetic token sequence."""

    MODEL = GPT
    CONFIG = BaseTrainerConfig
    DATASET = []

    def _init_data(self, spec: ExecutionSpec) -> None:
        self._tokens = np.arange(self.args.block_size + 1, dtype=np.int32)[None, :]

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        return {
            "x": self._tokens[:, :-1],
            "y": self._tokens[:, 1:],
            "padding_mask": np.ones_like(self._tokens[:, :-1], dtype=np.bool_),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("~/theseus-comet-smoke").expanduser()
    )
    parser.add_argument("--no-remote", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    with quick(args.root) as q:
        q.build(TinyGPT, "tiny-gpt-comet", project="fixtures", group="laptop")
        q.config.architecture.n_layers = 1
        q.config.architecture.n_embd = 16
        q.config.architecture.n_head = 2
        q.config.architecture.intermediate_size = 32
        q.config.architecture.block_size = 8
        q.config.architecture.vocab_size = 32
        q.config.architecture.dtype.param = "float32"
        q.config.architecture.dtype.activation = "float32"
        q.config.training.batch_size = 1
        q.config.training.per_device_batch_size = 1
        q.config.training.tokens = 32
        q.config.training.validation = False
        q.config.training.evaluate = False
        q.config.logging.report_interval = 1
        q.config.logging.checkpoint_interval = 4
        q.config.logging.validation_interval = 999
        q.config.logging.remote = not args.no_remote
        trainer = q.create()
        q()
        rows = (
            q.find().name(trainer.node.name).nonce(trainer.node.nonce).select(raw=True)
        )
        losses = [cast(float, row["train/loss"]) for row in rows if "train/loss" in row]
        checkpoints = [row["_x_seq"] for row in rows if row.get("_x_checkpoint")]
        assert len(losses) == 4 and np.isfinite(losses).all(), losses
        assert checkpoints == [2, 4], checkpoints
        print(
            json.dumps(
                {
                    "root": str(args.root),
                    "node": trainer.node.serialize(),
                    "losses": losses,
                    "checkpoint_steps": checkpoints,
                },
                indent=2,
            )
        )
