"""Synthetic GPT using the real trainer; measure without a profiler attachment.

Generate launch YAML with --import scripts.smoke.fsdp_memory configure
diagnostic/fsdp-memory. Compare identical YAMLs across trainer policy commits.
Compiler buffer assignments can be emitted via worker XLA_FLAGS; this fixture
does not manually lower or compile, preserving the normal PGLE path.
"""

from collections.abc import Callable
from time import perf_counter
from typing import Any

import jax
import numpy as np

from theseus.base import ExecutionSpec, PyTree
from theseus.model.models import GPT
from theseus.registry import job
from theseus.training.base import BaseTrainer, BaseTrainerConfig


@job("diagnostic/fsdp-memory")
class MemoryGPT(BaseTrainer[BaseTrainerConfig, GPT]):
    MODEL = GPT
    CONFIG = BaseTrainerConfig
    DATASET = []

    def _init_data(self, spec: ExecutionSpec) -> None:
        rows = self.args.batch_size // jax.process_count()
        tokens = np.arange(rows * (self.args.block_size + 1), dtype=np.int32)
        tokens = tokens.reshape(rows, self.args.block_size + 1) % self.model.vocab_size
        self._batch = {
            "x": tokens[:, :-1],
            "y": tokens[:, 1:],
            "padding_mask": np.ones_like(tokens[:, :-1], dtype=np.bool_),
        }

    def batch(self, slice: str = "train") -> PyTree[np.ndarray]:
        return self._batch

    def _make_train_step(self) -> Callable[..., Any]:
        step = super()._make_train_step()

        def measured_step(*args: Any, **kwargs: Any) -> Any:
            # Include execution/compilation, exclude data placement and store I/O.
            jax.block_until_ready(args)
            started = perf_counter()
            state, loss, meta, norm = jax.block_until_ready(step(*args, **kwargs))
            seconds = perf_counter() - started
            metrics = {
                **meta,
                "diagnostic/step_seconds": seconds,
                "diagnostic/tokens_per_second": (
                    self.args.batch_size * self.args.block_size / seconds
                ),
            }
            for device in jax.local_devices():
                for name, value in (device.memory_stats() or {}).items():
                    if isinstance(value, (int, float)):
                        metrics[f"diagnostic/device_{device.id}/{name}"] = value
            return state, loss, metrics, norm

        return measured_step
