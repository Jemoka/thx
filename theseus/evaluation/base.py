"""
Evaluation framework for theseus trainers.

Provides abstract base classes for different evaluation types:
- RolloutEvaluation: Autoregressive generation tasks
- EncodingEvaluation: Next-token prediction accuracy
- PerplexityEvaluation: Dataset perplexity (returns ppl, lower is better)
- BitsPerByteEvaluation: Dataset bits-per-byte (lower is better)
- PerplexityComparisonEvaluation: Multiple-choice via perplexity comparison

Also provides:
- Evaluator: InferenceJob subclass that runs multiple evaluations
"""

import json
import random
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Tuple,
    List,
    Optional,
    Union,
    Generic,
    Literal,
    TYPE_CHECKING,
)

import numpy as np

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P
from loguru import logger

from theseus.base import Axis, ExecutionSpec, Node

from theseus.config import field, configure
from theseus.inference.base import InferenceJob, M
from theseus.model.module import Module
from theseus.data.tokenizer import Tokenizer, TokenizerConfig, get_tokenizer
from theseus.data.datasets.dataset import ChatTemplate

if TYPE_CHECKING:
    from theseus.training.base import BaseTrainer


def _select_indices(inference: Any, n: int) -> list[int]:
    """Pick which examples to evaluate from an n-sample dataset.

    Reads ``inference.length`` (set by ``Evaluator``); when 0 < length < n,
    samples that many indices via the evaluator's per-call-deterministic
    ``random.Random``. Otherwise returns ``range(n)``. Every host computes
    the same indices because every Evaluator's RNG is seeded identically.
    """
    length = getattr(inference, "length", -1)
    rng = getattr(inference, "random", None)
    if rng is None or length <= 0 or length >= n:
        return list(range(n))
    sampled: list[int] = rng.sample(range(n), length)
    return sampled


def _pad_eval_inputs(
    batch_unit: int, *sequences: list[Any]
) -> tuple[int, tuple[list[Any], ...]]:
    """Pad example lists to a multiple of the global batch size by repeating the last item."""
    if not sequences:
        raise ValueError("Expected at least one sequence to pad.")

    original_size = len(sequences[0])
    if original_size == 0:
        raise ValueError("Evaluation dataset is empty.")

    if any(len(seq) != original_size for seq in sequences[1:]):
        raise ValueError("All evaluation sequences must have the same length.")

    padded_size = ((original_size + batch_unit - 1) // batch_unit) * batch_unit
    pad_count = padded_size - original_size
    if pad_count == 0:
        return original_size, tuple(sequences)

    padded = tuple(seq + [seq[-1]] * pad_count for seq in sequences)
    return original_size, padded


class Evaluation(ABC):
    """Abstract base class for all evaluations."""

    CONFIG: ClassVar[type[Any] | None] = None

    @classmethod
    def config(cls) -> List[type[Any]]:
        """Return configuration schemas required by this evaluation.

        Returns:
            The evaluation's schema, or an empty list when it has none.
        """

        return [cls.CONFIG] if cls.CONFIG is not None else []

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this evaluation."""
        ...

    def prefix(self) -> str:
        """Prefix for metrics from this evaluation."""
        return self.name + "_"

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in this evaluation."""
        ...

    @abstractmethod
    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the evaluation and return a score (and optionally intermediates).

        When ``return_intermediates=True``, also returns a list of
        ``(x, padding_mask)`` numpy arrays — one per sample — available on every
        host so that an RL trainer can use them as a training batch.
        """
        ...

    def _score(self, *args: Any, reduce: str = "mean") -> Any:
        """Reduce per-sample scores from ``self.score(...)`` into a final value.

        ``reduce="mean"`` and ``"sum"`` return a Python float;
        ``reduce="none"`` returns the per-sample np.ndarray.
        """
        per_sample = np.asarray(list(self.score(*args)), dtype=np.float32)
        if reduce == "mean":
            return float(per_sample.mean()) if per_sample.size > 0 else 0.0
        if reduce == "sum":
            return float(per_sample.sum())
        if reduce == "none":
            return per_sample
        raise ValueError(f"unknown reduce mode: {reduce!r}")

    def score(self, *args: Any) -> List[float]:
        """Return one float per evaluation sample. Subclasses override."""
        raise NotImplementedError("Override score() to return one float per sample.")

    @staticmethod
    def find_accumulation_steps(
        dataset_size: int, max_batch_size: int, dp_replicate: int
    ) -> Tuple[int, int] | Tuple[None, None]:
        """Find batch size and accumulation steps that evenly divide dataset.

        Args:
            dataset_size: Total number of samples
            max_batch_size: Maximum per-device batch size
            dp_replicate: Data parallel replication factor

        Returns:
            (batch_size, accumulation_steps) or (None, None) if no valid size found
        """
        for batch_size in reversed(range(1, max_batch_size + 1)):
            if dataset_size % (batch_size * dp_replicate) == 0:
                return batch_size, dataset_size // (batch_size * dp_replicate)

        logger.warning(
            f"Couldn't find a good size for dataset_size {dataset_size} "
            f"and max_batch_size {max_batch_size} and dp_replicate {dp_replicate}. "
            "Will chop off the end of this evaluation."
        )
        return None, None


class RolloutEvaluation(Evaluation):
    """Evaluation using autoregressive generation."""

    # Cached on first __call__ so PPO refills (and other repeated callers)
    # reuse the compiled chunk function instead of re-tracing every time.
    # Class-level defaults keep subclasses safe even if they override
    # __init__ without calling super().
    _chunk_jit: Optional[Any] = None
    _evaluator_ref: Optional[Any] = None

    def score(self, ys: list[str], y_hats: list[str]) -> List[float]:
        """Per-sample scores. Default: cast each ``check()`` to float."""
        return [float(self.check(y, y_hat)) for y, y_hat in zip(ys, y_hats)]

    def check(self, y: str, y_hat: str) -> bool:
        """Check if y_hat matches y.

        Args:
            y: Ground truth
            y_hat: Generated result

        Returns:
            Whether y_hat matches y
        """
        raise NotImplementedError(
            "Please override this method or self.score, not neither!"
        )

    @abstractmethod
    def clean(self, y_hat: str) -> str:
        """Clean generated result before checking.

        Args:
            y_hat: Generated result, which *can include* the prompt

        Returns:
            Cleaned/normalized result available for comparison
        """
        ...

    @abstractmethod
    def get(self, indx: int) -> Tuple[str, str]:
        """Get sample at index.

        Returns:
            (input_string, expected_output_string)
        """
        ...

    def max_new_tokens(self, inference: "InferenceJob[Any, M]") -> int:
        """Maximum tokens to generate. Subclasses MUST override.

        Drives the prompt/generation split (``prompt_max = block_size -
        max_new_tokens``) so the JIT shapes are constant across refills —
        defaulting to ``block_size`` would leave zero room for prompts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override max_new_tokens()."
        )

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        chunk_size: int = 200,
        samples_per_prompt: int = 1,
        **kwargs: Any,
    ) -> Any:
        """Run evaluation.

        Args:
            inference: InferenceJob instance for running inference
            encoding: Tokenizer with encode_batch/decode_batch methods
            reduce: how to reduce per-sample scores ("mean" | "sum" | "none")
            return_intermediates: also return per-sample (rollout, mask) numpy
                arrays on every host (for RL consumers).
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Nucleus sampling threshold
            chunk_size: Number of batches per JIT chunk (default 200)

        Returns:
            Evaluation score, or (score, intermediates) when return_intermediates.
        """
        # Stash the inference handle so subclasses' score()/clean() can report
        # side metrics through inference.log.
        # Mirrors the pattern EncodingEvaluation uses for its chunk_jit cache.
        self._evaluator_ref = inference

        batch_unit = inference.replicas * inference.per_device_batch_size
        indices = _select_indices(inference, len(self))
        original_size = len(indices)

        # Upper bound on the prompt window. The actual prompt_max is shrunk
        # below to the longest real prompt (rounded to a coarse bucket) so the
        # KV cache / left-padding aren't sized to the model's full context —
        # block_size is 262144 for Qwen3.5, which makes a full-context prompt
        # window blow up XLA compile/memory during rollout.
        max_new_tokens = self.max_new_tokens(inference)
        prompt_cap = inference.block_size - max_new_tokens
        if prompt_cap <= 0:
            raise ValueError(
                f"{self.name}: max_new_tokens={max_new_tokens} leaves no "
                f"room under inference.block_size={inference.block_size}."
            )

        batch_unit = inference.replicas * inference.per_device_batch_size
        indices = _select_indices(inference, len(self))
        if samples_per_prompt > 1:
            # Replicate each selected index G times consecutively so callers
            # (e.g. GRPO) get [p0_s0, p0_s1, ..., p0_s(G-1), p1_s0, ...]. The
            # G copies of each prompt diverge at sampling time via temperature.
            indices = [i for i in indices for _ in range(samples_per_prompt)]
        original_size = len(indices)

        # ──────────────────────────────────────────────────────────────────
        # ORDERING CONTRACT — DO NOT SHUFFLE.
        # `indices` and every per-rollout array derived from it (x_raw, y_raw,
        # encoded, rollout_inputs, raw_rollouts_np, decoded_results,
        # intermediates) MUST stay in the order produced above. GRPO assumes
        # the buffer arrives as G consecutive same-prompt rollouts per slot;
        # any shuffle here silently breaks group-relative advantage z-scoring.
        # If you need stochastic order, do it BEFORE _select_indices or AFTER
        # the trainer has consumed the buffer — never in between.
        # ──────────────────────────────────────────────────────────────────

        if jax.process_index() == 0:
            x_raw, y_raw = zip(*[self.get(i) for i in indices])
            x = list(x_raw)
            original_y = list(y_raw)

            # _pad_eval_inputs only APPENDS (repeats the last item); preserves
            # leading order. Do not change it to interleave/shuffle padding.
            _, (x, original_y) = _pad_eval_inputs(batch_unit, x, original_y)

            encoded = encoding.encode_batch(x, allowed_special="all")
            encoded = [seq[:prompt_cap] for seq in encoded]
            # Size the prompt window to the longest real prompt, rounded up to
            # a coarse bucket so distinct sampled subsets still share JIT
            # shapes. Capped at prompt_cap, so no prompt is truncated further
            # than the old block_size-based bound would have.
            longest = max((len(seq) for seq in encoded), default=1)
            prompt_max_arr = np.asarray(
                min(prompt_cap, ((longest + 63) // 64) * 64), dtype=np.int32
            )
            prompt_lengths = [len(seq) for seq in encoded]

            rollout_inputs = encoded
        else:
            original_y = None
            prompt_lengths = None
            rollout_inputs = None
            prompt_max_arr = np.asarray(0, dtype=np.int32)

        # All hosts must agree on the prompt window (it drives rollout's static
        # shapes); broadcast it from process 0 alongside the inputs.
        prompt_max = int(multihost_utils.broadcast_one_to_all(prompt_max_arr))

        # rollout itself broadcasts from process 0, so all hosts need the same Python
        # call. Nonzero hosts can pass a dummy list with the right nonempty type;
        # process 0's broadcasted xs/masks are the real source of data.
        rollout_inputs = multihost_utils.broadcast_one_to_all(rollout_inputs)

        raw_rollouts = inference.rollout(
            rollout_inputs,
            encoding=encoding,
            max_new_tokens=max_new_tokens,
            max_prompt_length=prompt_max,
            temperature=temperature,
            top_p=top_p,
            chunk_size=chunk_size,
            return_type="raw_indices",
        )

        raw_rollouts_np = np.asarray(raw_rollouts, dtype=np.int32)

        # Score generated text only.
        generated_rows = raw_rollouts_np[:original_size, prompt_max:].tolist()
        eot_token = getattr(encoding, "eot_token", None)
        if eot_token is not None:
            generated_rows = [
                row[: row.index(eot_token)] if eot_token in row else row
                for row in generated_rows
            ]
        decoded_results = encoding.decode_batch(generated_rows)

        if jax.process_index() == 0:
            assert original_y is not None
            score = self._score(
                original_y[:original_size],
                [self.clean(i) for i in decoded_results],
                reduce=reduce,
            )
        else:
            score = (
                np.zeros(original_size, dtype=np.float32) if reduce == "none" else 0.0
            )

        score = multihost_utils.broadcast_one_to_all(jnp.asarray(score))
        score = np.asarray(score) if reduce == "none" else float(score)

        if not return_intermediates:
            return score

        if jax.process_index() == 0:
            assert prompt_lengths is not None

            T_total = prompt_max + max_new_tokens
            positions = np.arange(T_total)

            base_action_mask = positions >= prompt_max

            # Built in dataset-index order — must match `indices` 1:1 so
            # GRPO's same-prompt grouping holds. Do not reorder, sort, or
            # shuffle this list.
            intermediates = []
            for i in range(original_size):
                padding_mask = positions >= (prompt_max - prompt_lengths[i])
                intermediates.append(
                    (
                        raw_rollouts_np[i],
                        base_action_mask.copy(),
                        padding_mask,
                    )
                )
        else:
            intermediates = []

        intermediates = multihost_utils.broadcast_one_to_all(intermediates)
        return score, intermediates


class EncodingEvaluation(Evaluation):
    """Evaluation using next-token prediction accuracy."""

    # See ``RolloutEvaluation`` — same compile-once-per-instance pattern.
    _chunk_jit: Optional[Any] = None
    _evaluator_ref: Optional[Any] = None

    def _chunk_step(
        self,
        state: Any,
        xs_chunk: Any,
        masks_chunk: Any,
    ) -> Any:
        inference = self._evaluator_ref
        assert inference is not None, (
            "_chunk_step called before __call__ set _evaluator_ref"
        )

        assert inference.spec.topology is not None
        fsdp = inference.spec.topology.shard.fsdp

        def reduce(_: Any, batch: Any) -> Any:
            params = state.params
            if fsdp:
                # Keep gathered weights inside this batch iteration.
                params, batch = jax.lax.optimization_barrier((params, batch))  # type: ignore[no-untyped-call]
            x_batch, mask_batch = batch
            with (
                jax.sharding.use_abstract_mesh(inference.mesh.abstract_mesh),
                nn.logical_axis_rules(inference.model.sharding._tp),
            ):
                logits, _, _ = inference.forward(
                    state,
                    params,
                    (x_batch, None, mask_batch),
                    None,
                    deterministic=True,
                )
            predictions = jnp.argmax(logits[:, :-1, :], axis=-1)
            return None, predictions

        _, results = jax.lax.scan(reduce, None, (xs_chunk, masks_chunk))
        return jnp.reshape(results, (-1, results.shape[-1]))

    def score(self, xs: list[str], y_hats: list[str]) -> List[float]:
        """Per-sample scores. Default: cast each ``check()`` to float."""
        return [float(self.check(x, y_hat)) for x, y_hat in zip(xs, y_hats)]

    def check(self, x: str, y_hat: str) -> bool:
        """Check if prediction is correct given input.

        Args:
            x: Input string
            y_hat: Model prediction (cleaned, decoded argmax)

        Returns:
            Whether prediction is correct
        """
        raise NotImplementedError("Please override this method or self.score!")

    @abstractmethod
    def clean(self, y_hat: str) -> str:
        """Clean model prediction before checking.

        Args:
            y_hat: Raw decoded model prediction

        Returns:
            Cleaned/normalized result available for comparison
        """
        ...

    @abstractmethod
    def get(self, indx: int) -> str:
        """Get input string at index."""
        ...

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> Any:
        """Run evaluation."""
        eval_data = self
        batch_unit = inference.replicas * inference.per_device_batch_size
        indices = _select_indices(inference, len(eval_data))
        original_size = len(indices)

        # Gather and encode all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            original_x = [eval_data.get(i) for i in indices]
            _, (x,) = _pad_eval_inputs(batch_unit, original_x)
            encoded = encoding.encode_batch(x, allowed_special="all")
            encoded = [seq[: inference.block_size] for seq in encoded]
            xs, masks = inference.pad(encoded)
        else:
            original_x = None
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Keep a global view for intermediates.
        xs_full = xs
        masks_full = masks

        # Divide across processes
        pieces_xs = np.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = np.array_split(masks, jax.process_count(), axis=0)
        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]

        # Reshape into (accumulate_steps, per_device_batch_size, T)
        xs = xs.reshape(
            -1, inference.local_replicas * inference.per_device_batch_size, xs.shape[-1]
        )
        masks = masks.reshape(
            -1, inference.local_replicas * inference.per_device_batch_size, xs.shape[-1]
        )

        # Create global arrays
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        xs = multihost_utils.host_local_array_to_global_array(
            xs, inference.mesh, data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, inference.mesh, data_pspec
        )

        # Cached chunk JIT — see RolloutEvaluation for rationale.
        if self._chunk_jit is None:
            self._evaluator_ref = inference
            data_sharding = NamedSharding(inference.mesh, data_pspec)
            self._chunk_jit = jax.jit(
                self._chunk_step,
                in_shardings=(
                    inference.state_sharding,
                    data_sharding,
                    data_sharding,
                ),
                out_shardings=None,
            )

        # Process in chunks with progress logging
        num_batches = xs.shape[0]
        all_results = []

        if jax.process_index() == 0:
            logger.debug(
                "EVAL | {} | samples={} seq={} batches={}",
                eval_data.name,
                original_size,
                xs.shape[-1],
                num_batches,
            )

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]

            if chunk_start == 0:
                logger.debug(
                    "EVAL | {} | tracing+compiling first chunk", eval_data.name
                )
            if jax.process_index() == 0 and num_batches > chunk_size:
                logger.debug(
                    "EVAL | {} | chunk {}/{} ({:.0f}%)",
                    eval_data.name,
                    chunk_end,
                    num_batches,
                    (chunk_end / num_batches) * 100,
                )

            chunk_results = self._chunk_jit(inference.state, xs_chunk, masks_chunk)
            all_results.append(chunk_results)

        # Concatenate all chunk results
        results = jnp.concatenate(all_results, axis=0)

        # Collect across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        results = multihost_utils.process_allgather(results)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten and decode
        results = jnp.reshape(results, (-1, results.shape[-1]))
        decoded_outputs = encoding.decode_batch(results.tolist()[:original_size])

        # Score on process 0 only, then broadcast
        if jax.process_index() == 0:
            assert original_x is not None
            score = eval_data._score(
                original_x,
                [eval_data.clean(i) for i in decoded_outputs],
                reduce=reduce,
            )
        else:
            score = (
                np.zeros(original_size, dtype=np.float32) if reduce == "none" else 0.0
            )
        score = multihost_utils.broadcast_one_to_all(jnp.asarray(score))
        score = np.asarray(score) if reduce == "none" else float(score)

        if not return_intermediates:
            return score

        # No "action" notion for encoding evals — every real token is fair game.
        xs_np = np.asarray(xs_full)
        masks_np = np.asarray(masks_full).astype(bool)
        intermediates = [
            (xs_np[i], masks_np[i], masks_np[i]) for i in range(original_size)
        ]
        return score, intermediates


class PerplexityEvaluation(Evaluation):
    """Evaluation that computes dataset perplexity and returns ppl (lower is better).

    Runs a blockwise forward pass like EncodingEvaluation, computes the mean
    negative log-likelihood over all non-padding tokens, and returns perplexity.
    """

    # See ``RolloutEvaluation`` — same compile-once-per-instance pattern.
    _chunk_jit: Optional[Any] = None
    _evaluator_ref: Optional[Any] = None

    def _chunk_step(
        self,
        state: Any,
        xs_chunk: Any,
        masks_chunk: Any,
    ) -> Any:
        inference = self._evaluator_ref
        assert inference is not None, (
            "_chunk_step called before __call__ set _evaluator_ref"
        )

        assert inference.spec.topology is not None
        fsdp = inference.spec.topology.shard.fsdp

        def reduce(_: Any, batch: Any) -> Any:
            params = state.params
            if fsdp:
                # Keep gathered weights inside this batch iteration.
                params, batch = jax.lax.optimization_barrier((params, batch))  # type: ignore[no-untyped-call]
            x_batch, mask_batch = batch

            y_batch = jnp.roll(x_batch, -1, axis=-1)
            y_batch = y_batch.at[:, -1].set(-1)
            y_batch = jnp.where(mask_batch == 0, -1, y_batch)

            with (
                jax.sharding.use_abstract_mesh(inference.mesh.abstract_mesh),
                nn.logical_axis_rules(inference.model.sharding._tp),
            ):
                logits, _, _ = inference.forward(
                    state,
                    params,
                    (x_batch, None, mask_batch),
                    None,
                    deterministic=True,
                )

            logits_f32 = logits.astype(jnp.float32)
            log_probs = jax.nn.log_softmax(logits_f32, axis=-1)

            token_mask = y_batch != -1
            y_safe = jnp.where(token_mask, y_batch, 0)

            token_nll = -jnp.take_along_axis(
                log_probs, y_safe[..., None], axis=-1
            ).squeeze(-1)
            token_nll = jnp.where(token_mask, token_nll, 0.0)

            return None, jnp.stack(
                [
                    jnp.sum(token_nll, axis=-1),
                    jnp.sum(token_mask, axis=-1).astype(jnp.float32),
                ],
                axis=-1,
            )

        _, stats = jax.lax.scan(reduce, None, (xs_chunk, masks_chunk))
        return jnp.reshape(stats, (-1, 2))

    @abstractmethod
    def get(self, indx: int) -> str:
        """Get input string at index."""
        ...

    def score(
        self, per_sample_nll: np.ndarray, per_sample_count: np.ndarray
    ) -> List[float]:
        """Per-sample perplexity (= exp(nll / max(count, 1))."""
        nll = np.asarray(per_sample_nll, dtype=np.float64)
        count = np.maximum(np.asarray(per_sample_count, dtype=np.float64), 1.0)
        return list(np.exp(nll / count).astype(np.float32))

    def _score(  # type: ignore[override]
        self,
        per_sample_nll: np.ndarray,
        per_sample_count: np.ndarray,
        reduce: str = "mean",
    ) -> Any:
        """Token-weighted aggregate ppl for mean/sum; per-sample ppl for none."""
        if reduce == "none":
            return np.asarray(
                self.score(per_sample_nll, per_sample_count), dtype=np.float32
            )
        nll = float(np.asarray(per_sample_nll, dtype=np.float64).sum())
        count = float(np.asarray(per_sample_count, dtype=np.float64).sum())
        return float(np.exp(nll / max(count, 1.0)))

    def _score_from_stats(
        self,
        per_sample_nll: np.ndarray,
        per_sample_count: np.ndarray,
        texts: list[str],
        reduce: str = "mean",
    ) -> Any:
        del texts
        return self._score(per_sample_nll, per_sample_count, reduce=reduce)

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> Any:
        """Run evaluation."""
        eval_data = self
        batch_unit = inference.replicas * inference.per_device_batch_size
        indices = _select_indices(inference, len(eval_data))
        original_size = len(indices)

        # Gather and encode all data on main process
        texts: Optional[list[str]] = None
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            texts = [eval_data.get(i) for i in indices]
            x = texts
            _, (x,) = _pad_eval_inputs(batch_unit, x)
            encoded = encoding.encode_batch(x, allowed_special="all")
            encoded = [seq[: inference.block_size] for seq in encoded]
            xs, masks = inference.pad(encoded)
        else:
            xs, masks = None, None
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        xs_full = xs
        masks_full = masks

        # Divide across processes
        pieces_xs = np.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = np.array_split(masks, jax.process_count(), axis=0)
        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]

        # Reshape into (accumulate_steps, per_device_batch_size, T)
        xs = xs.reshape(
            -1, inference.local_replicas * inference.per_device_batch_size, xs.shape[-1]
        )
        masks = masks.reshape(
            -1, inference.local_replicas * inference.per_device_batch_size, xs.shape[-1]
        )

        # Create global arrays
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        xs = multihost_utils.host_local_array_to_global_array(
            xs, inference.mesh, data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, inference.mesh, data_pspec
        )

        # Cached chunk JIT — see RolloutEvaluation for rationale.
        if self._chunk_jit is None:
            self._evaluator_ref = inference
            data_sharding = NamedSharding(inference.mesh, data_pspec)
            self._chunk_jit = jax.jit(
                self._chunk_step,
                in_shardings=(
                    inference.state_sharding,
                    data_sharding,
                    data_sharding,
                ),
                out_shardings=None,
            )

        # Process in chunks with progress logging
        num_batches = xs.shape[0]
        all_stats = []

        if jax.process_index() == 0:
            logger.debug(
                "EVAL | {} | samples={} seq={} batches={}",
                eval_data.name,
                original_size,
                xs.shape[-1],
                num_batches,
            )

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]

            if chunk_start == 0:
                logger.debug(
                    "EVAL | {} | tracing+compiling first chunk", eval_data.name
                )
            if jax.process_index() == 0 and num_batches > chunk_size:
                logger.debug(
                    "EVAL | {} | chunk {}/{} ({:.0f}%)",
                    eval_data.name,
                    chunk_end,
                    num_batches,
                    (chunk_end / num_batches) * 100,
                )

            chunk_stats = self._chunk_jit(inference.state, xs_chunk, masks_chunk)
            all_stats.append(chunk_stats)

        # Concatenate all chunk stats: shape (total_local_examples, 2)
        stats = jnp.concatenate(all_stats, axis=0)

        # Gather across hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        stats = multihost_utils.process_allgather(stats)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten process dimension and drop repeated pad examples.
        stats = jnp.reshape(stats, (-1, 2))
        stats = stats[:original_size]
        stats_np = np.asarray(stats)
        per_sample_nll = stats_np[:, 0]
        per_sample_count = stats_np[:, 1]

        if jax.process_index() == 0:
            assert texts is not None
            score = eval_data._score_from_stats(
                per_sample_nll, per_sample_count, texts, reduce=reduce
            )
        else:
            score = (
                np.zeros(original_size, dtype=np.float32) if reduce == "none" else 0.0
            )
        score = multihost_utils.broadcast_one_to_all(jnp.asarray(score))
        score = np.asarray(score) if reduce == "none" else float(score)

        if not return_intermediates:
            return score

        xs_np = np.asarray(xs_full)
        masks_np = np.asarray(masks_full).astype(bool)
        intermediates = [
            (xs_np[i], masks_np[i], masks_np[i]) for i in range(original_size)
        ]
        return score, intermediates


class BitsPerByteEvaluation(PerplexityEvaluation):
    """Evaluation that computes dataset bits-per-byte (lower is better).

    Uses the same forward-pass statistics as PerplexityEvaluation, then
    converts natural-log token loss to bits per input byte:
    ``loss * (tokens / bytes) / ln(2)``.
    """

    def score(  # type: ignore[override]
        self,
        per_sample_nll: np.ndarray,
        per_sample_count: np.ndarray,
        per_sample_byte_count: np.ndarray,
    ) -> List[float]:
        """Per-sample bits-per-byte."""
        nll = np.asarray(per_sample_nll, dtype=np.float64)
        count = np.asarray(per_sample_count, dtype=np.float64)
        byte_count = np.maximum(
            np.asarray(per_sample_byte_count, dtype=np.float64), 1.0
        )
        loss = nll / np.maximum(count, 1.0)
        tokens_per_byte = count / byte_count
        return list((loss * tokens_per_byte / np.log(2.0)).astype(np.float32))

    def _score_from_stats(
        self,
        per_sample_nll: np.ndarray,
        per_sample_count: np.ndarray,
        texts: list[str],
        reduce: str = "mean",
    ) -> Any:
        """Byte-weighted aggregate BPB for mean/sum; per-sample BPB for none."""
        byte_count = np.asarray(
            [len(text.encode("utf-8")) for text in texts], dtype=np.float64
        )
        if reduce == "none":
            return np.asarray(
                self.score(per_sample_nll, per_sample_count, byte_count),
                dtype=np.float32,
            )

        nll = float(np.asarray(per_sample_nll, dtype=np.float64).sum())
        token_count = float(np.asarray(per_sample_count, dtype=np.float64).sum())
        byte_count_total = float(byte_count.sum())
        loss = nll / max(token_count, 1.0)
        tokens_per_byte = token_count / max(byte_count_total, 1.0)
        return float(loss * tokens_per_byte / np.log(2.0))


class PerplexityComparisonEvaluation(Evaluation):
    """Evaluation using perplexity comparison for multiple-choice tasks."""

    # See ``RolloutEvaluation`` — same compile-once-per-instance pattern.
    _chunk_jit: Optional[Any] = None
    _evaluator_ref: Optional[Any] = None

    def _chunk_step(
        self,
        state: Any,
        xs_chunk: Any,
        masks_chunk: Any,
        prefix_lens_chunk: Any,
    ) -> Any:
        inference = self._evaluator_ref
        assert inference is not None, (
            "_chunk_step called before __call__ set _evaluator_ref"
        )

        assert inference.spec.topology is not None
        fsdp = inference.spec.topology.shard.fsdp

        def reduce(_: Any, batch: Any) -> Any:
            params = state.params
            if fsdp:
                # Keep gathered weights inside this batch iteration.
                params, batch = jax.lax.optimization_barrier((params, batch))  # type: ignore[no-untyped-call]
            x_batch, mask_batch, prefix_len_batch = batch

            y_batch = jnp.roll(x_batch, -1, axis=-1)
            y_batch = y_batch.at[:, -1].set(0)

            seq_len = x_batch.shape[-1]
            num_real_tokens = jnp.sum(
                mask_batch.astype(jnp.int32), axis=-1, keepdims=True
            )
            content_start = seq_len - num_real_tokens

            seq_positions = jnp.arange(seq_len)[None, :]
            prefix_end = content_start + prefix_len_batch[:, None]
            is_padding_or_prefix = seq_positions < prefix_end
            y_batch = jnp.where(is_padding_or_prefix, -1, y_batch)

            with (
                jax.sharding.use_abstract_mesh(inference.mesh.abstract_mesh),
                nn.logical_axis_rules(inference.model.sharding._tp),
            ):
                logits, _, _ = inference.forward(
                    state,
                    params,
                    (x_batch, None, mask_batch),
                    None,
                    deterministic=True,
                )

            logits_f32 = logits.astype(jnp.float32)
            logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
            targets_flat = y_batch.reshape(-1)

            mask = targets_flat != -1
            targets_masked = jnp.where(mask, targets_flat, 0)

            log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
            token_losses = -jnp.take_along_axis(
                log_probs, targets_masked[:, None], axis=-1
            ).squeeze(-1)
            token_losses = jnp.where(mask, token_losses, 0.0)

            token_losses = token_losses.reshape(x_batch.shape[0], x_batch.shape[1])
            mask = mask.reshape(x_batch.shape[0], x_batch.shape[1])
            per_sample_loss = jnp.sum(token_losses, axis=-1) / jnp.maximum(
                jnp.sum(mask, axis=-1), 1.0
            )

            return None, per_sample_loss

        _, losses = jax.lax.scan(
            reduce, None, (xs_chunk, masks_chunk, prefix_lens_chunk)
        )
        return jnp.reshape(losses, (-1,))

    @abstractmethod
    def get(self, indx: int) -> Tuple[str, list[str], int]:
        """Get sample at index.

        Returns:
            (prefix, list_of_continuations, correct_index)
        """
        ...

    def score(self, correct_flags: List[float]) -> List[float]:
        """Per-sample correctness (1.0 / 0.0)."""
        return [float(c) for c in correct_flags]

    def __call__(
        self,
        inference: "InferenceJob[Any, M]",
        encoding: Any,
        reduce: str = "mean",
        return_intermediates: bool = False,
        chunk_size: int = 200,
        **kwargs: Any,
    ) -> Any:
        """Run evaluation."""
        eval_data = self
        batch_unit = inference.replicas * inference.per_device_batch_size
        indices = _select_indices(inference, len(eval_data))
        n_samples = len(indices)

        # Gather all data on main process
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        if jax.process_index() == 0:
            all_data = [eval_data.get(i) for i in indices]

            # Flatten: create (prefix+continuation, sample_idx, continuation_idx, prefix_len)
            flattened_inputs = []
            prefix_lengths = []
            metadata = []  # (sample_idx, continuation_idx, num_continuations)

            for sample_idx, (prefix, continuations, correct_idx) in enumerate(all_data):
                prefix_encoded = encoding.encode(prefix)
                prefix_len = len(prefix_encoded)

                for cont_idx, continuation in enumerate(continuations):
                    full_text = prefix + continuation
                    flattened_inputs.append(full_text)
                    prefix_lengths.append(prefix_len)
                    metadata.append((sample_idx, cont_idx, len(continuations)))

            original_flat_size = len(flattened_inputs)
            _, (flattened_inputs, prefix_lengths, metadata) = _pad_eval_inputs(
                batch_unit, flattened_inputs, prefix_lengths, metadata
            )

            # Encode all inputs
            encoded_inputs = encoding.encode_batch(
                flattened_inputs, allowed_special="all"
            )
            encoded_inputs = [seq[: inference.block_size] for seq in encoded_inputs]
            xs, masks = inference.pad(encoded_inputs)
            prefix_lengths_array = np.asarray(prefix_lengths, dtype=np.int32)
            metadata_array = np.asarray(metadata, dtype=np.int32)
            correct_indices_array = np.asarray([d[2] for d in all_data], dtype=np.int32)
            original_flat_size_array = np.asarray(original_flat_size, dtype=np.int32)
        else:
            xs, masks = None, None
            (
                prefix_lengths_array,
                metadata_array,
                correct_indices_array,
                original_flat_size_array,
            ) = (
                None,
                None,
                None,
                None,
            )

        # Broadcast to all hosts
        xs = multihost_utils.broadcast_one_to_all(xs)
        masks = multihost_utils.broadcast_one_to_all(masks)
        prefix_lengths_array = multihost_utils.broadcast_one_to_all(
            prefix_lengths_array
        )
        metadata_array = multihost_utils.broadcast_one_to_all(metadata_array)
        correct_indices_array = multihost_utils.broadcast_one_to_all(
            correct_indices_array
        )
        original_flat_size_array = multihost_utils.broadcast_one_to_all(
            original_flat_size_array
        )
        multihost_utils.sync_global_devices("eval_gather_all:post")

        xs_full = xs
        masks_full = masks

        # Divide across processes
        pieces_xs = np.array_split(xs, jax.process_count(), axis=0)
        pieces_masks = np.array_split(masks, jax.process_count(), axis=0)
        pieces_prefix_lens = np.array_split(
            prefix_lengths_array, jax.process_count(), axis=0
        )
        pieces_metadata = np.array_split(metadata_array, jax.process_count(), axis=0)

        xs = pieces_xs[jax.process_index()]
        masks = pieces_masks[jax.process_index()]
        prefix_lens_local = pieces_prefix_lens[jax.process_index()]
        metadata_local = pieces_metadata[jax.process_index()]

        # Reshape into (accumulate_steps, batch_size, T)
        batch_size = inference.local_replicas * inference.per_device_batch_size
        xs = xs.reshape(-1, batch_size, xs.shape[-1])
        masks = masks.reshape(-1, batch_size, masks.shape[-1])
        prefix_lens_local = prefix_lens_local.reshape(-1, batch_size)

        # Create global arrays
        data_pspec = P(None, Axis.BATCH, None)  # type: ignore[no-untyped-call]
        prefix_lens_pspec = P(None, Axis.BATCH)  # type: ignore[no-untyped-call]

        xs = multihost_utils.host_local_array_to_global_array(
            xs, inference.mesh, data_pspec
        )
        masks = multihost_utils.host_local_array_to_global_array(
            masks, inference.mesh, data_pspec
        )
        prefix_lens_local = multihost_utils.host_local_array_to_global_array(
            prefix_lens_local, inference.mesh, prefix_lens_pspec
        )

        # Cached chunk JIT — see RolloutEvaluation for rationale.
        if self._chunk_jit is None:
            self._evaluator_ref = inference
            data_sharding = NamedSharding(inference.mesh, data_pspec)
            prefix_lens_sharding = NamedSharding(inference.mesh, prefix_lens_pspec)
            self._chunk_jit = jax.jit(
                self._chunk_step,
                in_shardings=(
                    inference.state_sharding,
                    data_sharding,
                    data_sharding,
                    prefix_lens_sharding,
                ),
                out_shardings=None,
            )

        # Process in chunks with progress logging
        num_batches = xs.shape[0]
        all_losses = []

        if jax.process_index() == 0:
            logger.debug(
                "EVAL | {} | samples={} flat={} seq={} batches={}",
                eval_data.name,
                n_samples,
                int(original_flat_size_array),
                xs.shape[-1],
                num_batches,
            )

        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            xs_chunk = xs[chunk_start:chunk_end]
            masks_chunk = masks[chunk_start:chunk_end]
            prefix_lens_chunk = prefix_lens_local[chunk_start:chunk_end]

            if chunk_start == 0:
                logger.debug(
                    "EVAL | {} | tracing+compiling first chunk", eval_data.name
                )
            if jax.process_index() == 0 and num_batches > chunk_size:
                logger.debug(
                    "EVAL | {} | chunk {}/{} ({:.0f}%)",
                    eval_data.name,
                    chunk_end,
                    num_batches,
                    (chunk_end / num_batches) * 100,
                )

            chunk_losses = self._chunk_jit(
                inference.state, xs_chunk, masks_chunk, prefix_lens_chunk
            )
            all_losses.append(chunk_losses)

        # Concatenate all chunk results
        losses = jnp.concatenate(all_losses, axis=0)

        # Gather results across all hosts
        multihost_utils.sync_global_devices("eval_gather_all:pre")
        losses = multihost_utils.process_allgather(losses)
        metadata_gathered = multihost_utils.process_allgather(metadata_local)
        multihost_utils.sync_global_devices("eval_gather_all:post")

        # Flatten
        losses = jnp.reshape(losses, (-1,))
        metadata_gathered = jnp.reshape(metadata_gathered, (-1, 3))
        original_flat_size = int(jax.device_get(original_flat_size_array))
        losses = losses[:original_flat_size]
        metadata_gathered = metadata_gathered[:original_flat_size]

        # Convert to numpy
        losses = jax.device_get(losses)
        metadata_gathered = jax.device_get(metadata_gathered)
        correct_indices_array = jax.device_get(correct_indices_array)

        # Score on process 0 only, then broadcast
        if jax.process_index() == 0:
            # Group losses by sample_idx
            sample_losses = defaultdict(list)
            sample_num_continuations = {}

            for i, (sample_idx, cont_idx, num_conts) in enumerate(metadata_gathered):
                sample_idx = int(sample_idx)
                sample_losses[sample_idx].append(losses[i])
                sample_num_continuations[sample_idx] = int(num_conts)

            # Per-sample correctness
            correct_flags: List[float] = []
            for sample_idx in sorted(sample_losses.keys()):
                expected_conts = sample_num_continuations[sample_idx]
                actual_conts = len(sample_losses[sample_idx])

                if actual_conts != expected_conts:
                    continue

                losses_for_sample = jnp.array(sample_losses[sample_idx])
                pred = int(jnp.argmin(losses_for_sample))
                correct_idx = int(correct_indices_array[sample_idx])

                correct_flags.append(1.0 if pred == correct_idx else 0.0)

            score = eval_data._score(correct_flags, reduce=reduce)
        else:
            score = np.zeros(n_samples, dtype=np.float32) if reduce == "none" else 0.0
        score = multihost_utils.broadcast_one_to_all(jnp.asarray(score))
        score = np.asarray(score) if reduce == "none" else float(score)

        if not return_intermediates:
            return score

        xs_np = np.asarray(xs_full)
        masks_np = np.asarray(masks_full).astype(bool)
        intermediates = [(xs_np[i], masks_np[i]) for i in range(original_flat_size)]
        return score, intermediates


@dataclass
class EvaluatorConfig:
    """Configuration for Evaluator."""

    length: int = field("eval/length", default=-1)


class Evaluator(InferenceJob[EvaluatorConfig, M], Generic[M]):
    """InferenceJob that runs evaluations and saves results.

    Created from a trainer, holds a list of evaluations, and runs them when
    ``run()`` is called. Standalone inference subclasses can use the evaluation
    components directly.

    Example:
        evaluator = Evaluator.from_trainer(trainer)
        evaluator()  # Runs evaluations and saves results
    """

    MODEL: type[M] = Module  # type: ignore[assignment]
    EVALUATION: List[type[Evaluation]] = []
    evaluations: List[Evaluation]
    encoding: Tokenizer
    length: int
    random: random.Random

    def __init__(self, spec: ExecutionSpec, base: Node | None = None) -> None:
        super().__init__(spec, base=base)
        self.encoding = get_tokenizer()
        self.random = random.Random(0xC0FFEE)
        self.length = configure(EvaluatorConfig).length
        self.evaluations = [evaluation() for evaluation in self.EVALUATION]

    @classmethod
    def config(
        cls, evaluations: Optional[List[type[Evaluation]]] = None
    ) -> List[type[Any]]:
        """Return evaluator and child-evaluation configuration schemas.

        Args:
            evaluations: Evaluation classes to aggregate. Defaults to the
                evaluator's static ``EVALUATION`` declaration.

        Returns:
            Schemas required to construct the evaluator and its children.
        """

        children = cls.EVALUATION if evaluations is None else evaluations
        return [
            *([] if evaluations is not None else super().config()),
            EvaluatorConfig,
            TokenizerConfig,
            *(schema for evaluation in children for schema in evaluation.config()),
        ]

    @classmethod
    def from_trainer(
        cls,
        trainer: "BaseTrainer[Any, Any]",
    ) -> "Evaluator[M]":
        """Create Evaluator from trainer.

        Args:
            trainer: BaseTrainer instance to get inference state from

        Returns:
            Evaluator instance ready to run evaluations
        """
        evaluator = super().from_trainer(trainer)
        evaluator.encoding = get_tokenizer()
        evaluator.random = random.Random(0xC0FFEE)
        evaluator.length = configure(EvaluatorConfig).length
        evaluator.evaluations = [evaluation() for evaluation in trainer.EVALUATION]

        return evaluator

    def rollout(
        self,
        inputs: List[Union[str, ChatTemplate, jax.Array, List[int]]],
        encoding: Optional[Tokenizer] = None,
        max_new_tokens: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        chunk_size: int = 200,
        return_type: Literal[
            "decoded",
            "indices",
            "output_decoded",
            "output_indices",
            "raw_indices",
        ] = "decoded",
    ) -> Union[List[Union[str, ChatTemplate]], List[str], List[List[int]]]:
        return super().rollout(
            inputs,
            encoding if encoding is not None else self.encoding,
            max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            temperature=temperature,
            top_p=top_p,
            chunk_size=chunk_size,
            return_type=return_type,
        )

    def evaluate(
        self,
        reduce: str = "mean",
        return_intermediates: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run all evaluations.

        Args:
            reduce: passed through to each evaluation. "mean"/"sum" → float per
                evaluation; "none" → np.ndarray of per-sample scores.
            return_intermediates: when True, also return the per-evaluation list
                of (x, mask) rollouts (one inner list per evaluation).
            **kwargs: forwarded to each evaluation's __call__ (e.g. temperature,
                top_p, chunk_size).
        """
        results: dict[str, Any] = {}
        all_intermediates: List[List[Tuple[np.ndarray, np.ndarray]]] = []

        for evaluation in self.evaluations:
            logger.debug("EVAL | Running {}", evaluation.name)
            if return_intermediates:
                score, intermediates = evaluation(
                    self,
                    self.encoding,
                    reduce=reduce,
                    return_intermediates=True,
                    **kwargs,
                )
                all_intermediates.append(intermediates)
            else:
                score = evaluation(
                    self,
                    self.encoding,
                    reduce=reduce,
                    return_intermediates=False,
                    **kwargs,
                )
            results[evaluation.name] = score
            logger.debug("EVAL | {} done", evaluation.name)

        if return_intermediates:
            return results, all_intermediates
        return results

    def run(self) -> None:
        """Run all evaluations and save results to disk."""

        logger.info("EVAL | Starting evaluations for job {}", self.spec.name)
        results = self.evaluate()

        # Log summary
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        for key, value in results.items():
            logger.info("  {}: {:.4f}", key, value)
        logger.info("=" * 60)
        self.log(results)

        # Save results to JSON (only on main process)
        with self.spec.result("results.json", main_process_only=True) as f:
            if f is not None:
                json.dump(
                    {
                        "job": self.spec.name,
                        "project": self.spec.project,
                        "group": self.spec.group,
                        "results": {k: float(v) for k, v in results.items()},
                    },
                    f,
                    indent=2,
                )
                logger.info("EVAL | Results saved to {}", Path(f.name))
