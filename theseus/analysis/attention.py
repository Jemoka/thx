"""Masked softmax attention heatmap for one example and query head.

Combine this unregistered template with your trainer (analysis first)::

    @analysis(
        "gpt/analyze/attention"
    )
    class GPTAttention(
        AttentionHeatmapAnalysis,
        GPTPretrain,
    ):
        pass

The trainer supplies MODEL, DATASET, and restoration/surgery. CONFIG defaults to
AttentionAnalysisConfig, extending BaseTrainerConfig. For a custom trainer config,
inherit both schemas in a dataclass and set CONFIG on the concrete analysis. This adds
analysis/layer, sample, head, max_tokens to its configuration. Layer indices
follow discovery order. Execute with base=checkpoint through the job lifecycle.

This analyzes full-sequence batches, not KV-cache decoding. All hosts compute;
only the bounded attention matrix is gathered, then host zero creates a CPU plot.
Softmax uses all keys before cropping: displayed rows can sum to less than one
when undisplayed keys receive attention. An entirely masked row is shown as zero.
"""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import flax.linen as nn
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns

from theseus.config import field
from theseus.model.attention.base import SelfAttention
from theseus.model.debug import DebugInputs
from theseus.training.base import BaseTrainerConfig, M

from .base import AnalysisBase
from .plots import BLUE


@dataclass
class AttentionAnalysisConfig(BaseTrainerConfig):
    layer: int = field("analysis/layer", default=0)
    sample: int = field("analysis/sample", default=0)
    head: int = field("analysis/head", default=0)
    max_tokens: int = field("analysis/max_tokens", default=128)


C = TypeVar("C", bound=AttentionAnalysisConfig)


class AttentionHeatmapAnalysis(AnalysisBase[C, M], Generic[C, M]):
    """Unregistered analysis template; put it before your trainer in the bases."""

    TARGET = SelfAttention
    NAME = "attention"
    CONFIG: type[Any] = AttentionAnalysisConfig

    def select(self, paths: list[str]) -> str:
        index = self.args.layer
        if not 0 <= index < len(paths):
            raise ValueError(f"analysis/layer must be between 0 and {len(paths) - 1}")
        return paths[index]

    def analyze(
        self,
        layer: nn.Module | list[nn.Module],
        inputs: DebugInputs | list[DebugInputs],
    ) -> jax.Array:
        assert not isinstance(layer, list) and isinstance(inputs, DebugInputs)
        x = inputs.x
        if (
            not 0 <= self.args.sample < x.shape[0]
            or not 0 <= self.args.head < layer.n_head
        ):
            raise ValueError("analysis/sample or analysis/head is out of range")
        if self.args.max_tokens <= 0:
            raise ValueError("analysis/max_tokens must be positive")
        if layer.has_variable("cache", "cache_index"):
            raise ValueError(
                "QK analysis requires a full-sequence trace without KV cache"
            )

        if getattr(layer, "q_output_gate", False):
            q, k, v, _ = layer._project_with_gate(x)
        else:
            q, k, v = layer.project(x)
        kwargs = dict(inputs.get("kwargs", {}))
        if "positions" not in kwargs:
            kwargs["positions"] = layer._positions_from_padding(
                x.shape[1], inputs.get("padding_mask")
            )
        q, k, v = layer.preprocess_qkv(q, k, v, **kwargs)
        # Consecutive groups of query heads share a KV head in GQA.
        kv_head = self.args.head // (q.shape[2] // k.shape[2])
        query = q[self.args.sample, : self.args.max_tokens, self.args.head].astype(
            jnp.float32
        )
        key = k[self.args.sample, :, kv_head].astype(jnp.float32)
        scores = query @ key.T / jnp.sqrt(query.shape[-1])
        mask = layer.build_mask(x.shape[1], inputs.get("padding_mask"), **kwargs)
        if mask is None:
            mask = (
                jnp.arange(key.shape[0])[None, :] <= jnp.arange(query.shape[0])[:, None]
            )
        else:
            mask = mask[
                self.args.sample if mask.shape[0] > 1 else 0,
                self.args.head if mask.shape[1] > 1 else 0,
                : query.shape[0],
                :,
            ]
        weights = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1)
        weights = jnp.where(mask, weights, 0)[:, : self.args.max_tokens]
        return weights

    def plot(self, results: np.ndarray) -> Figure:
        return self.heatmap(
            results,
            vmin=0,
            vmax=1,
            cmap=sns.light_palette(BLUE, as_cmap=True),
            title="Attention",
            xlabel="Key token",
            ylabel="Query token",
            cbar_kws={"label": "Weight"},
        )
