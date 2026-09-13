"""
Thoughtbubbles: A GPT variant with token forking capability.
Extends GPT with branching/forking tokens based on learned scores.
"""

import math

import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Optional, Tuple, List, Any, Type, ClassVar

from theseus.model.models.base import GPT
from theseus.model.block.forking import ThoughtBlock, ForkingBlock
from theseus.model.layers import LayerNorm
from theseus.model.axes import Axes

from theseus.base.axis import ShardingPlan
from theseus.config import field, configure


class Thoughtbubbles(GPT):
    """
    Thoughtbubbles model with token forking support.
    Extends GPT with the ability to fork tokens during processing.
    """

    FORK_BLOCK: ClassVar[type[ForkingBlock]] = ForkingBlock

    max_block_size: int = field("architecture/max_block_size", default=1024)
    fork: List[int] = field(
        "architecture/fork",
        default_factory=lambda: [3, 6, 9],
    )

    @property
    def sharding(self) -> ShardingPlan:
        base_sharding = super().sharding
        return ShardingPlan(
            tp=(*base_sharding.tp, (Axes.N_FORK.value, None)),
            zero=base_sharding.zero,
        )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [ThoughtBlock, cls.FORK_BLOCK, LayerNorm]

    def flops(self, seq: int) -> float:
        # Sections 2.3 and 2.5 of Thoughtbubbles: fork up to the budget,
        # then average residuals back to the original length before the head.
        # https://arxiv.org/html/2510.00219v2
        length = seq
        budget = math.ceil(seq * self.max_block_size / self.block_size)
        total = 6.0 * seq * self.n_embd * self.vocab_size
        for block in self.blocks:
            if isinstance(block, ForkingBlock):
                total += block.flops(length, input_seq_len=seq)
                length = min(2 * length, budget)
            else:
                total += block.flops(length)
        return total

    def setup(self) -> None:
        assert self.vocab_size is not None
        assert self.block_size is not None

        # Token embedding table (no positional - using RoPE in attention)
        self.wte: jax.Array = self.param(
            "wte",
            nn.with_logical_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            jnp.float32,
        )  # type: ignore

        self.drop = nn.Dropout(rate=self.dropout)

        # Create blocks: ForkingBlock for layers in fork list, ThoughtBlock otherwise
        fork_set = set(self.fork)
        self.blocks = [
            configure(self.FORK_BLOCK) if i in fork_set else configure(ThoughtBlock)
            for i in range(self.n_layers)
        ]

        self.ln_f = configure(LayerNorm)

    def embed(
        self, idx: jax.Array, deterministic: bool = False, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute token embeddings and initialize forking state.
        Overrides GPT.embed to return forking state.
        """
        b, t = idx.shape

        # Token embeddings (no positional - RoPE handles position)
        x = jnp.take(self.wte, idx, axis=0).astype(jnp.bfloat16)
        x = self.drop(x, deterministic=deterministic)

        # Initialize forking state
        cumulative_scores = jnp.zeros((b, t), dtype=x.dtype)
        token_index = jnp.tile(jnp.arange(t), (b, 1))

        return x, cumulative_scores, token_index

    def decode(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Process through transformer blocks with forking.
        Overrides GPT.decode to handle forking state.
        """
        cumulative_scores: jax.Array = kwargs["cumulative_scores"]
        token_index: jax.Array = kwargs["token_index"]
        input_seq_len: Optional[int] = kwargs.get("input_seq_len")

        for block in self.blocks:
            result = block(
                x,
                cumulative_scores=cumulative_scores,
                token_index=token_index,
                padding_mask=padding_mask,
                deterministic=deterministic,
                input_seq_len=input_seq_len,
            )
            # Block returns tuple (x, scores, indices)
            x, cumulative_scores, token_index = result

        return x, cumulative_scores, token_index

    def residual_average(
        self,
        input_size: int,
        residuals: jax.Array,
        cumulative_scores: jax.Array,
        token_index: jax.Array,
    ) -> jax.Array:
        """Average residuals weighted by cumulative scores."""
        scaled_residuals = residuals * jnp.exp(cumulative_scores)[:, :, None]

        summed_residuals = jnp.zeros(
            (residuals.shape[0], input_size, self.n_embd),
            dtype=residuals.dtype,
        )
        B, _ = token_index.shape
        b = jnp.arange(B)[:, None]

        return summed_residuals.at[b, token_index].add(scaled_residuals)

    def unembed(self, x: jax.Array) -> jax.Array:
        """
        Standard unembed - used when no forking state provided.
        """
        x = self.ln_f(x)
        return jnp.einsum("bth,vh->btv", x, self.wte.astype(jnp.bfloat16))

    def unembed_forked(
        self,
        x: jax.Array,
        cumulative_scores: jax.Array,
        token_index: jax.Array,
        input_seq_len: int,
    ) -> jax.Array:
        """
        Compute output logits for forked tokens using residual averaging.
        """
        if x.shape[1] == input_seq_len:
            # No forking occurred, standard path
            return self.unembed(x)

        # Handle forked tokens with residual averaging
        averaged = self.residual_average(
            input_seq_len, x, cumulative_scores, token_index
        )
        return jnp.einsum(
            "blh,vh->blv", self.ln_f(averaged), self.wte.astype(jnp.bfloat16)
        )

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        """
        Forward pass through Thoughtbubbles.

        Args:
            idx: Input token indices of shape (B, T).
            targets: Target token indices of shape (B, T). Use -1 to ignore.
            padding_mask: Boolean tensor of shape (B, T). True=valid, False=padding.
            deterministic: If False, applies dropout.

        Returns:
            logits: Output logits of shape (B, T, vocab_size).
            loss: Cross-entropy loss if targets provided, else None.
        """
        b, t = idx.shape
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.block_size}"
        )

        # Embed tokens and initialize forking state
        x, cumulative_scores, token_index = self.embed(idx, deterministic, **kwargs)
        self.sow("plots", "embeddings", x)

        # Process through transformer blocks
        x, cumulative_scores, token_index = self.decode(
            x,
            padding_mask=padding_mask,
            deterministic=deterministic,
            cumulative_scores=cumulative_scores,
            token_index=token_index,
            input_seq_len=t,
            **kwargs,
        )

        # Compute logits
        logits = self.unembed_forked(x, cumulative_scores, token_index, t)

        # Compute loss if targets provided
        if targets is not None:
            loss_val = self.loss(logits, targets)
        else:
            loss_val = None

        return logits, loss_val
