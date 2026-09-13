"""
Transformer blocks with forking support for Thoughtbubbles.
Extends Block with cumulative score weighting and forking capability.
"""

import math
from typing import Tuple, Any, List, Type, Optional, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax

from theseus.model.block.block import Block
from theseus.model.attention.forking import ForkingAttention
from theseus.model.layers.layernorm import LayerNorm
from theseus.model.layers.mlp import MLP
from theseus.model.axes import Axes

from theseus.config import configure, field

# Return type for forking blocks
ForkingOutput = Tuple[jax.Array, jax.Array, jax.Array]


class ThoughtBlock(Block):
    """
    Transformer block for Thoughtbubbles with cumulative score support.
    Extends Block by weighting attention and MLP outputs by cumulative scores.
    """

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [LayerNorm, ForkingAttention, MLP]

    def setup(self) -> None:
        self.ln_1 = configure(LayerNorm)
        self.attn = configure(ForkingAttention)
        self.ln_2 = configure(LayerNorm)
        self.mlp = configure(MLP)

    def __call__(  # type: ignore[override]
        self,
        x: jax.Array,
        **kwargs: Any,
    ) -> Union[jax.Array, ForkingOutput]:
        """
        Forward pass with cumulative score weighting.

        Args:
            x: Input tensor of shape (B, T, C).
            **kwargs: Must include cumulative_scores, token_index for forking mode.

        Returns:
            If forking: Tuple of (output, cumulative_scores, token_index).
            If not forking: Just the output tensor.
        """
        cumulative_scores: Optional[jax.Array] = kwargs.get("cumulative_scores")
        token_index: Optional[jax.Array] = kwargs.get("token_index")
        padding_mask: Optional[jax.Array] = kwargs.get("padding_mask")
        deterministic: bool = kwargs.get("deterministic", False)

        if cumulative_scores is None or token_index is None:
            # Fall back to standard Block behavior
            return super().__call__(x, **kwargs)

        exponentiated_scores = jnp.exp(cumulative_scores)

        # Attention with score weighting
        attn_out = self.attn(
            self.ln_1(x),
            padding_mask=padding_mask,
            cumulative_scores=cumulative_scores,
            token_index=token_index,
            deterministic=deterministic,
        )
        x = x + jnp.einsum("bl,blh->blh", exponentiated_scores, attn_out)

        # MLP with score weighting
        mlp_out = self.mlp(self.ln_2(x), deterministic=deterministic)
        x = x + jnp.einsum("bl,blh->blh", exponentiated_scores, mlp_out)

        return x, cumulative_scores, token_index


class ForkingBlock(ThoughtBlock):
    """
    Transformer block with forking capability.
    Extends ThoughtBlock by adding token forking based on learned scores.
    """

    n_embd: int = field("architecture/n_embd", default=2048)
    block_size: int = field("architecture/block_size", default=512)
    max_block_size: int = field("architecture/max_block_size", default=1024)
    bias: bool = field("architecture/bias", default=True)

    def flops(self, seq: int, *, input_seq_len: Optional[int] = None) -> float:
        # Two fork/keep scores per input residual; transformer sees retained streams.
        budget = (
            self.max_block_size
            if input_seq_len is None
            else math.ceil(input_seq_len * self.max_block_size / self.block_size)
        )
        retained = min(2 * seq, budget)
        return 12.0 * seq * self.n_embd + super().flops(retained)

    def setup(self) -> None:
        super().setup()

        self.forking_ln = configure(LayerNorm)

        # Forking score projection (outputs 2: fork score and keep score)
        self.forking_score = nn.Dense(
            2,
            use_bias=False,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(stddev=0.02),
                (Axes.N_EMBD.value, Axes.N_FORK.value),
            ),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16,
        )

    @staticmethod
    def clipped_logsigmoid(x: jax.Array, min_val: float = -20.0) -> jax.Array:
        """Compute log-sigmoid with clipping for numerical stability."""
        logsigmoid = -jax.nn.softplus(-x)
        return jnp.clip(logsigmoid, a_min=min_val)

    # self.clipped_logsigmoid = clipped_logsigmoid

    def fork(
        self,
        x: jax.Array,
        cumulative_scores: jax.Array,
        token_index: jax.Array,
        padding_mask: Optional[jax.Array],
        input_seq_len: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Top-k forking: doubles tokens then selects top-k to keep.

        Args:
            x: Input tensor of shape (B, T, C).
            cumulative_scores: Log-probability scores of shape (B, T).
            token_index: Original token indices of shape (B, T).
            padding_mask: Boolean tensor of shape (B, T). True for valid.
            input_seq_len: Original input sequence length for ratio calc.

        Returns:
            Tuple of (forked_x, new_scores, new_token_indices).
        """
        batch_size = cumulative_scores.shape[0]

        # Compute current layer's forking scores
        curr_layer_forking_score = self.forking_score(self.forking_ln(x))

        # Update cumulative scores (in log space)
        forking_scores_cum = (
            self.clipped_logsigmoid(curr_layer_forking_score)
            + cumulative_scores[:, :, None]
        ).reshape(batch_size, -1)  # (batch_size, 2T)
        forking_scores_cum_for_topk = forking_scores_cum

        # Copy token index twice (for original and fork)
        forked_token_index = token_index.repeat(2, axis=-1)

        # Mark rightmost token of each original token with +inf (always keep)
        is_rightmost = jnp.concatenate(
            (
                forked_token_index[:, :-1] != forked_token_index[:, 1:],
                jnp.ones((batch_size, 1), dtype=jnp.bool_),
            ),
            axis=-1,
        )
        forking_scores_cum_for_topk = jnp.where(
            is_rightmost, float("+inf"), forking_scores_cum_for_topk
        )

        if padding_mask is not None:
            new_padding_mask = jnp.take_along_axis(
                padding_mask, forked_token_index, axis=-1
            )

            forking_scores_cum = jnp.where(
                new_padding_mask, forking_scores_cum, float("-inf")
            )
            forking_scores_cum_for_topk = jnp.where(
                new_padding_mask, forking_scores_cum_for_topk, float("-inf")
            )

            # Soft kill to maintain ratio with padding
            keep_p = self.max_block_size / self.block_size
            n_per_row = padding_mask.sum(axis=-1)
            k_per_row = jnp.floor(keep_p * n_per_row).astype(jnp.int32)

            score_order = jnp.argsort(forking_scores_cum_for_topk, axis=-1)[:, ::-1]
            rank_in_row = jnp.argsort(score_order, axis=-1)
            scores_to_kill = ~(rank_in_row < k_per_row[:, None])
            forking_scores_cum_for_topk = jnp.where(
                scores_to_kill, -jnp.inf, forking_scores_cum_for_topk
            )
            forking_scores_cum = jnp.where(
                scores_to_kill, float("-inf"), forking_scores_cum
            )

        # Perform top-k selection
        if input_seq_len is not None:
            k = int(math.ceil(input_seq_len * (self.max_block_size / self.block_size)))
        else:
            k = self.max_block_size

        k = min(k, forking_scores_cum_for_topk.shape[-1])
        _, top_k_indices = lax.top_k(forking_scores_cum_for_topk, k)
        top_k_indices = jnp.sort(top_k_indices, axis=-1)

        # Gather based on indices that survived
        orig_indices = top_k_indices // 2
        is_fork = (top_k_indices % 2) == 0

        # Gather x values
        batch_indices = jnp.arange(batch_size)[:, None]
        x_to_consider = x[batch_indices, orig_indices, :]

        x_to_consider = jnp.where(
            is_fork[..., None],
            self.fork_token(
                x,
                orig_indices,
                cumulative_scores=cumulative_scores,
                token_index=token_index,
                padding_mask=padding_mask,
            ),
            x_to_consider,
        )

        # Gather cumulative scores and token indices
        new_cumulative_scores = jnp.take_along_axis(
            forking_scores_cum, top_k_indices, axis=-1
        )
        new_token_indices = jnp.take_along_axis(token_index, orig_indices, axis=-1)
        self.sow("plots", "new_cumulative_scores", new_cumulative_scores)
        self.sow("plots", "embeddings", x_to_consider)

        # x = x_to_consider
        # cumulative_scores = new_cumulative_scores
        # token_index = new_token_indices

        return x_to_consider, new_cumulative_scores, new_token_indices

    @nn.compact
    def fork_token(
        self,
        x: jax.Array,
        parent_indices: jax.Array,
        *,
        cumulative_scores: jax.Array,
        token_index: jax.Array,
        padding_mask: Optional[jax.Array],
    ) -> jax.Array:
        """Return candidate replacements at parent_indices from the full source x.

        Source scores, original positions, and padding support contextual forks.
        The caller applies replacements only to new children, keeping selection
        fixed-size for JAX and retaining keep children unchanged.
        """
        embedding = self.param(
            "fork_embedding",
            nn.with_logical_partitioning(
                lambda rng, shape: (
                    jax.random.normal(rng, shape) * (1 / jnp.sqrt(self.n_embd))
                ),
                (Axes.N_EMBD.value,),
            ),
            (self.n_embd,),
        )
        return x[jnp.arange(x.shape[0])[:, None], parent_indices] + jnp.asarray(
            embedding, dtype=jnp.bfloat16
        )

    def __call__(  # type: ignore[override]
        self,
        x: jax.Array,
        **kwargs: Any,
    ) -> Union[jax.Array, ForkingOutput]:
        """
        Forward pass: fork first, then apply standard block operations.

        Args:
            x: Input tensor of shape (B, T, C).
            **kwargs: cumulative_scores, token_index, padding_mask, deterministic,
                      input_seq_len.

        Returns:
            Tuple of (output, cumulative_scores, token_index).
        """
        cumulative_scores: Optional[jax.Array] = kwargs.get("cumulative_scores")
        token_index: Optional[jax.Array] = kwargs.get("token_index")
        padding_mask: Optional[jax.Array] = kwargs.get("padding_mask")
        input_seq_len: Optional[int] = kwargs.get("input_seq_len")

        if cumulative_scores is None or token_index is None:
            # Initialize if not provided
            b, t = x.shape[:2]
            cumulative_scores = jnp.zeros((b, t), dtype=x.dtype)
            token_index = jnp.tile(jnp.arange(t), (b, 1))

        # Fork first
        x, cumulative_scores, token_index = self.fork(
            x, cumulative_scores, token_index, padding_mask, input_seq_len=input_seq_len
        )

        # Update kwargs with new forking state
        kwargs["cumulative_scores"] = cumulative_scores
        kwargs["token_index"] = token_index

        # Then apply ThoughtBlock operations (score-weighted attention/MLP)
        return super().__call__(
            x,
            **kwargs,
        )
