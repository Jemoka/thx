import jax
import flax.linen as nn
import jax.numpy as jnp

from typing import Optional, Tuple, List, Any, Type

from theseus.model.block import Block
from theseus.model.layers import LayerNorm
from theseus.model.axes import Axes
from theseus.model.module import Module

from theseus.base.axis import ShardingPlan
from theseus.config import field, configure
from theseus.base.axis import Axis


class GPT(Module):
    n_layers: int = field("architecture/n_layers", default=32)
    n_embd: int = field("architecture/n_embd", default=2048)
    rope: bool = field("architecture/rope", default=True)
    block_size: int = field("architecture/block_size", default=512)
    dropout: float = field("architecture/dropout", default=0.0)
    vocab_size: int = field("architecture/vocab_size", default=100288)
    instrument_residual: bool = field(
        "architecture/instrumentation/residual", default=False
    )

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(
            tp=[
                (Axes.VOCAB.value, None),
                (Axes.BLOCK_SIZE.value, None),
                (Axes.N_EMBD.value, None),
                (Axes.N_EMBD_FF.value, Axis.SHARD),
                (Axes.N_EMBD_OUT.value, Axis.SHARD),
                (Axes.N_ATTN.value, Axis.SHARD),
            ],
            zero=[
                (Axes.VOCAB.value, Axis.BATCH),
                (Axes.BLOCK_SIZE.value, Axis.BATCH),
                (Axes.N_EMBD_FF.value, Axis.BATCH),
                (Axes.N_EMBD_OUT.value, Axis.BATCH),
                (Axes.N_ATTN.value, Axis.BATCH),
            ],
        )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [Block, LayerNorm]

    def flops(self, seq: int) -> float:
        return float(
            sum(block.flops(seq) for block in self.blocks)
            + 6 * seq * self.n_embd * self.vocab_size
        )

    def setup(self) -> None:
        assert self.vocab_size is not None
        assert self.block_size is not None

        # Token embedding table
        self.wte: jax.Array = self.param(
            "wte",
            nn.with_logical_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            self._param_dtype,
        )  # type: ignore

        # Positional embedding table (only when not using RoPE)
        if not self.rope:
            self.wpe: jax.Array = self.param(
                "wpe",
                nn.with_logical_partitioning(
                    nn.initializers.normal(stddev=0.02),
                    (Axes.BLOCK_SIZE.value, Axes.N_EMBD.value),
                ),
                (self.block_size, self.n_embd),
                self._param_dtype,
            )  # type: ignore

        self.drop = nn.Dropout(rate=self.dropout)
        self.blocks = [configure(Block) for _ in range(self.n_layers)]
        self.ln_f = configure(LayerNorm)

    def embed(self, idx: jax.Array, deterministic: bool = False, **kwargs: Any) -> Any:
        """Compute token and positional embeddings given inputs."""

        _, t = idx.shape

        # Token embeddings
        x = jnp.take(self.wte, idx, axis=0).astype(self._activation_dtype)

        # Positional embeddings (only when not using RoPE)
        if not self.rope:
            pos = jnp.arange(0, t)
            x = x + jnp.take(self.wpe, pos, axis=0).astype(self._activation_dtype)

        x = self.drop(x, deterministic=deterministic)

        return x

    def decode(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Compute decoded residual channels given embeddings."""

        for layer, block in enumerate(self.blocks):
            residual = x
            x = block(x, padding_mask=padding_mask, deterministic=deterministic)
            self.measure(x, residual, layer, padding_mask)
        return x

    def measure(
        self,
        x: jax.Array,
        residual: jax.Array,
        depth: int,
        padding_mask: Optional[jax.Array],
    ) -> None:
        """Measure the residual update after one effective layer."""

        ####### residual diagnostics #######

        if not self.instrument_residual:
            return

        update = x - residual
        residual_f32 = residual.astype(jnp.float32)
        update_f32 = update.astype(jnp.float32)
        residual_rms = jnp.sqrt(jnp.mean(residual_f32**2, axis=-1))
        update_rms = jnp.sqrt(jnp.mean(update_f32**2, axis=-1))
        update_residual_cosine = jnp.mean(
            residual_f32 * update_f32, axis=-1
        ) / jnp.maximum(residual_rms * update_rms, 1e-8)
        if padding_mask is None:
            scalar_mask = jnp.ones_like(residual_rms)
        else:
            scalar_mask = padding_mask.astype(residual_rms.dtype)
        scalar_count = jnp.maximum(jnp.sum(scalar_mask), 1)

        self.sow(
            "scalars",
            f"depth/{depth}/update_to_residual_norm",
            jnp.sum(update_rms / jnp.maximum(residual_rms, 1e-8) * scalar_mask)
            / scalar_count,
        )
        self.sow(
            "scalars",
            f"depth/{depth}/update_rms_norm",
            jnp.sum(update_rms * scalar_mask) / scalar_count,
        )
        self.sow(
            "scalars",
            f"depth/{depth}/residual_rms_norm",
            jnp.sum(residual_rms * scalar_mask) / scalar_count,
        )
        self.sow(
            "scalars",
            f"depth/{depth}/update_residual_cosine",
            jnp.sum(update_residual_cosine * scalar_mask) / scalar_count,
        )

        ####### end residual diagnostics #######

    def unembed(self, x: jax.Array) -> Any:
        """Compute output distribution."""

        x = self.ln_f(x)
        logits = jnp.einsum("bth,vh->btv", x, self.wte.astype(self._activation_dtype))

        return logits

    def loss(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        """Compute cross-entropy loss given logits and targets."""

        logits_f32 = logits.astype(jnp.float32)
        logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
        targets_flat = targets.reshape(-1)

        # Mask out ignore index (-1)
        mask = targets_flat != -1
        targets_masked = jnp.where(mask, targets_flat, 0)

        loss = -jnp.sum(
            jax.nn.log_softmax(logits_flat, axis=-1)
            * jax.nn.one_hot(targets_masked, self.vocab_size)
            * mask[:, None]
        ) / mask.sum().clip(min=1)

        return loss

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        """
        Args:
            idx: Input token indices of shape (B, T).
            targets: Target token indices of shape (B, T). Use -1 to ignore positions.
            padding_mask: Boolean tensor of shape (B, T). True for valid tokens,
                False for padding tokens.
            deterministic: If False, applies dropout.

        Returns:
            logits: Output logits of shape (B, T, vocab_size).
            loss: Cross-entropy loss if targets provided, else None.
        """
        b, t = idx.shape
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        )

        # get embeddings
        x = self.embed(idx, deterministic, **kwargs)

        # Transformer blocks
        x = self.decode(
            x, padding_mask=padding_mask, deterministic=deterministic, **kwargs
        )

        # Final layer norm and logits
        logits = self.unembed(x)

        # Compute loss if targets provided
        if targets is not None:
            loss = self.loss(logits, targets)
        else:
            loss = None

        return logits, loss
