"""
Grouped Query Attention (GQA) with RoPE, sliding window, and partial rotation support.
Inherits from SelfAttention, overriding projection, RoPE, masking, and attention.
"""

import math
from typing import Any, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.config import configure, field
from theseus.model.axes import Axes
from theseus.model.attention.base import SelfAttention
from theseus.model.layers.rmsnorm import RMSNorm
from theseus.model.layers.rope import RotaryPosEncoding
from theseus.model.masks import (
    causal_mask,
    sliding_window_mask,
    combine_padding,
    cache_mask,
)


class GroupedOutputProjection(nn.Module):
    n_embd: int
    n_kv_head: int
    n_rep: int
    head_dim: int
    use_bias: bool
    kernel_init: nn.initializers.Initializer
    param_dtype: Any
    dtype: Any

    @nn.compact
    def __call__(self, y: jax.Array) -> jax.Array:
        kernel_param = self.param(
            "kernel",
            nn.with_logical_partitioning(
                self.kernel_init,
                (Axes.N_ATTN.value, Axes.N_EMBD.value),
            ),
            (self.n_kv_head * self.n_rep * self.head_dim, self.n_embd),
            self.param_dtype,
        )
        kernel = jnp.asarray(kernel_param, self.dtype)
        kernel = kernel.reshape(self.n_kv_head, self.n_rep, self.head_dim, self.n_embd)
        out = jnp.einsum("btknd,kndo->bto", y.astype(self.dtype), kernel)
        if self.use_bias:
            bias = self.param(
                "bias",
                nn.initializers.zeros,
                (self.n_embd,),
                self.param_dtype,
            )
            out = out + jnp.asarray(bias, self.dtype)
        return out


class GroupedSelfAttention(SelfAttention):
    n_embd: int = field("architecture/n_embd", default=4096)
    n_layers: int = field("architecture/n_layers", default=32)
    n_head: int = field("architecture/n_head", default=32)
    n_kv_head: int = field("architecture/n_kv_head", default=-1)  # -1 -> use n_head
    # explicit head_dim (Qwen 3.5: 256 with n_embd 1024). -1 -> n_embd/n_head.
    head_dim_override: int = field("architecture/head_dim", default=-1)
    dropout: float = field("architecture/dropout", default=0.0)
    attn_dropout: float = field("architecture/attn_dropout", default=0.0)
    rope_theta: float = field("architecture/rope_theta", default=1e6)
    partial_rotary_factor: float = field(
        "architecture/partial_rotary_factor", default=1.0
    )
    use_sliding_window: bool = field("architecture/use_sliding_window", default=False)
    sliding_window: int = field(
        "architecture/sliding_window", default=-1
    )  # -1 -> no sliding
    bias: bool = field("architecture/bias", default=True)
    attn_bias: bool = field("architecture/attention_bias", default=True)
    # Qwen 3.5-style additions:
    #   q_output_gate: q_proj produces 2*head_dim, gate half is applied as
    #     sigmoid(gate) to the post-attention output before o_proj.
    #   qk_head_norm: RMSNorm (centered) on Q and K head_dim before RoPE.
    q_output_gate: bool = False
    qk_head_norm: bool = False

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [RMSNorm]

    def _head_dim(self) -> int:
        if self.head_dim_override > 0:
            return self.head_dim_override
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head

    def flops(self, seq: int) -> float:
        head_dim = self._head_dim()
        kv_heads = self.n_head if self.n_kv_head == -1 else self.n_kv_head
        projections = self.n_head * (3 if self.q_output_gate else 2) + 2 * kv_heads
        return float(
            6 * seq * self.n_embd * head_dim * projections
            + 12 * seq**2 * self.n_head * head_dim
        )

    def setup(self) -> None:
        head_dim = self._head_dim()
        n_kv_head = self.n_head if self.n_kv_head == -1 else self.n_kv_head
        assert self.n_head % n_kv_head == 0
        n_rep = self.n_head // n_kv_head
        self.head_dim = head_dim
        self.n_kv_head_eff = n_kv_head
        self.n_rep = n_rep

        kernel_init_std = 0.02
        proj_init_std = 0.02 / math.sqrt(2 * self.n_layers)

        q_out = self.n_head * head_dim * (2 if self.q_output_gate else 1)
        self.q_proj = nn.Dense(
            q_out,
            use_bias=self.attn_bias,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(stddev=kernel_init_std),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.k_proj = nn.Dense(
            n_kv_head * head_dim,
            use_bias=self.attn_bias,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(stddev=kernel_init_std),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.v_proj = nn.Dense(
            n_kv_head * head_dim,
            use_bias=self.attn_bias,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(stddev=kernel_init_std),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.o_proj = GroupedOutputProjection(
            n_embd=self.n_embd,
            n_kv_head=n_kv_head,
            n_rep=n_rep,
            head_dim=head_dim,
            use_bias=self.attn_bias,
            kernel_init=jax.nn.initializers.normal(stddev=proj_init_std),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

        if self.qk_head_norm:
            self.q_norm = configure(RMSNorm, ndim=head_dim, centered=True)
            self.k_norm = configure(RMSNorm, ndim=head_dim, centered=True)

        self.rope = RotaryPosEncoding(
            head_dim,
            base=int(self.rope_theta),
            seq_dim=1,
            partial_rotary_factor=self.partial_rotary_factor,
        )

    def _repeat_kv(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.n_rep == 1:
            return x
        b, t, kvh, d = x.shape
        x = x[:, :, :, None, :]
        x = jnp.broadcast_to(x, (b, t, kvh, self.n_rep, d))
        return x.reshape(b, t, kvh * self.n_rep, d)

    def _project_inner(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.n_head, self.head_dim)
        k = self.k_proj(x).reshape(b, t, self.n_kv_head_eff, self.head_dim)
        v = self.v_proj(x).reshape(b, t, self.n_kv_head_eff, self.head_dim)
        return q, k, v

    def _project_with_gate(
        self, x: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, Optional[jax.Array]]:
        """Like ``_project_inner`` but also returns a per-token output gate
        when ``q_output_gate`` is set; otherwise ``gate`` is None."""
        b, t, _ = x.shape
        if self.q_output_gate:
            q_raw = self.q_proj(x).reshape(b, t, self.n_head, 2 * self.head_dim)
            q, gate = jnp.split(q_raw, 2, axis=-1)
            gate = gate.reshape(b, t, self.n_head * self.head_dim)
        else:
            q = self.q_proj(x).reshape(b, t, self.n_head, self.head_dim)
            gate = None
        k = self.k_proj(x).reshape(b, t, self.n_kv_head_eff, self.head_dim)
        v = self.v_proj(x).reshape(b, t, self.n_kv_head_eff, self.head_dim)
        return q, k, v, gate

    def preprocess_qkv(
        self, q: jax.Array, k: jax.Array, v: jax.Array, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        if self.qk_head_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rope(q, k, t=kwargs.get("positions"))
        return q, k, v

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        cache_max_len: Optional[int] = None,
        **kwargs: Any,
    ) -> jax.Array:
        B, T, C = x.shape

        q, k, v, gate = self._project_with_gate(x)

        # For decode steps with cache, inject correct RoPE positions.
        if self.has_variable("cache", "cache_index"):
            ci: Any = self.get_variable("cache", "cache_index")
            kwargs = {**kwargs, "positions": self._cached_positions(T, ci)}
        elif "positions" not in kwargs:
            kwargs = {
                **kwargs,
                "positions": self._positions_from_padding(T, padding_mask),
            }

        q, k, v = self.preprocess_qkv(q, k, v, **kwargs)

        # Cache compact GQA KV heads. Attention handles grouped query heads
        # directly so we do not materialize repeated K/V or reshape a sharded
        # KV-head axis together with an unsharded repeat axis.
        k, v, cache_idx = self._cached_kv(
            k, v, padding_mask=padding_mask, cache_max_len=cache_max_len
        )

        T_kv = k.shape[1]
        mask = self.build_mask(T_kv, padding_mask, _cache_index=cache_idx, **kwargs)
        y = self.attn(q, k, v, mask, **kwargs)
        y = self.postprocess_attn(y, padding_mask, deterministic, **kwargs)
        if gate is not None:
            # y: (B, T, n_kv_head, n_rep, head_dim); reshape to (B, T, n_head*head_dim)
            # to multiply with the per-token gate, then back out before o_proj.
            y_flat = y.reshape(B, T, self.n_head * self.head_dim)
            y_flat = y_flat * jax.nn.sigmoid(gate.astype(y_flat.dtype))
            y = y_flat.reshape(B, T, self.n_kv_head_eff, self.n_rep, self.head_dim)
        y = self.output_proj(y)

        if not deterministic and self.dropout > 0:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return y

    def build_mask(
        self,
        t: int,
        padding_mask: Optional[jax.Array],
        **kwargs: Any,
    ) -> Optional[jax.Array]:
        # When cache is active, _cache_index is passed from __call__
        ci = kwargs.get("_cache_index")
        if ci is not None:
            mask = cache_mask(t, ci)
            if self.has_variable("cache", "cached_padding_mask"):
                pad: jax.Array = self.get_variable("cache", "cached_padding_mask")
                mask = mask & pad[:, None, None, :]
            return mask
        # Accept an already-prepared 4D mask (True=keep) for parity/debug
        if padding_mask is not None and padding_mask.ndim == 4:
            return padding_mask
        sliding = kwargs.get("sliding", False)
        base = (
            sliding_window_mask(t, self.sliding_window)
            if sliding and self.sliding_window > 0
            else causal_mask(t)
        )
        return combine_padding(base, padding_mask)

    def attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> jax.Array:
        b = q.shape[0]
        t_q = q.shape[1]
        kvh = k.shape[2]
        qh = q.reshape(b, t_q, kvh, self.n_rep, self.head_dim)
        qh = qh.transpose(0, 2, 3, 1, 4).astype(self._activation_dtype)
        kh = k.transpose(0, 2, 1, 3).astype(self._activation_dtype)
        vh = v.transpose(0, 2, 1, 3).astype(self._activation_dtype)

        scores = jnp.einsum("bkntd,bkTd->bkntT", qh, kh)
        scores = scores / jnp.sqrt(self.head_dim).astype(jnp.float32)

        if mask is not None:
            if mask.ndim == 4:
                mask = mask[:, :, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, None, :, :]
            elif mask.ndim == 2:
                mask = mask[:, None, None, None, :]
            bias = jnp.where(jnp.broadcast_to(mask, scores.shape), 0.0, -1e9).astype(
                jnp.float32
            )
            scores = scores + bias

        attn_w = jax.nn.softmax(scores, axis=-1).astype(vh.dtype)
        y = jnp.einsum("bkntT,bkTd->bkntd", attn_w, vh.astype(attn_w.dtype))
        return y.transpose(0, 3, 1, 2, 4)

    def postprocess_attn(
        self,
        y: jax.Array,
        padding_mask: Optional[jax.Array],
        deterministic: bool,
        **kwargs: Any,
    ) -> jax.Array:
        # No-op: dropout is handled in the shared __call__ after output_proj
        return y

    def output_proj(self, y: jax.Array) -> jax.Array:
        return self.o_proj(y)
