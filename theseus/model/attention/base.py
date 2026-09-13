"""
Base self-attention module with hook-based API and KV cache support.

All Q/K/V hooks operate on (B, T, H, D) format.
Subclasses override _project_inner/preprocess_qkv/build_mask/attn/postprocess_attn/output_proj.

KV cache is activated by calling model.apply(..., mutable=['cache']).
Without mutable=['cache'], the cache is a no-op (training mode).
"""

import math
import os
from typing import Any, List, Literal, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.base.axis import ShardingPlan
from theseus.config import field
from theseus.model.axes import Axes
from theseus.model.module import Module
from theseus.model.masks import cache_mask


_CUDNN_DEVICE_KINDS = ("B200", "H100", "H200", "GH200", "B100")
_AttnBackend = Optional[Literal["xla", "cudnn"]]
_ATTENTION_BACKEND_RESOLVED: bool = False
_ATTENTION_BACKEND: _AttnBackend = None


def _select_attention_backend() -> _AttnBackend:
    """Pick a jax.nn.dot_product_attention `implementation` for the local hw.

    Returns None to let JAX choose (safe default on CPU/TPU). Returns "cudnn"
    on NVIDIA datacenter GPUs where the fused FMHA kernel is a large win.
    Honors THESEUS_ATTN_BACKEND ∈ {auto, xla, cudnn} as an escape hatch.
    """
    global _ATTENTION_BACKEND, _ATTENTION_BACKEND_RESOLVED
    if _ATTENTION_BACKEND_RESOLVED:
        return _ATTENTION_BACKEND

    override = os.environ.get("THESEUS_ATTN_BACKEND", "auto").lower()
    if override == "xla":
        _ATTENTION_BACKEND = "xla"
        _ATTENTION_BACKEND_RESOLVED = True
        return _ATTENTION_BACKEND
    if override == "cudnn":
        _ATTENTION_BACKEND = "cudnn"
        _ATTENTION_BACKEND_RESOLVED = True
        return _ATTENTION_BACKEND

    try:
        device = jax.devices()[0]
        platform = device.platform
        kind = getattr(device, "device_kind", "") or ""
    except Exception:
        _ATTENTION_BACKEND = None
        _ATTENTION_BACKEND_RESOLVED = True
        return None

    if platform == "gpu" and any(k in kind for k in _CUDNN_DEVICE_KINDS):
        _ATTENTION_BACKEND = "cudnn"
    else:
        _ATTENTION_BACKEND = None
    _ATTENTION_BACKEND_RESOLVED = True
    return _ATTENTION_BACKEND


class SelfAttention(Module):
    n_embd: int = field("architecture/n_embd", default=2048)
    n_layers: int = field("architecture/n_layers", default=32)
    bias: bool = field("architecture/bias", default=True)
    dropout: float = field("architecture/dropout", default=0.0)

    n_head: int = field("architecture/n_head", default=16)
    block_size: int = field("architecture/block_size", default=512)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return []

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(tp=[])

    def flops(self, seq: int) -> float:
        # Q/K/V/output projections plus QK^T and attention-times-V.
        return float(24 * seq * self.n_embd**2 + 12 * seq**2 * self.n_embd)

    def setup(self) -> None:
        assert self.n_embd % self.n_head == 0
        self.head_dim = self.n_embd // self.n_head
        self.c_attn = nn.Dense(
            3 * self.n_embd,
            use_bias=self.bias,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(stddev=0.02),
                (Axes.N_EMBD.value, Axes.N_ATTN.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

        self.c_proj = nn.Dense(
            self.n_embd,
            use_bias=self.bias,
            kernel_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(stddev=0.02 / math.sqrt(2 * self.n_layers)),
                (Axes.N_EMBD_OUT.value, Axes.N_EMBD.value),
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

    # ------------------------------------------------------------------
    # Projection hooks
    # ------------------------------------------------------------------

    def _project_inner(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Raw Q/K/V projection. Override in subclasses. Returns (B, T, H, D)."""
        B, T, _C = x.shape
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.reshape(B, T, self.n_head, self.head_dim)
        k = k.reshape(B, T, self.n_head, self.head_dim)
        v = v.reshape(B, T, self.n_head, self.head_dim)
        return q, k, v

    def project(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Project input to (q, k, v). Calls _project_inner."""
        return self._project_inner(x)

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------

    @nn.compact
    def _cached_kv(
        self,
        k: jax.Array,
        v: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        cache_max_len: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        """Update KV cache if active. k, v: (B, T, H, D).

        Args:
            cache_max_len: cache slot count. Defaults to ``self.block_size``
                — pass an explicit value (e.g. ``prompt_max + max_new_tokens``)
                to size the cache to actual decode need instead of the
                model's full max-context. Must be a Python int (used in
                shape construction).

        Returns (k, v, cache_index_after_update) where cache_index is None
        when cache is not active (training mode).
        """
        if not self.is_mutable_collection("cache"):
            return k, v, None

        # Check BEFORE creating variables: True only when cache was passed
        # in (decode step), False when cache is freshly initialized (prefill).
        is_initialized = self.has_variable("cache", "cache_index")

        # Cache is requested — create or update variables
        cache_len = cache_max_len if cache_max_len is not None else self.block_size
        B, _T, H, D = k.shape
        cache_shape = (B, cache_len, H, D)
        cached_key = self.variable(
            "cache", "cached_key", jnp.zeros, cache_shape, k.dtype
        )
        cached_value = self.variable(
            "cache", "cached_value", jnp.zeros, cache_shape, v.dtype
        )
        cache_index = self.variable(
            "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
        )
        # Padding mask for cached positions: True = real token, False = padding.
        # Initialized to all-True; prefill overwrites with actual mask.
        cached_pad = self.variable(
            "cache",
            "cached_padding_mask",
            jnp.ones,
            (B, cache_len),
            jnp.bool_,
        )

        if is_initialized:
            # Decode step: k, v are (B, 1, H, D) — single new token
            cur_index = cache_index.value
            batch_dims = k.ndim - 3  # typically 1 (B)
            zero = jnp.array(0, dtype=jnp.int32)
            indices: tuple[jax.Array, ...] = (zero,) * batch_dims + (
                cur_index,
                zero,
                zero,
            )
            k = jax.lax.dynamic_update_slice(cached_key.value, k, indices)
            v = jax.lax.dynamic_update_slice(cached_value.value, v, indices)
            cached_key.value = k
            cached_value.value = v
            new_index = cur_index + 1
            cache_index.value = new_index
            return k, v, new_index
        else:
            # Prefill: k, v are (B, T_prefill, H, D) — write into the larger cache
            T_prefill = k.shape[1]
            batch_dims = k.ndim - 3
            zero = jnp.array(0, dtype=jnp.int32)
            indices = (zero,) * batch_dims + (zero, zero, zero)
            new_cached_k = jax.lax.dynamic_update_slice(cached_key.value, k, indices)
            new_cached_v = jax.lax.dynamic_update_slice(cached_value.value, v, indices)
            cached_key.value = new_cached_k
            cached_value.value = new_cached_v
            cache_index.value = jnp.array(T_prefill, dtype=jnp.int32)
            # Store padding mask for decode steps
            if padding_mask is not None:
                pad_full = jnp.ones((B, cache_len), dtype=jnp.bool_)
                pad_full = jax.lax.dynamic_update_slice(
                    pad_full, padding_mask, (zero, zero)
                )
                cached_pad.value = pad_full
            # Return original k, v (not full cache) — prefill uses normal causal mask
            return k, v, None

    # ------------------------------------------------------------------
    # Attention hooks
    # ------------------------------------------------------------------

    def _positions_from_padding(
        self, length: int, padding_mask: Optional[jax.Array]
    ) -> jax.Array:
        if padding_mask is None:
            return jnp.arange(length)
        return jnp.maximum(jnp.cumsum(padding_mask, axis=-1) - 1, 0)

    def _cached_positions(self, length: int, cache_index: jax.Array) -> jax.Array:
        positions = jnp.arange(length) + cache_index
        if self.has_variable("cache", "cached_padding_mask"):
            pad: jax.Array = self.get_variable("cache", "cached_padding_mask")
            cached = jnp.arange(pad.shape[-1]) < cache_index
            n_pad = jnp.sum(cached & ~pad, axis=-1)
            positions = positions[None, :] - n_pad[:, None]
        return positions

    def preprocess_qkv(
        self, q: jax.Array, k: jax.Array, v: jax.Array, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Hook for RoPE, KV repeat, etc. Input/output: (B, T, H, D)."""
        return q, k, v

    def build_mask(
        self,
        t: int,
        padding_mask: Optional[jax.Array],
        **kwargs: Any,
    ) -> Optional[jax.Array]:
        """Construct attention mask. Returns bool mask or None."""
        ci = kwargs.get("_cache_index")
        if ci is not None:
            mask = cache_mask(t, ci)
            # Combine with cached padding mask so decode doesn't attend to
            # padding positions from the prefill.
            if self.has_variable("cache", "cached_padding_mask"):
                pad: jax.Array = self.get_variable("cache", "cached_padding_mask")
                mask = mask & pad[:, None, None, :]  # (B, 1, 1, T_kv)
            return mask
        if padding_mask is not None:
            # Combine causal mask with padding mask so future tokens AND
            # padding tokens are both blocked.
            causal = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))[
                None, None, :, :
            ]  # (1, 1, T, T)
            pad = padding_mask[:, None, None, :]  # (B, 1, 1, T)
            return causal & pad
        return None

    def attn(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Core attention. Input q/k/v: (B, T_q/T_kv, H, D). Output: (B, T_q, H, D)."""
        # dot_product_attention expects (B, T, H, D) — same as our convention
        # Compute attention in float32 for precision parity between
        # full-sequence and cached single-token paths
        q = q.astype(self._activation_dtype)
        k = k.astype(self._activation_dtype)
        v = v.astype(self._activation_dtype)

        backend = _select_attention_backend()
        # The batch dimension is unchanged here. cuDNN's fused DPA requires
        # the bias query *sequence* dimension to match the KV sequence length.
        # Cached prefill can have T_q < T_kv when prompts occupy only part of a
        # fixed-size cache, so use the XLA kernel for that rollout-only shape.
        if backend == "cudnn" and q.shape[1] != k.shape[1]:
            backend = "xla"
        if mask is not None:
            # Use additive bias (-inf for masked) instead of boolean mask
            # to ensure identical DPA kernel path as is_causal=True.
            # The bias must carry the q/k/v dtype: the cudnn fused kernel
            # returns NaN for an f32 bias against bf16 operands.
            finfo = jnp.finfo(q.dtype)  # type: ignore[no-untyped-call]
            neg = max(-1e9, float(finfo.min) / 2)
            bias = jnp.where(mask, 0.0, neg).astype(q.dtype)
            y = jax.nn.dot_product_attention(q, k, v, bias=bias, implementation=backend)
        else:
            y = jax.nn.dot_product_attention(
                q, k, v, is_causal=True, implementation=backend
            )

        return y

    def postprocess_attn(
        self,
        y: jax.Array,
        padding_mask: Optional[jax.Array],
        deterministic: bool,
        **kwargs: Any,
    ) -> jax.Array:
        """Post-attention processing. Input/output: (B, T, H, D)."""
        if padding_mask is not None:
            y = y * padding_mask[:, :, None, None].astype(y.dtype)

        if not deterministic and self.dropout > 0:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return y

    def output_proj(self, y: jax.Array) -> jax.Array:
        """Output projection. (B, T, C) → (B, T, C)."""
        return self.c_proj(y)

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

        q, k, v = self.project(x)

        # For decode steps with cache, inject correct RoPE positions
        if self.has_variable("cache", "cache_index"):
            ci: Any = self.get_variable("cache", "cache_index")
            kwargs = {**kwargs, "positions": self._cached_positions(T, ci)}
        elif "positions" not in kwargs:
            kwargs = {
                **kwargs,
                "positions": self._positions_from_padding(T, padding_mask),
            }

        q, k, v = self.preprocess_qkv(q, k, v, **kwargs)
        k, v, cache_idx = self._cached_kv(
            k, v, padding_mask=padding_mask, cache_max_len=cache_max_len
        )

        T_kv = k.shape[1]
        mask = self.build_mask(T_kv, padding_mask, _cache_index=cache_idx, **kwargs)
        y = self.attn(q, k, v, mask, **kwargs)
        y = self.postprocess_attn(y, padding_mask, deterministic, **kwargs)

        y = y.reshape(B, T, C)
        y = self.output_proj(y)

        if not deterministic and self.dropout > 0:
            y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        return y
