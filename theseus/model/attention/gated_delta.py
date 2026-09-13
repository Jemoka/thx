"""Gated delta-rule linear attention (Qwen 3.5 / DeltaNet-style).

The token mixer is a depthwise causal conv1d feeding into a chunked
gated-delta-rule kernel, then an RMSNorm gated by a per-token ``z`` gate.

Token shapes throughout (JAX-side):
    Q, K: ``(B, T, num_k_heads, head_k_dim)``
    V:    ``(B, T, num_v_heads, head_v_dim)``
    g, beta: ``(B, T, num_v_heads)``

Parity reference: ``transformers.models.qwen3_5.modeling_qwen3_5
.torch_chunk_gated_delta_rule``. We port the chunked variant (no recurrent
single-token path) — sufficient for full-sequence forward parity.
"""

from typing import Any, List, Optional, Tuple, Type

import flax.linen as nn
import jax
import jax.numpy as jnp

from theseus.base.axis import ShardingPlan
from theseus.config import configure, field
from theseus.model.axes import Axes
from theseus.model.layers.rmsnorm import RMSNormGated
from theseus.model.module import Module


def _l2norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def chunk_gated_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    return_state: bool = False,
) -> Any:
    """Chunked gated delta-rule attention. See module docstring for shapes.

    Returns ``(B, T, num_v_heads, head_v_dim)``. When ``return_state`` is set,
    also returns the final recurrent state ``(B, H, head_k_dim, head_v_dim)``
    so a decode loop can continue token-by-token via
    :func:`recurrent_gated_delta_step`. Padding positions (zero q/k/v/beta/g)
    leave the state unchanged, so the returned state reflects only real tokens.
    """
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)

    # Reorder to (B, H, T, D) in fp32 — matches HF reference.
    q = q.transpose(0, 2, 1, 3).astype(jnp.float32)
    k = k.transpose(0, 2, 1, 3).astype(jnp.float32)
    v = v.transpose(0, 2, 1, 3).astype(jnp.float32)
    beta = beta.transpose(0, 2, 1).astype(jnp.float32)
    g = g.transpose(0, 2, 1).astype(jnp.float32)

    B, H, T, D_k = k.shape
    D_v = v.shape[-1]
    pad_size = (chunk_size - T % chunk_size) % chunk_size
    if pad_size:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))
    T_total = T + pad_size
    n_chunks = T_total // chunk_size

    scale = 1.0 / jnp.sqrt(jnp.asarray(D_k, dtype=jnp.float32))
    q = q * scale

    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]

    def chunked(x: jax.Array) -> jax.Array:
        return x.reshape(B, H, n_chunks, chunk_size, *x.shape[3:])

    q = chunked(q)
    k = chunked(k)
    v = chunked(v)
    k_beta = chunked(k_beta)
    v_beta = chunked(v_beta)
    g = chunked(g)

    g_cum = jnp.cumsum(g, axis=-1)  # (B, H, n_chunks, chunk_size)

    # decay_mask[..., i, j] = exp(g_cum[..., i] - g_cum[..., j]) for i >= j.
    # The upper triangle has g_diff > 0 → exp(g_diff) can overflow to inf in
    # forward; even though we mask it out, autodiff would propagate
    # ``0 * inf = NaN`` in backward. Sanitize g_diff before ``exp``.
    g_diff = g_cum[..., :, None] - g_cum[..., None, :]
    tri = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    safe_g_diff = jnp.where(tri, g_diff, 0.0)
    decay_mask = jnp.where(tri, jnp.exp(safe_g_diff), 0.0).astype(jnp.float32)

    # Per-chunk attn matrix: attn = (I - L)^{-1} where L is strictly
    # lower-triangular and nilpotent. Direct triangular solve is faster than
    # expanding the Neumann series and (with the safe-where on decay_mask
    # above) numerically stable in both forward and backward.
    L = -jnp.einsum("bhcsd,bhcSd->bhcsS", k_beta, k) * decay_mask
    upper_incl = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=0)
    L = jnp.where(upper_incl[None, None, None, :, :], 0.0, L)
    I_minus_L = jnp.eye(chunk_size, dtype=L.dtype) - L
    v_corrected = jax.lax.linalg.triangular_solve(
        I_minus_L, v_beta, lower=True, left_side=True
    )
    k_cumdecay = jax.lax.linalg.triangular_solve(
        I_minus_L,
        k_beta * jnp.exp(g_cum)[..., None],
        lower=True,
        left_side=True,
    )

    def scan_body(state: jax.Array, inputs: Any) -> Tuple[jax.Array, jax.Array]:
        q_i, k_i, v_corr_i, k_cd_i, g_cum_i, decay_i = inputs
        attn_inner = jnp.einsum("bhsd,bhSd->bhsS", q_i, k_i) * decay_i
        v_prime = jnp.einsum("bhsd,bhde->bhse", k_cd_i, state)
        v_new = v_corr_i - v_prime
        attn_inter = jnp.einsum(
            "bhsd,bhde->bhse", q_i * jnp.exp(g_cum_i)[..., None], state
        )
        out = attn_inter + jnp.einsum("bhsS,bhSe->bhse", attn_inner, v_new)
        g_last = g_cum_i[..., -1:]
        decay_to_end = jnp.exp(g_last - g_cum_i)
        new_state = state * jnp.exp(g_last)[..., None] + jnp.einsum(
            "bhsd,bhse->bhde", k_i * decay_to_end[..., None], v_new
        )
        return new_state, out

    def cfirst(x: jax.Array) -> jax.Array:
        return jnp.transpose(x, (2, 0, 1) + tuple(range(3, x.ndim)))

    state0 = jnp.zeros((B, H, D_k, D_v), dtype=jnp.float32)
    inputs = (
        cfirst(q),
        cfirst(k),
        cfirst(v_corrected),
        cfirst(k_cumdecay),
        cfirst(g_cum),
        cfirst(decay_mask),
    )
    final_state, outs = jax.lax.scan(scan_body, state0, inputs)
    core_attn_out = jnp.transpose(
        outs, (1, 2, 0, 3, 4)
    )  # (B, H, n_chunks, chunk_size, D_v)
    core_attn_out = core_attn_out.reshape(B, H, T_total, D_v)[:, :, :T]
    out = core_attn_out.transpose(0, 2, 1, 3)  # (B, T, H, D_v)
    if return_state:
        return out, final_state  # final_state: (B, H, D_k, D_v)
    return out


def recurrent_gated_delta_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    state: jax.Array,
    use_qk_l2norm: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Single-token gated delta-rule recurrence (the decode counterpart of
    :func:`chunk_gated_delta_rule`).

    Shapes: ``q, k`` ``(B, 1, H, D_k)``, ``v`` ``(B, 1, H, D_v)``,
    ``g, beta`` ``(B, 1, H)``, ``state`` ``(B, H, D_k, D_v)``.
    Returns ``(out (B, 1, H, D_v), new_state (B, H, D_k, D_v))``.

    Derived as the ``chunk_size == 1`` specialization of ``scan_body`` above,
    so token-by-token decoding reproduces the chunked path's logits exactly
    (up to fp accumulation order).
    """
    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)
    # (B, 1, H, D) -> (B, H, 1, D); g/beta (B, 1, H) -> (B, H, 1)
    q = q.transpose(0, 2, 1, 3).astype(jnp.float32)
    k = k.transpose(0, 2, 1, 3).astype(jnp.float32)
    v = v.transpose(0, 2, 1, 3).astype(jnp.float32)
    g = g.transpose(0, 2, 1).astype(jnp.float32)  # (B, H, 1)
    beta = beta.transpose(0, 2, 1).astype(jnp.float32)  # (B, H, 1)
    state = state.astype(jnp.float32)

    D_k = k.shape[-1]
    q = q * (1.0 / jnp.sqrt(jnp.asarray(D_k, dtype=jnp.float32)))

    e = jnp.exp(g)  # (B, H, 1)
    v_beta = v * beta[..., None]  # (B, H, 1, D_v)
    kS = jnp.einsum("bhsd,bhde->bhse", k, state)  # (B, H, 1, D_v)
    v_prime = (beta * e)[..., None] * kS
    v_new = v_beta - v_prime
    qS = jnp.einsum("bhsd,bhde->bhse", q, state)  # (B, H, 1, D_v)
    qk = jnp.sum(q * k, axis=-1, keepdims=True)  # (B, H, 1, 1)
    out = e[..., None] * qS + qk * v_new  # (B, H, 1, D_v)
    # e is (B, H, 1); broadcast its scalar over (D_k, D_v) of the state.
    new_state = e[..., None] * state + jnp.einsum("bhsd,bhse->bhde", k, v_new)
    return out.transpose(0, 2, 1, 3), new_state  # out: (B, 1, H, D_v)


def causal_conv1d_step(
    x_t: jax.Array, conv_state: jax.Array, weight: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Single-step causal depthwise conv1d using a rolling input cache.

    ``x_t`` (B, 1, C); ``conv_state`` (B, K-1, C) holds the previous K-1 inputs;
    ``weight`` (C, K). Returns ``(out (B, 1, C), new_conv_state (B, K-1, C))``.
    Matches :func:`causal_depthwise_conv1d` at the appended position.
    """
    window = jnp.concatenate([conv_state, x_t], axis=1)  # (B, K, C)
    out = jnp.einsum("bkc,ck->bc", window, weight)[:, None, :]
    return out, window[:, 1:, :]


def _conv_input_tail(x: jax.Array, need: int) -> jax.Array:
    """Last ``need`` timesteps of ``x`` (B, T, C), left-zero-padded if T < need.

    This is the conv-cache seed after a prefill: the next decode step's window
    is ``[tail, x_new]``, reproducing ``causal_depthwise_conv1d`` at that
    position.
    """
    B, T, C = x.shape
    if T >= need:
        return x[:, T - need :, :]
    pad = jnp.zeros((B, need - T, C), x.dtype)
    return jnp.concatenate([pad, x], axis=1)


def causal_depthwise_conv1d(
    x: jax.Array, weight: jax.Array, kernel_size: int
) -> jax.Array:
    """Causal depthwise conv1d: ``x`` (B, T, C); ``weight`` (C, kernel_size).

    Matches HF Conv1d(groups=C, padding=K-1) followed by ``[:, :, :T]``.
    """
    B, T, C = x.shape
    x_pad = jnp.pad(x, ((0, 0), (kernel_size - 1, 0), (0, 0)))
    windows = jnp.stack([x_pad[:, k : k + T, :] for k in range(kernel_size)], axis=-1)
    return jnp.einsum("btck,ck->btc", windows, weight)


class GatedDeltaNet(Module):
    """Gated delta-rule linear attention token mixer."""

    n_embd: int = field("architecture/n_embd", default=1024)
    rms_norm_eps: float = field("architecture/rms_norm_eps", default=1e-6)
    linear_num_value_heads: int = field(
        "architecture/linear_num_value_heads", default=16
    )
    linear_num_key_heads: int = field("architecture/linear_num_key_heads", default=16)
    linear_key_head_dim: int = field("architecture/linear_key_head_dim", default=128)
    linear_value_head_dim: int = field(
        "architecture/linear_value_head_dim", default=128
    )
    linear_conv_kernel_dim: int = field(
        "architecture/linear_conv_kernel_dim", default=4
    )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [RMSNormGated]

    @property
    def sharding(self) -> ShardingPlan:
        return ShardingPlan(tp=[])

    @property
    def _key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def _value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def _conv_dim(self) -> int:
        return 2 * self._key_dim + self._value_dim

    def setup(self) -> None:
        kernel_init = nn.initializers.normal(stddev=0.02)

        self.in_proj_qkv = nn.Dense(
            self._conv_dim,
            use_bias=False,
            kernel_init=nn.with_logical_partitioning(
                kernel_init, (Axes.N_EMBD.value, Axes.N_ATTN.value)
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.in_proj_z = nn.Dense(
            self._value_dim,
            use_bias=False,
            kernel_init=nn.with_logical_partitioning(
                kernel_init, (Axes.N_EMBD.value, Axes.N_ATTN.value)
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.in_proj_b = nn.Dense(
            self.linear_num_value_heads,
            use_bias=False,
            kernel_init=kernel_init,
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.in_proj_a = nn.Dense(
            self.linear_num_value_heads,
            use_bias=False,
            kernel_init=kernel_init,
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )
        self.out_proj = nn.Dense(
            self.n_embd,
            use_bias=False,
            kernel_init=nn.with_logical_partitioning(
                kernel_init, (Axes.N_ATTN.value, Axes.N_EMBD.value)
            ),
            param_dtype=self._param_dtype,
            dtype=self._activation_dtype,
        )

        # Depthwise causal conv. weight shape (conv_dim, kernel_size); the HF
        # checkpoint stores it as (conv_dim, 1, kernel_size) — loader squeezes.
        # conv_dim is the concat of [Q, K, V] head channels — partition on N_ATTN.
        self.conv_weight = self.param(
            "conv_weight",
            nn.with_logical_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.N_ATTN.value, None),
            ),
            (self._conv_dim, self.linear_conv_kernel_dim),
            self._param_dtype,
        )
        # dt_bias initialized to ones; A_log set by HF init to log(U(0,16)).
        self.dt_bias = self.param(
            "dt_bias",
            nn.initializers.ones,
            (self.linear_num_value_heads,),
            self._param_dtype,
        )
        self.A_log = self.param(
            "A_log",
            nn.initializers.zeros,
            (self.linear_num_value_heads,),
            self._param_dtype,
        )
        self.norm = configure(RMSNormGated, ndim=self.linear_value_head_dim)

    def _split_qkv(
        self, mixed_qkv: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        B, T, _ = mixed_qkv.shape
        k_dim = self._key_dim
        q = mixed_qkv[..., :k_dim].reshape(
            B, T, self.linear_num_key_heads, self.linear_key_head_dim
        )
        k = mixed_qkv[..., k_dim : 2 * k_dim].reshape(
            B, T, self.linear_num_key_heads, self.linear_key_head_dim
        )
        v = mixed_qkv[..., 2 * k_dim :].reshape(
            B, T, self.linear_num_value_heads, self.linear_value_head_dim
        )
        # Repeat K/Q heads if num_v_heads > num_k_heads (GQA-style)
        rep = self.linear_num_value_heads // self.linear_num_key_heads
        if rep > 1:
            q = jnp.repeat(q, rep, axis=2)
            k = jnp.repeat(k, rep, axis=2)
        return q, k, v

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        # cache_max_len is irrelevant here: the recurrent state is fixed-size
        # (H, D_k, D_v), independent of sequence length. Accept and ignore it.
        del deterministic, kwargs

        B, T, _ = x.shape
        # HF ``apply_mask_to_padding_states`` zeroes padded tokens before the
        # linear-attention block when both batch>1 and seq>1. Matching this is
        # important for training: gradients through the chunked delta rule on
        # padded positions can NaN otherwise.
        if padding_mask is not None and B > 1 and T > 1:
            x = x * padding_mask[:, :, None].astype(x.dtype)

        conv_w = jnp.asarray(self.conv_weight, dtype=self._activation_dtype)
        K = self.linear_conv_kernel_dim
        Hv, Dk, Dv = (
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
        )

        mixed_pre = self.in_proj_qkv(x)  # (B, T, conv_dim), pre-conv
        z = self.in_proj_z(x).reshape(B, T, Hv, Dv)
        b_raw = self.in_proj_b(x)
        a_raw = self.in_proj_a(x)
        beta = jax.nn.sigmoid(b_raw.astype(jnp.float32))
        g = -jnp.exp(self.A_log.astype(jnp.float32)) * jax.nn.softplus(
            a_raw.astype(jnp.float32) + self.dt_bias.astype(jnp.float32)
        )

        # Cache wiring mirrors SelfAttention._cached_kv: check has_variable
        # BEFORE creating the variables to distinguish prefill from decode.
        cache_active = self.is_mutable_collection("cache")
        is_decode = self.has_variable("cache", "gd_state")
        if cache_active:
            gd_state = self.variable(
                "cache", "gd_state", jnp.zeros, (B, Hv, Dk, Dv), jnp.float32
            )
            conv_state = self.variable(
                "cache",
                "conv_state",
                jnp.zeros,
                (B, K - 1, self._conv_dim),
                conv_w.dtype,
            )

        if cache_active and is_decode:
            # Decode: one token. Advance conv via rolling cache, then apply the
            # recurrent delta-rule update to the stored state.
            mixed_qkv, new_conv = causal_conv1d_step(
                mixed_pre, conv_state.value, conv_w
            )
            conv_state.value = new_conv
            mixed_qkv = jax.nn.silu(mixed_qkv)
            q, k, v = self._split_qkv(mixed_qkv)
            core, new_state = recurrent_gated_delta_step(
                q, k, v, g, beta, gd_state.value
            )
            gd_state.value = new_state
        else:
            # Full-sequence / prefill: chunked kernel. On prefill, also persist
            # the final recurrent state and the conv input tail for later decode.
            mixed_qkv = causal_depthwise_conv1d(mixed_pre, conv_w, K)
            mixed_qkv = jax.nn.silu(mixed_qkv)
            q, k, v = self._split_qkv(mixed_qkv)
            if cache_active:
                core, final_state = chunk_gated_delta_rule(
                    q, k, v, g, beta, return_state=True
                )
                gd_state.value = final_state
                conv_state.value = _conv_input_tail(mixed_pre, K - 1)
            else:
                core = chunk_gated_delta_rule(q, k, v, g, beta)

        core = self.norm(core, gate=z)
        core = core.reshape(B, T, self._value_dim)
        return self.out_proj(core)
