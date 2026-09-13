from dataclasses import dataclass
from theseus.config import field
from .base import Optimizer
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.core import meta
from jax.sharding import PartitionSpec as P
from optax._src import base


# Polar Express coefficients for orthogonalization.
# From https://arxiv.org/pdf/2505.16932 (computed for num_iters=5, safety_factor=2e-2, cushion=2)
_POLAR_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@dataclass
class MuonConfig:
    # Muon-specific hyperparameters
    momentum: float = field("optimization/muon/momentum", default=0.95)
    ns_steps: int = field("optimization/muon/ns_steps", default=5)
    muon_beta2: float = field("optimization/muon/beta2", default=0.95)

    # LR multipliers relative to the base schedule LR (matrix params define the "1x" baseline)
    # Example raw values: matrix=0.02, embedding=0.3, unembedding=0.004, scalar=0.5
    # → multipliers: matrix=1.0, embedding=15.0, unembedding=0.2, scalar=25.0
    matrix_lr_multiplier: float = field(
        "optimization/muon/matrix_lr_multiplier", default=1.0
    )
    embedding_lr_multiplier: float = field(
        "optimization/muon/embedding_lr_multiplier", default=15.0
    )
    unembedding_lr_multiplier: float = field(
        "optimization/muon/unembedding_lr_multiplier", default=0.2
    )
    scalar_lr_multiplier: float = field(
        "optimization/muon/scalar_lr_multiplier", default=25.0
    )

    # AdamW hyperparameters for embedding / unembedding / scalar params.
    # Share paths with AdamWConfig so old/mixed configs compose without duplication.
    weight_decay: float = field("optimization/weight_decay", default=0.0)
    beta1: float = field("optimization/beta1", default=0.8)
    beta2: float = field("optimization/beta2", default=0.95)
    adam_eps: float = field("optimization/eps", default=1e-10)
    # > 0 clips the whole gradient tree by global norm before any branch
    clip_global_norm: float = field("optimization/muon/clip_global_norm", default=0.0)


class _MuonState(NamedTuple):
    momentum: Any  # pytree matching params: first-moment buffers
    second_moment: Any  # pytree matching params: factored second-moment buffers
    count: Any  # scalar step counter


def _polar_express(X: jax.Array, num_steps: int) -> jax.Array:
    """Polar Express orthogonalization for a 2D matrix (JAX port of the PyTorch kernel).

    Iteratively applies quintic Newton-Schulz-style iterations to compute the
    polar factor U of X ≈ U S V^T.  The coefficients maximise the slope at zero
    so the update is near-orthogonal after `num_steps` passes.
    """
    X = X / (jnp.linalg.norm(X) * 1.02 + 1e-6)
    if X.shape[0] >= X.shape[1]:  # tall matrix
        for a, b, c in _POLAR_COEFFS[:num_steps]:
            A = X.T @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # wide matrix
        for a, b, c in _POLAR_COEFFS[:num_steps]:
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def scale_by_muon(
    momentum: float = 0.95,
    ns_steps: int = 5,
    beta2: float = 0.95,
) -> base.GradientTransformation:
    """Core Muon gradient transformation.

    Applies three sequential operations to each update:
      1. Nesterov momentum (EMA of gradients with look-ahead blend)
      2. Polar Express orthogonalization for 2-D+ parameters
      3. NorMuon per-row/col adaptive variance reduction

    Parameters with fewer than 2 dimensions receive only the momentum step.

    Args:
        momentum: EMA coefficient for the first-moment buffer.
        ns_steps:  Number of Polar Express iterations (1–5; 5 recommended).
        beta2:     EMA coefficient for the factored second-moment buffer.

    Returns:
        An :class:`optax.GradientTransformation`.
    """

    def init_fn(params: Any) -> _MuonState:
        mu = jax.tree.map(jnp.zeros_like, params)

        def _make_nu(p: Any) -> Any:
            annotation = p if isinstance(p, meta.Partitioned) else None
            value = annotation.value if annotation is not None else p
            shape = list(value.shape)
            names = list(annotation.names) if annotation is not None else None
            if value.ndim >= 2:
                # The reduced dimension has no logical axis in a factored statistic.
                reduced = -1 if shape[-2] >= shape[-1] else -2
                shape[reduced] = 1
                if names is not None:
                    names[reduced] = None
            result = jnp.zeros(shape, dtype=jnp.float32)
            return (
                annotation.replace(value=result, names=tuple(names))
                if annotation is not None and names is not None
                else result
            )

        nu = jax.tree.map(_make_nu, params, is_leaf=meta.is_axis_metadata)
        return _MuonState(momentum=mu, second_moment=nu, count=jnp.zeros([], jnp.int32))

    def update_fn(
        updates: Any, state: Any, params: Any = None
    ) -> tuple[Any, _MuonState]:
        mu, nu, count = state

        # --- 1. Nesterov momentum (mirrors PyTorch's lerp_ pair) ---
        new_mu = jax.tree.map(
            lambda m, g: momentum * m + (1 - momentum) * g,
            mu,
            updates,
        )
        g = jax.tree.map(
            lambda new_m, orig_g: (1 - momentum) * orig_g + momentum * new_m,
            new_mu,
            updates,
        )

        # --- 2. Polar Express orthogonalization for matrix params ---
        def _orth(grad: Any) -> Any:
            annotation = grad if isinstance(grad, meta.Partitioned) else None
            grad = annotation.value if annotation is not None else grad
            if grad.ndim < 2:
                return annotation if annotation is not None else grad
            orig_shape = grad.shape
            g2d = grad.reshape(grad.shape[0], -1).astype(jnp.float32)
            if meta.global_mesh_defined():
                # Distributed Muon: orthogonalize the complete momentum-adjusted
                # matrix, including both data-parallel and TP partitions.
                g2d = jax.lax.with_sharding_constraint(g2d, P())  # type: ignore[no-untyped-call]
            g_orth = _polar_express(g2d, ns_steps)
            # Rectangular correction: scale by sqrt(max(1, M/N))
            # Flax kernels are [in, out]: the canonical max(1, out/in)**0.5 step
            # scale is shape[1]/shape[0]
            rect_scale = jnp.sqrt(jnp.maximum(1.0, g2d.shape[1] / g2d.shape[0]))
            result = (g_orth * rect_scale).reshape(orig_shape).astype(grad.dtype)
            if annotation is not None:
                result = nn.with_logical_constraint(  # type: ignore[attr-defined]
                    result, annotation.get_partition_spec()
                )
                return annotation.replace_boxed(result)
            return result

        g = jax.tree.map(_orth, g, is_leaf=meta.is_axis_metadata)

        # --- 3. NorMuon variance reduction ---
        def _update_nu(grad: Any, v: Any) -> Any:
            annotation = v if isinstance(v, meta.Partitioned) else None
            grad = grad.value if isinstance(grad, meta.Partitioned) else grad
            if grad.ndim < 2:
                return v
            v = annotation.value if annotation is not None else v
            g_f = grad.astype(jnp.float32)
            if grad.shape[-2] >= grad.shape[-1]:
                v_mean = jnp.mean(g_f**2, axis=-1, keepdims=True)
            else:
                v_mean = jnp.mean(g_f**2, axis=-2, keepdims=True)
            result = (1 - beta2) * v_mean + beta2 * v
            return (
                annotation.replace_boxed(result) if annotation is not None else result
            )

        new_nu = jax.tree.map(_update_nu, g, nu, is_leaf=meta.is_axis_metadata)

        def _apply_normuon(grad: Any, new_v: Any) -> Any:
            annotation = grad if isinstance(grad, meta.Partitioned) else None
            grad = annotation.value if annotation is not None else grad
            if grad.ndim < 2:
                return annotation if annotation is not None else grad
            new_v = new_v.value if isinstance(new_v, meta.Partitioned) else new_v
            g_f = grad.astype(jnp.float32)
            if grad.shape[-2] >= grad.shape[-1]:
                v_mean = jnp.mean(g_f**2, axis=-1, keepdims=True)
                red_dim = grad.shape[-1]
            else:
                v_mean = jnp.mean(g_f**2, axis=-2, keepdims=True)
                red_dim = grad.shape[-2]

            v_norm = jnp.sqrt(jnp.sum(v_mean * red_dim))
            step_size = jnp.maximum(new_v, 1e-10) ** -0.5
            scaled_sq = (v_mean * red_dim) * step_size**2
            v_norm_new = jnp.sqrt(jnp.sum(scaled_sq))
            final_scale = step_size * (v_norm / jnp.maximum(v_norm_new, 1e-10))
            result = (grad * final_scale).astype(grad.dtype)
            return (
                annotation.replace_boxed(result) if annotation is not None else result
            )

        g = jax.tree.map(_apply_normuon, g, new_nu, is_leaf=meta.is_axis_metadata)

        return g, _MuonState(momentum=new_mu, second_moment=new_nu, count=count + 1)

    return base.GradientTransformation(init_fn, update_fn)


def _cautious_weight_decay(weight_decay: float) -> base.GradientTransformation:
    """Apply weight decay only where the update and parameter agree in sign.

    This is the "cautious" variant from the reference implementation:
      update += weight_decay * param * (sign(update) == sign(param))
    """

    def init_fn(params: Any) -> tuple[()]:
        return ()

    def update_fn(updates: Any, state: Any, params: Any = None) -> tuple[Any, Any]:
        if params is None or weight_decay == 0.0:
            return updates, state

        def _apply(u: jax.Array, p: jax.Array) -> jax.Array:
            mask = (u * p >= 0).astype(u.dtype)
            return u + weight_decay * p * mask

        return jax.tree.map(_apply, updates, params), state

    return base.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Default parameter labeller
# ---------------------------------------------------------------------------

_EMBED_KWS = {"embed", "wte", "wpe", "embedding", "token_embed"}
_UNEMBED_KWS = {"lm_head", "unembed", "unembedding"}


def _label_params(params: Any) -> Any:
    """Classify each parameter leaf for :func:`optax.multi_transform`.

    Labels:
      ``'matrix'``      – 2-D+ params that are not embeddings/output (→ Muon)
      ``'embedding'``   – 2-D params whose path contains an embedding keyword
      ``'unembedding'`` – 2-D params whose path contains an unembedding keyword
      ``'scalar'``      – 0-D or 1-D params (biases, norms, scalars)

    The path matching is intentionally conservative and keyword-based; pass a
    custom ``param_labels`` callable to :func:`muon` for non-standard layouts.
    """

    def _classify(path: Any, leaf: jax.Array) -> str:
        path_str = ".".join(
            k.key if hasattr(k, "key") else str(k) for k in path
        ).lower()
        if leaf.ndim <= 1:
            return "scalar"
        if any(kw in path_str for kw in _UNEMBED_KWS):
            return "unembedding"
        if any(kw in path_str for kw in _EMBED_KWS):
            return "embedding"
        return "matrix"

    return jax.tree_util.tree_map_with_path(_classify, params)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def muon(
    lr: optax._src.base.Schedule | float,
    cfg: MuonConfig,
    param_labels: Callable[[Any], Any] | None = None,
) -> optax.GradientTransformation:
    """Muon + AdamW mixed optimizer, mirroring the :func:`adamw` API.

    Applies the Muon update (momentum → Polar-Express orthogonalization →
    NorMuon variance reduction) to matrix-shaped parameters, and standard
    AdamW to embedding, unembedding, and scalar parameters.  Each group is
    scaled by its own LR multiplier relative to the base schedule.

    Args:
        lr:           Base LR schedule (``step → float``) or constant float.
                      Matrix params receive ``lr * cfg.matrix_lr_multiplier``.
        cfg:          :class:`MuonConfig` with all hyperparameters.
        param_labels: Optional callable ``params → labels_pytree``.  If
                      ``None``, :func:`_label_params` is used (path-keyword
                      heuristic).  Pass a custom callable for architectures
                      with non-standard naming.

    Returns:
        An :class:`optax.GradientTransformation`.
    """

    def _scaled(multiplier: float) -> optax._src.base.Schedule | float:
        if callable(lr):
            return lambda step: lr(step) * multiplier
        return lr * multiplier

    matrix_tx = optax.chain(
        scale_by_muon(cfg.momentum, cfg.ns_steps, cfg.muon_beta2),
        _cautious_weight_decay(cfg.weight_decay),
        optax.scale_by_learning_rate(_scaled(cfg.matrix_lr_multiplier)),
    )

    def _adamw_tx(multiplier: float) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(b1=cfg.beta1, b2=cfg.beta2, eps=cfg.adam_eps),
            optax.scale_by_learning_rate(_scaled(multiplier)),
        )

    tx = optax.multi_transform(
        transforms={
            "matrix": matrix_tx,
            "embedding": _adamw_tx(cfg.embedding_lr_multiplier),
            "unembedding": _adamw_tx(cfg.unembedding_lr_multiplier),
            "scalar": _adamw_tx(cfg.scalar_lr_multiplier),
        },
        param_labels=param_labels if param_labels is not None else _label_params,
    )
    if cfg.clip_global_norm > 0:
        return optax.chain(optax.clip_by_global_norm(cfg.clip_global_norm), tx)
    return tx


Muon = Optimizer(MuonConfig, muon)
