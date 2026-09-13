# Extend Attention

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Inherit `theseus.model.attention.base.SelfAttention` and override the stage you
need. Its `__call__()` handles projection, positions, KV caching, masking,
attention, and output projection.

```python
from theseus.config import field
from theseus.model.attention.base import SelfAttention

class ScaledQueries(SelfAttention):
    query_scale: float = field("architecture/query_scale", default=1.0)

    def preprocess_qkv(self, q, k, v, **kwargs):
        return q * self.query_scale, k, v
```

This changes queries before caching and attention. Q/K/V use `[B, T, H, D]`:
batch, tokens, heads, and head width.

| Override | Contract |
|---|---|
| [`_project_inner(x)`](#change-projections) | Project `[B, T, C]` into `(q, k, v)` |
| [`preprocess_qkv(q, k, v, **kwargs)`](#transform-qkv) | Transform Q/K/V; return the same three arrays |
| [`build_mask(t, padding_mask, **kwargs)`](#restrict-the-mask) | Return a boolean attention mask or `None` |
| [`attn(q, k, v, mask=None, **kwargs)`](#change-attention) | Return attention output `[B, T_query, H, D]` |
| [`postprocess_attn(y, padding_mask, deterministic, **kwargs)`](#postprocess-heads) | Preserve head-shaped output; base applies padding and dropout |
| [`output_proj(y)`](#change-output-projection) | Project flattened `[B, T, C]` output |

Keep the inherited `__call__()` to retain KV-cache handling. Cache updates happen
after `preprocess_qkv`; positional transforms should use `kwargs["positions"]`,
which accounts for cached decoding. A custom mask must also handle the
`_cache_index` keyword. `None` selects causal attention in the base `attn()`.

The example extends plain attention, so it does not add RoPE. Install it through
a [custom block](block.md), updating both `setup()` and `components()`.

The examples below are independent `SelfAttention` subclasses. Install one in
your block's `setup()` **and add it to the block's `components()`** so its
configuration is discovered. Keep inherited components if you call the inherited
setup. For the surrounding model wiring, see [replacing GPT's block](gpt.md#replace-the-block).

## Change projections

Tie the projected keys to the queries while retaining the base projection shapes.
This example changes the computation, not the allocated projection parameters.

```python
class TiedQueriesAndKeys(SelfAttention):
    def _project_inner(self, x):
        q, k, v = super()._project_inner(x)
        return q, q, v
```

## Transform Q/K/V

Normalize queries and keys before they enter attention or the KV cache.

```python
import jax
import jax.numpy as jnp

class UnitQueriesAndKeys(SelfAttention):
    def preprocess_qkv(self, q, k, v, **kwargs):
        def normalize(x):
            x32 = x.astype(jnp.float32)
            norm = jax.lax.rsqrt(jnp.sum(x32 ** 2, axis=-1, keepdims=True) + 1e-6)
            return (x32 * norm).astype(x.dtype)
        return normalize(q), normalize(k), v
```

## Restrict the mask

Keep only the four most recent keys while retaining the base causal, padding,
and cache masks. `True` means a key is visible.

```python
class LocalAttention(SelfAttention):
    def build_mask(self, t, padding_mask, **kwargs):
        mask = super().build_mask(t, padding_mask, **kwargs)
        if mask is None:
            mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))[None, None]
        cache_index = kwargs.get("_cache_index")
        query_positions = (
            jnp.arange(t) if cache_index is None else jnp.atleast_1d(cache_index - 1)
        )
        recent = jnp.arange(t)[None, :] > query_positions[:, None] - 4
        return mask & recent[None, None]
```

The cached path uses the position after the single-token cache update, rather
than treating the cache's allocated length as the current query position.

## Change attention

Change the query scale at the attention operation while keeping the inherited
backend and mask handling. KV values in the cache are unchanged.

```python
class CoolerAttention(SelfAttention):
    def attn(self, q, k, v, mask=None, **kwargs):
        return super().attn(q * 0.5, k, v, mask=mask, **kwargs)
```

## Postprocess heads

Zero the first head after retaining the base padding and dropout behavior.
The output still has shape `[B, T, H, D]`.

```python
class DropFirstHead(SelfAttention):
    def postprocess_attn(self, y, padding_mask, deterministic, **kwargs):
        y = super().postprocess_attn(y, padding_mask, deterministic, **kwargs)
        return y.at[:, :, 0, :].set(0)
```

## Change output projection

The head dimension has already been flattened here. Reuse the projection and
scale the resulting `[B, T, C]` tensor.

```python
class ScaledAttentionOutput(SelfAttention):
    def output_proj(self, y):
        return 0.5 * super().output_proj(y)
```
