# Extend an MLP

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Inherit `theseus.model.layers.mlp.MLP` for a feed-forward layer. The base applies
an input projection, GELU, an output projection, and dropout, preserving
`[B, T, C]` shape.

```python
from theseus.model.layers.mlp import MLP

class UnscaledMLP(MLP):
    layer_scaling: bool = False
```

This keeps the computation but disables depth scaling of the output projection's
initializer. To change the activation or add a gate, override `__call__()` and,
when adding parameters, `setup()`.

| Override | Use it for |
|---|---|
| [`setup()`](#add-a-child-layer) | Construct projections; base names them `c_fc` and `c_proj` |
| [`__call__(x, deterministic=False)`](#change-the-activation) | Change the feed-forward computation; preserve feature width |
| [`_intermediate_features`](#change-hidden-width) | Change hidden width; base uses configured `intermediate_size` or `4 * n_embd` |
| [`_make_dense(features, sharding_axes, stddev)`](#change-projection-initialization) | Change how projections are constructed |
| [`components()`](#add-a-child-layer) | Declare additional configurable children |
| [`flops(seq)`](#count-projection-work) | Update the estimate when changing projection count or width |

`_make_dense()` supplies the configured dtypes, bias, initialization, and logical
parameter axes. Reuse it when those conventions still apply. Preserve dropout's
`deterministic` control in a custom forward method.

Install the class as `self.mlp` in a [block](block.md) and include it in that
block's `components()`.

These examples use the `MLP` import above and keep input/output shape `[B, T, C]`.

## Change the activation

Reuse the inherited projections, replace GELU with SiLU, and preserve dropout.

```python
import jax
import flax.linen as nn

class SiLUMLP(MLP):
    @nn.compact
    def __call__(self, x, deterministic=False):
        x = self.c_proj(jax.nn.silu(self.c_fc(x)))
        return nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
```

## Change hidden width

The base setup and FLOP calculation both consult this property.

```python
class DoubleWidthMLP(MLP):
    @property
    def _intermediate_features(self):
        return 2 * self.n_embd
```

## Change projection initialization

Halve both projection initializers while preserving dtype and sharding conventions.

```python
class SmallInitMLP(MLP):
    def _make_dense(self, features, sharding_axes, stddev):
        return super()._make_dense(features, sharding_axes, stddev * 0.5)
```

## Add a child layer

**Declare new configurable children in `components()`** as well as constructing
them in `setup()`. Here the child is a normalization layer, not an extra projection.

```python
from theseus.config import configure
from theseus.model.layers import LayerNorm

class NormalizedMLP(MLP):
    @classmethod
    def components(cls):
        return [*super().components(), LayerNorm]

    def setup(self):
        super().setup()
        self.input_norm = configure(LayerNorm)

    def __call__(self, x, deterministic=False):
        return super().__call__(self.input_norm(x), deterministic=deterministic)
```

## Count projection work

If your forward method adds another call to the same MLP, its arithmetic doubles
even though it reuses parameters.

```python
class TwiceMLP(MLP):
    def __call__(self, x, deterministic=False):
        x = super().__call__(x, deterministic=deterministic)
        return super().__call__(x, deterministic=deterministic)

    def flops(self, seq):
        return 2 * super().flops(seq)
```
