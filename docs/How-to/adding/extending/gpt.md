# Extend GPT

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Inherit `theseus.model.models.base.GPT` to change part of the language model
while keeping its embeddings, transformer stack, and loss implementation.

```python
from theseus.config import field
from theseus.model.models.base import GPT

class ScaledGPT(GPT):
    logit_scale: float = field("architecture/logit_scale", default=1.0)

    def unembed(self, x):
        return super().unembed(x) * self.logit_scale
```

The inherited `__call__()` runs `embed → decode → unembed → loss`. In this
example, `super().unembed()` supplies final normalization and the tied embedding
projection; the override scales its logits.

| Override | Contract |
|---|---|
| [`embed(idx, deterministic=False, **kwargs)`](#scale-embeddings) | Token IDs `[B, T]` → features `[B, T, C]` |
| [`decode(x, padding_mask=None, deterministic=False, **kwargs)`](#change-layer-execution) | Features → features; runs the blocks |
| [`unembed(x)`](#scale-logits) | Features → logits `[B, T, vocab_size]` |
| [`loss(logits, targets)`](#change-the-loss) | Return a scalar loss; the base ignores targets equal to `-1` |
| [`setup()`](#replace-the-block) and [`components()`](#replace-the-block) | Construct and declare different blocks or normalization layers |

`__call__()` returns `(logits, loss)`, with `loss=None` when no targets are given.
Preserve that contract for the default trainer. Forward padding masks and
`deterministic` when changing layer execution. Call `measure()` in a custom
`decode()` if you want to retain residual diagnostics.

Set your trainer's `MODEL` to your subclass; models have no registration decorator.
The examples below override one stage at a time.

## Scale embeddings

Keep token lookup, positional embeddings, and dropout; scale their result.
The following examples use the `GPT` import above.

```python
class ScaledEmbeddings(GPT):
    def embed(self, idx, deterministic=False, **kwargs):
        return 0.5 * super().embed(idx, deterministic=deterministic, **kwargs)
```

## Change layer execution

Reverse the order of the existing blocks. Preserve mask/dropout arguments and
`measure()` so residual diagnostics still work.

```python
class ReversedGPT(GPT):
    def decode(self, x, padding_mask=None, deterministic=False, **kwargs):
        for depth, block in enumerate(reversed(self.blocks)):
            residual = x
            x = block(x, padding_mask=padding_mask, deterministic=deterministic)
            self.measure(x, residual, depth, padding_mask)
        return x
```

## Scale logits

`unembed()` includes final normalization and the tied output projection. This
changes the logits used both for prediction and for the inherited loss.

```python
class HalfLogits(GPT):
    def unembed(self, x):
        return 0.5 * super().unembed(x)
```

## Change the loss

Add a logit penalty, excluding ignored target positions just as the base loss does.

```python
import jax.numpy as jnp

class PenalizedLogits(GPT):
    def loss(self, logits, targets):
        valid = targets != -1
        penalty = jnp.mean(logits.astype(jnp.float32) ** 2, axis=-1)
        penalty = jnp.sum(penalty * valid) / jnp.maximum(valid.sum(), 1)
        return super().loss(logits, targets) + 1e-4 * penalty
```

## Replace the block

You can keep `decode()`: it iterates over `self.blocks`. Base `GPT` has no `BLOCK`
attribute or `new_block()` hook; replace the list in `setup()` instead.

```python
from theseus.config import configure, field
from theseus.model.block import Block

class ScaledBlock(Block):
    scale: float = field("architecture/block_scale", default=1.0)

    def __call__(self, x, **kwargs):
        return super().__call__(x, **kwargs) * self.scale

class CustomBlocksGPT(GPT):
    @classmethod
    def components(cls):
        return [*super().components(), ScaledBlock]

    def setup(self):
        super().setup()  # Keep embeddings, dropout, and final normalization.
        self.blocks = [
            configure(ScaledBlock, name=f"custom_blocks_{i}")
            for i in range(self.n_layers)
        ]
```

The explicit names avoid colliding with the blocks assigned by `super().setup()`.
They also change checkpoint parameter paths; restoring an existing GPT checkpoint
needs matching names or checkpoint surgery.

**Add new configurable children to `components()`**, the component list used for
configuration discovery. Here it makes `architecture/block_scale` available.
Preserve the inherited list when you call `super().setup()`, because that setup
still configures the original children. There is no separate global component
registry, and listing a class does not install it: `setup()` does that.
