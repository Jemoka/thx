# Extend a Transformer Block

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Inherit `theseus.model.block.block.Block` to change the sublayers or residual
connections within one transformer layer. Its input and output are `[B, T, C]`.

The base computation is:

```python
x_attn = x + self.attn(self.ln_1(x), **kwargs)
x_out = x_attn + self.mlp(self.ln_2(x_attn))
```

| Override | Use it for |
|---|---|
| [`setup()`](#replace-a-child) | Construct `ln_1`, `attn`, `ln_2`, and `mlp` with `configure()` |
| [`components()`](#replace-a-child) | Declare the configurable classes you actually construct |
| [`__call__(x, **kwargs)`](#change-residual-computation) | Change residual ordering or sublayer computation |
| [`flops(seq)`](#update-the-flop-estimate) | Account for added or replaced computation |

The default `setup()` chooses `RopeAttention` when `architecture/rope` is true,
otherwise `SelfAttention`. It uses `LayerNorm` and `MLP` for the other children.
When replacing attention, pass through `padding_mask` and `deterministic`.
`MLP` accepts `deterministic`, but not arbitrary attention arguments.

Install your block in a [GPT subclass](gpt.md): update its `setup()` to construct
that block and its `components()` to declare it. Declaring a new block class
alone does not change the model.

## Replace a child

Keep the inherited setup, then replace the feed-forward layer. The example uses
a named replacement to avoid Flax child-name collisions.

```python
from theseus.config import configure
from theseus.model.block import Block
from theseus.model.layers import MLP

class UnscaledMLP(MLP):
    layer_scaling: bool = False

class CustomMLPBlock(Block):
    @classmethod
    def components(cls):
        return [*super().components(), UnscaledMLP]

    def setup(self):
        super().setup()
        self.mlp = configure(UnscaledMLP, name="custom_mlp")
```

**Add the replacement to `components()`** for configuration discovery. Keep the
parent's entries because `super().setup()` still uses them. The new name changes
its checkpoint parameter path. Use this same pattern for a custom attention class,
then [install the block in GPT](gpt.md#replace-the-block).

## Change residual computation

Scale the block's update while keeping its incoming residual unchanged.

```python
class HalfUpdateBlock(Block):
    def __call__(self, x, **kwargs):
        output = super().__call__(x, **kwargs)
        return x + 0.5 * (output - x)
```

## Update the FLOP estimate

For a block that calls attention twice, count it twice. This is an example pair
of computation and accounting overrides; it uses the two inherited normalizers.

```python
class TwiceAttentionBlock(Block):
    def __call__(self, x, **kwargs):
        x = x + self.attn(self.ln_1(x), **kwargs)
        x = x + self.attn(self.ln_1(x), **kwargs)
        return x + self.mlp(self.ln_2(x), deterministic=kwargs.get("deterministic", False))

    def flops(self, seq):
        return 2 * self.attn.flops(seq) + self.mlp.flops(seq)
```
