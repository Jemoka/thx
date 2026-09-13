# Extend a Module

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Inherit `theseus.model.module.Module` when building a new model component.
It extends Flax Linen's module with configuration discovery, dtype settings,
FLOP estimates, and a sharding plan.

```python
from theseus.config import field
from theseus.model.module import Module

class Scale(Module):
    factor: float = field("architecture/scale", default=1.0)

    @classmethod
    def components(cls):
        return []  # No configurable child modules.

    def __call__(self, x):
        return x * self.factor
```

`field()` connects an attribute to a configuration key. Construct configurable
children with `configure(Child)` inside `setup()` and list their types in
`components()`. The trainer uses `MODEL.gather()` to collect those schemas.
**Listing a type does not instantiate it or replace an existing child.**

| Override | Use it for |
|---|---|
| [`components()`](#construct-and-declare-children) | Declare configurable child types; return `[]` for a leaf |
| [`setup()`](#construct-and-declare-children) / [`__call__()`](#construct-and-declare-children) | Create parameters and children / compute outputs |
| [`sharding`](#choose-a-sharding-plan) | Supply a `ShardingPlan`; the default uses replicated parameters |
| [`flops(seq)`](#estimate-computation) | Estimate forward + backward FLOPs for one sequence; default is zero |

Use inherited `_param_dtype` and `_activation_dtype` when allocating parameters
and computing activations. For a full language model, start with [GPT](gpt.md)
to retain the trainer's expected input/output contract.

## Construct and declare children

`components()` is the component list used by the configuration system. **Every
new configurable child must appear there**, including a child added by a subclass.
Use `[*super().components(), NewChild]` when keeping the parent's children.

```python
from theseus.config import configure
from theseus.model.layers import MLP

class FeedForward(Module):
    @classmethod
    def components(cls):
        return [MLP]

    def setup(self):
        self.mlp = configure(MLP)

    def __call__(self, x, deterministic=False):
        return self.mlp(x, deterministic=deterministic)
```

This declares the MLP's config, constructs it, and calls it. A plain Python helper
without configurable fields does not need to be listed.

## Choose a sharding plan

Return an empty plan to leave the model's parameters replicated. For partitioned
parameters, match the plan to their logical axes; see [sharding design](../../../Design/sharding.md).

```python
from theseus.base.axis import ShardingPlan

class ReplicatedFeedForward(FeedForward):
    @property
    def sharding(self):
        return ShardingPlan()
```

## Estimate computation

Delegate to the child when the wrapper adds no substantial arithmetic. The module
must be bound so that `setup()` has created `self.mlp`.

```python
class CountedFeedForward(FeedForward):
    def flops(self, seq):
        return self.mlp.flops(seq)
```
