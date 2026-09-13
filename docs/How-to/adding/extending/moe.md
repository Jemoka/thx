# Extend a Mixture of Experts

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Inherit `theseus.model.moe.base.MoE` to change expert implementations or routing
while retaining fixed-capacity expert packing and output combination.

```python
from theseus.model.moe.base import MoE

class UniformSelectedExperts(MoE):
    def _select_experts(self, router_logits):
        weights, indices = super()._select_experts(router_logits)
        return weights * 0 + 1.0 / weights.shape[-1], indices
```

This keeps the base top-k selection but weights the selected experts equally.
The returned weights and indices must have matching `[tokens, selected_experts]`
shapes.

| Override | Use it for |
|---|---|
| [`_select_experts(router_logits)`](#change-routing) | Return routing weights and expert indices |
| [`_expert_cls()`](#replace-the-experts) | Choose the module vmapped across experts |
| [`_expert_kwargs()`](#replace-the-experts) | Supply extra keyword arguments to `configure()` for each expert |
| [`_router_kernel_init()`](#change-router-initialization) | Change router initialization |
| [`_router_partitioning()`](#set-router-axes) | Change router-kernel logical axes |
| [`components()`](#replace-the-experts) | Declare the configuration of replacement experts |

The base uses a private MLP variant with an unsharded intermediate axis because
the leading expert axis is already sharded. Preserve compatible logical axes
when replacing experts. Capacity limits still apply when changing routing.

Construct your MoE in a custom block's `setup()` and declare it in `components()`.
The containing block must call it with the arguments its forward method accepts.

## Change routing

The `UniformSelectedExperts` example above overrides `_select_experts()`: it keeps
the base indices and replaces their weights. To keep routing probabilities but
reverse the chosen expert IDs instead:

```python
class ReversedExpertIds(MoE):
    def _select_experts(self, router_logits):
        weights, indices = super()._select_experts(router_logits)
        return weights, self.num_experts - 1 - indices
```

## Replace the experts

Select a public `MLP` subclass, but keep its intermediate axis replicated to avoid
conflicting with the expert axis. `setup()` in `MoE` handles vectorizing it.

```python
from theseus.config import field
from theseus.model.layers import MLP
from theseus.model.axes import Axes

class ExpertMLP(MLP):
    width: int = field("architecture/moe/expert_width", default=256)

    def setup(self):
        self.c_fc = self._make_dense(self.width, (Axes.N_EMBD.value, None), 0.02)
        self.c_proj = self._make_dense(self.n_embd, (None, Axes.N_EMBD.value), 0.02)

    def flops(self, seq):
        return float(12 * seq * self.n_embd * self.width)

class CustomExperts(MoE):
    @classmethod
    def components(cls):
        return [ExpertMLP]

    def _expert_cls(self):
        return ExpertMLP

    def _expert_kwargs(self):
        return {"bias": False}
```

`components()` replaces the original expert schema with the new one, exposing
`architecture/moe/expert_width`. `_expert_kwargs()` supplies per-instance overrides
when the base calls `configure()`; here each expert has no biases.

## Change router initialization

Return an initializer, not an initialized parameter array.

```python
import flax.linen as nn

class SmallRouter(MoE):
    def _router_kernel_init(self):
        return nn.initializers.normal(stddev=0.005)
```

## Set router axes

Provide logical axes in input-feature / expert order. Actual partitioning depends
on the containing model's sharding plan.

```python
class NamedRouterAxes(MoE):
    def _router_partitioning(self):
        return (Axes.N_EMBD.value, Axes.N_EXPERT.value)
```
