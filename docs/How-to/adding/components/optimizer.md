# Adding an Optimizer

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

An optimizer pairs a configuration dataclass with a factory that builds an Optax
transformation. Assign that pair to your trainer's `OPTIMIZER`; `BaseTrainer`
defaults to `AdamW`.

---

## Define the config and factory

The factory takes `(lr, cfg)` and returns an `optax.GradientTransformation`.
`lr` may be an Optax schedule or a scalar; the trainer supplies its cached
learning-rate schedule. The transformation must support Optax's `init(params)`
and `update(grads, state, params)` interface.

```python
# my_optimizer.py
from dataclasses import dataclass

import optax

from theseus.config import field
from theseus.training.optimizers import Optimizer


@dataclass
class SGDConfig:
    momentum: float = field("optimization/sgd/momentum", default=0.9)


def sgd(lr: optax.Schedule | float, cfg: SGDConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.sgd(learning_rate=lr, momentum=cfg.momentum),
    )


SGD = Optimizer(SGDConfig, sgd)
```

Include clipping in your factory if you want it: the trainer does not add it
around the returned transformation. Keep updates compatible with JAX compilation
and the model's parameter tree.

`Optimizer` also accepts `flops_per_param`, an optional estimate of arithmetic
per trainable parameter per update, including clipping. Leaving it as `None`
makes the trainer's theoretical MFU estimate unavailable.

---

## Wire it into a trainer

Import the pair and set `OPTIMIZER` on an existing trainer that already declares
its model, configuration, and data:

```python
from my_optimizer import SGD


class SGDTrainer(MyTrainer):
    OPTIMIZER = SGD
```

Here `MyTrainer` is your existing experiment trainer. The trainer collects
`SGDConfig` automatically and calls `sgd(self.schedule, configure(SGDConfig))`
once, caching the result. You do not need a registry decorator or a YAML
optimizer-name selector.

The new field appears in the generated configuration:

```yaml
config:
  optimization:
    sgd:
      momentum: 0.9
```

Your implementation can stay in an external module imported by the experiment.
For an in-repository implementation, put it in `theseus/training/optimizers/`
and export the pair, factory, and config from that package's `__init__.py`
(including `__all__`), following `AdamW`.

---

## Next step

Choose or add a [learning-rate schedule](schedule.md). The trainer passes it
to your optimizer automatically.
