# Adding a Learning Rate Schedule

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

A schedule pairs a configuration dataclass with a factory that returns an Optax
learning-rate function. Assign the pair to your trainer's `SCHEDULE`.
`BaseTrainer` defaults to `SCHEDULE = None`, which uses a constant
`optimization/lr`.

---

## Define the config and factory

The factory takes `(total_steps, cfg)` and returns an `optax.Schedule`: a
JAX-compatible callable mapping an optimizer update count to a scalar learning
rate. `total_steps` is the trainer's total training-step budget.

```python
# my_schedule.py
from dataclasses import dataclass

import optax

from theseus.config import field
from theseus.training.schedules import Schedule


@dataclass
class LinearDecayConfig:
    lr: float = field("optimization/lr", default=3e-4)
    final_lr_frac: float = field("optimization/linear/final_lr_frac", default=0.1)


def linear_decay(total_steps: int, cfg: LinearDecayConfig) -> optax.Schedule:
    return optax.linear_schedule(
        init_value=cfg.lr,
        end_value=cfg.lr * cfg.final_lr_frac,
        transition_steps=max(total_steps, 1),
    )


LinearDecay = Schedule(LinearDecayConfig, linear_decay)
```

This starts at `lr` and reaches `lr * final_lr_frac` at `total_steps`, then stays
there. Optax handles the traced step value; avoid Python conditionals on that
value in a custom schedule. For schedules with multiple phases, compute phase
lengths in the factory and handle zero-length phases explicitly.

---

## Wire it into a trainer

Import the pair and set `SCHEDULE` on your existing experiment trainer:

```python
from my_schedule import LinearDecay


class LinearDecayTrainer(MyTrainer):
    SCHEDULE = LinearDecay
```

Here `MyTrainer` already declares its model, configuration, and data. The trainer
collects `LinearDecayConfig` automatically, builds the schedule with
`linear_decay(self.total_steps, configure(LinearDecayConfig))`, and caches it.
The selected optimizer receives that same callable.

The fields appear in the generated configuration:

```yaml
config:
  optimization:
    lr: 0.0003
    linear:
      final_lr_frac: 0.1
```

Set `SCHEDULE = None` to return to a constant learning rate; the custom schedule's
schema is then no longer collected. No registry decorator or YAML schedule-name
selector is needed.

Your implementation can stay in an external module imported by the experiment.
For an in-repository implementation, put it in `theseus/training/schedules/`
and export the pair, factory, and config from that package's `__init__.py`
(including `__all__`), following `WSD`.

---

## Next step

See [Adding an Optimizer](optimizer.md) for the factory that consumes the
schedule, or the [Config System](../../../Design/config.md) for how the schemas
become YAML fields.
