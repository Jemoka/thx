# Extend a Trainer

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Inherit `theseus.training.base.BaseTrainer` to define what a training job uses.
For a small change to an existing experiment, inherit that trainer instead and
replace only the declaration you need.

```python
from theseus.data.datasets import FineWeb
from theseus.model.models import GPT
from theseus.registry import job
from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.training.flywheel.strategy import Sampling
from theseus.training.optimizers import AdamW
from theseus.training.schedules import WSD

@job("my/gpt/train")
class MyTrainer(BaseTrainer):
    CONFIG = BaseTrainerConfig
    MODEL = GPT
    DATASET = Sampling(FineWeb, 1.0, "pmd")
    OPTIMIZER = AdamW
    SCHEDULE = WSD
    EVALUATION = []
    ANALYSIS = []
```

Import the file to register the job. The data must already be prepared in your
root; see [Adding an experiment](../components/experiment.md) for running it.

## Choose components

| Declaration | What to supply |
|---|---|
| `MODEL` | A model class; its `gather()` supplies model and child configuration |
| `CONFIG` | A configuration dataclass, normally extending `BaseTrainerConfig` |
| `DATASET` | A dataset class, `Sampling`, or a list of either; `[]` for custom batching |
| `OPTIMIZER` | An `Optimizer` factory container; defaults to `AdamW` |
| `SCHEDULE` | A `Schedule` factory container; `None` uses constant learning rate |
| `EVALUATION` | Evaluation classes; see [evaluation setup](../../evaluation.md) |
| `ANALYSIS` | Analysis classes; see [analysis setup](../../analysis.md) |

These declarations contribute their configuration schemas automatically. Use
`field()` in your `CONFIG` dataclass for additional settings and read them through
`self.args`. Optimizers and schedules wrap a config type and a factory function;
you do not need to subclass them.

## Change the computation

Keep the training loop unless your experiment needs a different lifecycle.
Override the smallest method that implements your change.

| Override | Contract |
|---|---|
| [`batch(slice="train")`](#transform-a-batch) | Retrieve host-side batch data; base returns `x`, `y`, and `padding_mask` |
| [`forward(state, params, batch, key=None, deterministic=False, intermediates=False)`](#extend-the-forward-pass) | Static method returning `(logits, loss, metadata)` |
| [`train_step(...)`](#extend-step-metadata) / [`val_step(...)`](#extend-step-metadata) | Classmethods for changing update or validation computation; preserve the reference signatures |
| [`trace(state, batch, key, *, sharding)`](#change-the-inspection-trace) | Classmethod defining the pure model call used by debugging and analysis |

The default `forward()` calls the model with token IDs, targets, a padding mask,
and dropout controls. If your model changes that interface, adapt `forward()`
too. A custom `trace()` must use its supplied state, batch, and key rather than
fetching data or reading mutable trainer state.

The examples below extend `MyTrainer` from the first example. Register the subclass
with `@job(...)` when you want to select it by name.

## Transform a batch

Copy the returned mapping before changing it; the base batch can be cached on the
current node. This example removes the first token of each sequence from the loss.

```python
class IgnoreFirstTarget(MyTrainer):
    def batch(self, slice="train"):
        batch = dict(super().batch(slice))
        batch["y"] = batch["y"].copy()
        batch["y"][..., 0] = -1
        return batch
```

## Extend the forward pass

Add a scalar measurement without changing the loss or model interface.

```python
import jax.numpy as jnp

class LogitRMS(MyTrainer):
    @staticmethod
    def forward(state, params, batch, key=None, deterministic=False, intermediates=False):
        logits, loss, meta = MyTrainer.forward(
            state, params, batch, key=key,
            deterministic=deterministic, intermediates=intermediates,
        )
        rms = jnp.sqrt(jnp.mean(logits.astype(jnp.float32) ** 2))
        return logits, loss, {**meta, "logit_rms": rms}
```

## Extend step metadata

Delegate the update and aggregation to the base, then add metadata. `**kwargs`
forwards the base's keyword-only sharding and training options.

```python
class StepMetrics(MyTrainer):
    @classmethod
    def train_step(cls, state, batch, key, accumulate_steps, **kwargs):
        state, loss, meta, grad_norm = super().train_step(
            state, batch, key, accumulate_steps, **kwargs,
        )
        return state, loss, {**meta, "grad_norm_squared": grad_norm ** 2}, grad_norm

    @classmethod
    def val_step(cls, state, batch, *, sharding):
        loss_sum, count, meta = super().val_step(state, batch, sharding=sharding)
        return loss_sum, count, {**meta, "valid_tokens": count}
```

## Change the inspection trace

For deterministic inspection, retain the base's sharding context but call the
forward pass with dropout disabled. This controls debugging and analysis, not the
training update.

```python
import jax
import flax.linen as nn

class DeterministicInspection(MyTrainer):
    @classmethod
    def trace(cls, state, batch, key, *, sharding):
        with (
            jax.sharding.use_abstract_mesh(sharding.mesh.abstract_mesh),
            nn.logical_axis_rules(sharding.parameter_fwdbwd_mapping),
        ):
            return cls.forward(state, state.params, batch, key=key, deterministic=True)
```

For new model children, update the model's `components()` list; the trainer finds
them through `MODEL.gather()`. Trainer declarations such as `EVALUATION` and
`OPTIMIZER` contribute their schemas through `BaseTrainer.config()` instead.

## Estimated training FLOPs

`train/flops` records cumulative estimated training arithmetic alongside
`train/tokens`, including the terminal validation event. Each completed update
adds `global_batch * model.flops(context) + optimizer.flops_per_param * parameters`.
The model contract already includes forward and backward work. Accumulation and
replicas do not multiply the global estimate; optimizer work is counted once.

This estimate excludes evaluation, communication, compilation, rematerialization,
and operations omitted by a model's FLOP implementation. It does not require a
known device peak (unlike MFU). If a model has no estimate, a warning identifies
its fallback approximation of `6 * parameters * tokens`; unknown optimizer work
is excluded with a warning. Checkpoint metadata preserves accumulated work across
resumes and phase changes. Older checkpoints without this metadata use completed
steps times the current per-step estimate, with a warning that the historical
configuration is unknown. MFU setup remains in `run()` and refreshes only when
the training state changes type.
