# Config System

This document covers `theseus/config.py` — the lightweight, dataclass-driven configuration layer that ties together model architecture, training hyperparameters, and runtime settings into a single OmegaConf YAML.

---

## Overview

| Primitive | What it does |
|---|---|
| `field("path/key", default=...)` | Marks a dataclass field as config-driven |
| `build(*classes)` | Collects all marked fields across a class tree and produces an OmegaConf schema |
| `configuration(cfg)` | Context manager that activates a loaded config |
| `configure(cls)` | Hydrates a dataclass instance from the active config |
| `patch()` | Context manager for temporarily mutating the active config |

---

## `field()` — Declaring Config-Driven Fields

Any dataclass field annotated with `field("path/key")` participates in the config system:

```python
from dataclasses import dataclass
from theseus.config import field
from theseus.model.module import Module

@dataclass
class SelfAttention(Module):
    n_embd: int    = field("architecture/n_embd",   default=2048)
    n_head: int    = field("architecture/n_head",   default=16)
    block_size: int = field("architecture/block_size", default=512)
    dropout: float = field("architecture/dropout",  default=0.0)
```

Under the hood `field()` is a thin wrapper around `dataclasses.field()` that stashes the path string in the field's `metadata` dict under the key `"th_config_field"`. That metadata is the only marker `build()` and `configure()` look for.

The path string uses `/` as a separator and maps directly to a nested YAML structure:
`"architecture/n_embd"` → `architecture: { n_embd: ... }`.

---

## `build()` — Generating the OmegaConf Schema

`build(*classes)` collects config fields from every class you pass in (plus any dataclass-typed sub-fields) and returns a fully-structured OmegaConf config with defaults filled in:

```python
from theseus.config import build

cfg = build(BaseTrainerConfig, GPT)
print(OmegaConf.to_yaml(cfg))
```

Internally this runs through three stages.

### Stage 1 — DFS class expansion (`generate_canonical_config`)

`build()` calls `generate_canonical_config(*classes)`. Before collecting fields it recursively expands any field whose type is itself a dataclass:

```
BaseTrainerConfig
  └─ has field `optimizer_cfg: AdamWConfig`   ← dataclass type → recurse
AdamWConfig
  └─ lr: float = field("optimization/lr", ...)
```

This means you can compose arbitrarily nested config dataclasses and `build()` sees the full flat set of annotated fields.

### Stage 2 — Key union (LUB)

Multiple classes can declare the same path key. This is intentional: `SelfAttention`, `GroupedSelfAttention`, and `GPT` all declare `"architecture/n_embd"` because they all need it at construction time. `generate_canonical_config` deduplicates by computing the **least upper bound type**:

- One class → use that type directly
- Multiple classes, same type → same type
- Multiple classes, different types → `Union[t1, t2, ...]`

For default values, the first non-`None` value wins. If two classes declare the same key with conflicting defaults, `None` is used and the field becomes `???` (mandatory-missing) in the generated YAML.

### Stage 3 — Nested dict construction

`nest_slash_keys()` converts the flat `{"architecture/n_embd": 2048, "training/lr": 3e-4}` dict into a properly nested Python dict:

```python
{
    "architecture": {"n_embd": 2048},
    "training": {"lr": 3e-4},
}
```

This is then handed to `OmegaConf.create()` with `set_struct=True` (no unknown keys allowed), producing the final config object.

### Generated YAML

The CLI command `uv run theseus configure gpt/train/pretrain run.yaml` builds
the schemas for the execution's job classes and saves an execution document.
Component values are nested under `config`; the abbreviated structure is:

```yaml
execution:
  steps:
    - job: gpt/train/pretrain
      base: null
      resume: false
  sharding: {tp: 1, fsdp: false, zero: true}
config:
  architecture:
    n_embd: 2048
    n_head: 16
    block_size: 512
  training:
    batch_size: 512
    per_device_batch_size: 1
    tokens: 1000000000
  optimization:
    lr: 0.0003
```

CLI overrides such as `training.tokens=8192` are relative to `config`.
Execution resources and steps are separate from the component schema.

Users edit this file (or pass `key=value` overrides on the CLI) before running.

---

## `configuration()` / `configure()` — The Context Guard

Config values are activated via a context manager, never via global state that leaks across threads:

```python
from omegaconf import OmegaConf
from theseus.config import configuration, configure

cfg = OmegaConf.load("run.yaml")

with configuration(cfg.config):
    attention = configure(SelfAttention)   # reads architecture.n_embd etc.
    trainer_args = configure(BaseTrainerConfig)
```

`configuration(cfg)` uses a `ContextVar` (`_current_config`) — Python's per-async-task / per-thread context isolation primitive. Setting the config in one coroutine or thread doesn't affect others.

`configure(cls)` calls `hydrate(cls, config)` which:

1. Flattens the OmegaConf object back to `"a/b/c"` → value pairs.
2. Iterates over `fields(cls)`, matching each `th_config_field` key to the flat config.
3. For fields whose type is itself a dataclass, recurses via `hydrate(sub_cls, config)`.
4. For plain `Module` subclasses (Flax), calls `cls(**init_kwargs)` directly.
5. For pure-Python dataclasses, uses `OmegaConf.structured` + `OmegaConf.merge` to get typed coercion.

### Inside a Flax `setup()`

`configure()` is frequently called inside `nn.Module.setup()`:

```python
class GPT(Module):
    def setup(self):
        self.blocks = [configure(TransformerBlock) for _ in range(self.n_layers)]
        self.ln_f   = configure(LayerNorm)
```

This works because `setup()` is called inside `model.init(...)` which is always inside a `with configuration(cfg):` block established by the trainer.

---

## `patch()` — Temporary Config Mutations

`patch()` is a context manager that temporarily disables struct-mode, allowing free-form additions:

```python
from theseus.config import patch

with patch() as cfg:
    cfg.architecture.n_layers = 32
    cfg.architecture.attention_bias = False
    model.init(jax.random.PRNGKey(0), dummy_input)
```

If a `configuration()` context is already active, `patch()` mutates it directly (and re-seals it on exit). If there's no active context, it creates a fresh empty config scoped to the block. This is used by `from_pretrained()` style loaders to inject architecture fields read from a checkpoint into the active config.

---

## End-to-End Flow

`uv run theseus configure gpt/train/pretrain run.yaml` collects the job's
component schemas into a `Combobulation` and serializes its execution steps,
resources, and component config.

`uv run theseus run my-run run.yaml ./output` deserializes that document and
constructs a `DispatchSpec`. Its local `Runner` resolves each step's base node,
activates the component configuration, and constructs and executes the job.
`configure(BaseTrainerConfig)` and `configure(GPT)` read the active component
mapping, not the enclosing execution document.
