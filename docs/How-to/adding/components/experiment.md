# Adding an experiment

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

An experiment declares a model, configuration, datasets, evaluations, schedule,
and optimizer as classes or configured components:

```python
from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.training.flywheel.strategy import Sampling
from theseus.training.schedules import WSD
from theseus.training.optimizers import AdamW
from theseus.data.datasets import FineWeb
from theseus.model.models import GPT
from theseus.registry import job

@job("my-gpt/train/pretrain")
class MyTrainer(BaseTrainer[BaseTrainerConfig, GPT]):
    MODEL = GPT
    CONFIG = BaseTrainerConfig
    DATASET = Sampling(FineWeb, 1.0, "pmd")
    EVALUATION = []
    ANALYSIS = []
    SCHEDULE = WSD
    OPTIMIZER = AdamW
```

Import the module to register the job. Built-in experiments are imported from
`theseus.experiments`; external classes can also be passed directly to a
`Combobulator` or `quick` session.

Save the example as `my_experiment.py`. Generate and edit the combined schema with `theseus configure`, then execute it:

```bash
uv run theseus --import ./my_experiment.py configure my-gpt/train/pretrain train.yaml
uv run theseus run experiment train.yaml ./results training.per_device_batch_size=1
```

Extend `BaseTrainerConfig` with dataclass fields when a trainer needs extra
options. `config()` gathers the declared components' schemas, and construction
hydrates `CONFIG` into `self.args`. A schedule is `Schedule(ConfigType, factory)`;
an optimizer is `Optimizer(ConfigType, factory)`. The `schedule` and `optimizer`
properties cache their constructed values. `SCHEDULE = None` selects a constant
learning rate.

## State lifecycle

Construction configures topology, model, batching, and lazy data loaders without
allocating training state. `setup()` is idempotent and initializes or restores
state. `__call__()` performs host synchronization, setup, execution, and cleanup.

Override `_make_state(params)` to add fields to a Flax training state; keep it
pure so initialization, shape templates, and partial checkpoint surgery use the
same representation. `initialize()` creates the state on its target shards.
`template` describes the destination state; `surgery(partial)` initializes missing
leaves, and `apply(state, metadata)` installs a restored state.

`DATASET` must be declared by the trainer; `BaseTrainer` supplies no default.
It accepts a dataset class, `Sampling`, or a list of either. Use `[]` when the
job supplies batches itself. Loader batches
are keyed by the shared `node`: repeated `batch()` calls return the same data
until the job ticks. Resuming advances to the successor of the saved node.

The specialized trainers in `theseus.training` support DPO,
two-stage KL regularization, and LoRA. KL and LoRA use the normal static `DATASET`
declaration; their token budgets control objective or adapter transitions.
Benchmark trainers declare curriculum mixtures through `STAGES`. HuggingFace
backbone trainers use `architecture/backbone/implementation` and `weights` to
load the initial architecture and parameters.

Declare `EVALUATION` and `ANALYSIS` as lists of component classes. Evaluation and
analysis views borrow the live trainer's state and node. Plots belong to analysis
jobs; see [Adding an analysis job](analysis-job.md).
