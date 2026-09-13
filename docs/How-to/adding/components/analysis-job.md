# Adding an analysis job

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

An analysis is a trainer-backed job that replaces training with inspection. Its
model, configuration, initialization, and checkpoint surgery follow the same
lifecycle as the trainer it assumes.

```python
from dataclasses import dataclass
from theseus.analysis import AnalysisBase
from theseus.experiments.models.gpt import PretrainGPT
from theseus.training.base import BaseTrainerConfig
from theseus.model.attention.base import SelfAttention
from theseus.config import field
from theseus.model.debug import DebugInputs
from theseus.registry import analysis

@dataclass
class InspectConfig(BaseTrainerConfig):
    target: str = field("analysis/target", default="blocks_0/attn")

@analysis("my-model/analyze/activation")
class InspectActivation(AnalysisBase, PretrainGPT):
    CONFIG = InspectConfig
    TARGET = SelfAttention
    NAME = "activation"

    def select(self, paths: list[str]) -> str:
        if self.args.target not in paths:
            raise ValueError(f"Choose a discovered path: {paths}")
        return self.args.target

    def analyze(self, layer, inputs: DebugInputs):
        # Bound the result before the framework gathers it.
        return inputs.x[0, :64, :128]

    def plot(self, results):
        return self.heatmap(results, xlabel="Feature", ylabel="Token")
```

Put the analysis before the trainer in the inheritance list. `AnalysisBase`
checks that `run()` resolves to an analysis. The `@analysis` decorator also
registers the class as a job, so no second decorator is needed.

`CONFIG` is one composed dataclass schema. All its values are available through
`self.args`. Include the trainer fields needed to construct the model even when
the job will only inspect it. `TARGET` identifies candidate module types, and
`select()` chooses an exact discovered path. `analyze()` runs at the selected
forward invocation with a bound Flax module and its original arguments. It must
return a JAX-compatible PyTree of arrays, scalars, or `None`. `plot()` is also
required: it receives gathered CPU results and returns a Matplotlib figure, a
JSON value, or `None`. Both methods may read fixed configuration from `self.args`.
Only `plot()` may render, decode tokens, convert to NumPy, or access mutable
trainer state. Numerical work must use the supplied layer and inputs, not
`self.state` or a separately fetched batch.

`AnalysisBase.run()` saves the artifact under `self.spec.result()` using `NAME`
and uploads it to Comet when configured. There is no separate export path.
The same output is stored as a node artifact with `filename` and `content`
(the exact bytes). Query it with `store.query().node(node).artifact().select()`.

Every host performs abstract discovery and compiled analysis. The framework
gathers the returned PyTree, preserving global array shapes, then calls `plot()`
synchronously on host zero and writes the artifact. Select/crop/reduce results in
`analyze()` before returning them; do not return a full activation history unless
you intend to materialize it on the CPU. All hosts must agree on selections and
collective order.

## Initialized or checkpointed state

Construct with `Analysis(spec)` to analyze newly initialized weights. Pass a
checkpoint node as `base=` to load it through the normal checkpoint and model
surgery path. Invoke the job normally, or let `quick.create()` perform setup:

```python
from theseus.quick import quick

with quick("./results") as q:
    q.build(InspectActivation, "inspect")
    q.find().spec(run="training-run").checkpoint().latest().resume()
    analysis_job = q.create()
    analysis_job()
```

Construction does not allocate training state. Idempotent `setup(resume=False)`
initializes or restores it; the job's `__call__` owns host synchronization and
execution. `resume=True` preserves the saved node and batch, so analysis results
attach to that checkpoint. Training advances once before consuming its next
batch. Use `branch()` to start a new lineage instead.

For integration during training, set `ANALYSIS = [InspectActivation]` on the
trainer. The borrowed analysis shares its trainer's node and observes live state
replacements. It cannot independently set up, tick, or replace that state.
`training/analyze` enables this integration. Each analysis retains its compiled
function across invocations; current state and the newly retrieved batch are
dynamic arguments. The trainer remains responsible for lifecycle and batch
advancement. Calling `batch()` again follows the existing node cache; it does not
implicitly advance the training cursor.

## Select several module calls

`select(paths)` can also return a nonempty list of discovered paths. In that
case `analyze(layers, inputs)` receives two lists in the same order, even for a
one-item selection. The first invocation at every selected path is captured in
one compiled trace. The empty string identifies the root module. Arguments are
only those of the selected module call; no raw batch is injected into them.
Use the supplied bound modules' methods for numerical work. Their scopes and
tracers must not be retained after `analyze()` returns.

## Trainer trace boundary

Trainer implementers may override the pure classmethod
`trace(state, batch, key, *, sharding: ParameterShardingContext)`. State includes
the current parameters and any trainer-specific state; `batch` is the placed
microbatch, and `key` is an explicit RNG. The method must not retrieve data or
read mutable trainer attributes. The default calls the trainer's existing
`forward()` under the supplied parameter execution sharding rules.

Both `find()`/`debug()` and scheduled analysis retrieve data through the existing
`batch()` method and prepare it outside the trace. Debugging executes eagerly;
scheduled analysis discovers paths with `jax.eval_shape` and caches a JIT of
trace plus `analyze()`. Stable shapes, dtypes, and static configuration reuse the
compilation; different shapes or state structures can compile again. Change
configuration by constructing a new analysis instance, not by mutating the
configuration captured by an existing compiled function.

The example's target path is a starting choice; replace it with a path discovered
for your actual model. Give different attached analyses different `NAME` values
so their artifacts do not collide. See [Analysis System](../../analysis.md) for a
a minimal activation plot and inline training example.
