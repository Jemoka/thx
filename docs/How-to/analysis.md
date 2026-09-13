# Analysis System

You know how we had `trainer.find()` and `trainer.debug()` in [time travel debugging](../Tutorials/inspecting-runs.md#time-travel-debugging)? 
What if you can run those *during* training, and also make pretty plots about it? That'd be pretty sick! 

## Make an analysis job

The analysis job follows a very similar API as `trainer.debug()`. I would [look at those docs](../Tutorials/inspecting-runs.md#time-travel-debugging) before you move on here because this would be `+\infty` more sense. When you are back, read the following basic example for analysis:

```python
from theseus.analysis import AnalysisBase
from theseus.experiments.models.gpt import PretrainGPT
from theseus.model.attention.base import SelfAttention
from theseus.registry import analysis, job

@analysis("my/gpt/activations")
class Activations(AnalysisBase, PretrainGPT):
    TARGET = SelfAttention
    NAME = "activations"

    def select(self, paths):
        return paths[0]  # First discovered attention module.

    def analyze(self, layer, inputs):
        return inputs.x[0, :64, :128]  # First sample; crop tokens/features.

    def plot(self, values):
        return self.heatmap(values, xlabel="Feature", ylabel="Token")
```

??? note "What would this look like using the `debug` api?"
    Not exactly the same, as we will see, but roughly:

    ```python
    def activations_scope(j) -> Figure:
        attentions = j.find(SelfAttention)
        layer, inputs = j.debug(attentions[0])
        values = inputs.x[0, :64, :128]
        return sns.heatmap(values)
    ```

Here, we are asking for a copped example of the **input** to the first `SelfAttention` layer, and then plotting it as a heatmap. 
There are three methods describe what to inspect and how to display it:

- `select(paths)`: chooses from discovered `TARGET` calls; path are keyed by strings like `blocks_0/attn`
- `analyze(layer, inputs)`: gives you the debugger `layer` and its captured input `inputs`
- `plot(values)`: takes whatever you returned from `analyze` on **Rank 0** and then does some transformations with it

Unlike the notebook debugger API, `analyze()` runs under JIT, so you should probably not do something that will make the hardworking compiler engineers at Google unhappy. Returning a matplotlib figure in `plot()` will save a PDF, and returning a dictionary with stuff in it will save a JSON.

## What's up with `.heatmap()`
I have font disease so I have very specific opinions about how plots should look like. You don't need to use these opinions and if you don't have this afliction you can just use seaborn yourself and return the `Figure` in `def plot(...)`. But, for convenience, the following font-disease approved plotting API with identical signatures as Seaborn equivalents is available to you:

- `self.heatmap()`: heatmaps
- `self.bar`: bar charts
- `self.line`: line graphs
- `self.scatter`: scatter plots

Isn't it interesting that English has a different word for each type of plot or am I tripping?

## Put it inline during training

You can in fact ask to do analyses during validation by adding it the `ANALYSIS` field of a Trainer:

```python
@job("my/gpt/train-with-analysis")
class GPTWithAnalysis(PretrainGPT):
    ANALYSIS = [Activations]
```
## Run on a saved checkpoint

You can also use the amazing time travel API to attach an evaluation to a historical checkpoint as if it had happened during training:

```python
from my_analysis import Activations
from theseus.quick import quick

with quick("./results") as q:
    q.build(Activations, "inspect")
    q.find().spec(run="train-gpt").checkpoint().latest().resume()
    analysis_job = q.create()
    analysis_job()
```

