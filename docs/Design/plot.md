# Analysis and plotting

Plots belong to analysis jobs. Models expose their forward computation; they do
not declare plotting callbacks. Compose an analysis with the trainer whose model
and configuration it understands:

```python
from theseus.analysis.attention import AttentionHeatmapAnalysis
from theseus.registry import analysis

@analysis("my-gpt/analyze/attention")
class AnalyzeAttention(AttentionHeatmapAnalysis, MyGPTTrainer):
    pass
```

Put the analysis first so its `run()` takes precedence. `AnalysisBase` checks this
ordering. Its `CONFIG` combines the trainer and analysis schemas through normal
Python dataclass inheritance.

Run it as an isolated job, with a checkpoint node passed as `base=` to restore
weights through the normal model-surgery path. Without a base, it analyzes newly
initialized weights. `select(paths)` chooses a target module; `analyze(layer,
inputs)` runs inside a compiled pure trainer trace and returns a numerical
PyTree. The framework gathers that result and synchronously calls the required
`plot(results)` on host zero. It returns a CPU Matplotlib figure, a JSON value,
or `None`. `NAME` determines the artifact filename. Host zero saves PDF or
JSON and uploads it to Comet when configured.

For trainer integration, declare `ANALYSIS = [MyAnalysis]`. Each analysis borrows
the live trainer's model, state, and node without initializing another state.
`training/analyze` controls whether these analyses run at validation cadence.

The `line`, `scatter`, `heatmap`, and `bar` helpers in `theseus.analysis.plots`
create figures with scoped Seaborn styling. They accept CPU data. Analysis authors
crop or reduce arrays inside `analyze()` before the framework gathers them and
calls `plot()`. All hosts must agree on collective order. The notebook debugger
remains eager and is not used by scheduled analysis. Both paths share the pure
`trace(state, batch, key, *, sharding)` execution boundary; data retrieval and
placement stay outside it, using the existing trainer batch APIs.
