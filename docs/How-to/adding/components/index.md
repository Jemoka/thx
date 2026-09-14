# Adding Components

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

theseus is structured around seven extension points. Choose an implementation, declare its configuration, and import it where you
use it. The folders below are where built-in implementations live; your own
classes can live in an external module. Decorators run on import. Models do not
have a model registry: pass the class through a trainer's `MODEL` attribute.

| What | Where | How |
|---|---|---|
| [Model](model.md) | `theseus/model/models/` | subclass `Module`, set the trainer's `MODEL` |
| [Experiment](experiment.md) | `theseus/experiments/` | `@job("key")` + subclass `BaseTrainer` |
| [Analysis](analysis-job.md) | `theseus/experiments/` or `projects/` | `@analysis("key")` + compose `AnalysisBase` before your trainer in the inheritance list |
| [Dataset](dataset.md) | `theseus/data/datasets/` | `@dataset("key")` + subclass a dataset base |
| [Evaluation](evaluation.md) | `theseus/evaluation/datasets/` | `@evaluation("key")` + subclass the appropriate evaluation strategy |
| [Optimizer](optimizer.md) | `theseus/training/optimizers/` | pair a config schema and factory with `Optimizer`, set the trainer's `OPTIMIZER` |
| [Learning Rate Schedule](schedule.md) | `theseus/training/schedules/` | pair a config schema and factory with `Schedule`, set the trainer's `SCHEDULE` |

For a worked sequence, start with [Making It Your Own](../../../Tutorials/adding.md):
a new dataset, a `Sampling` mixture, an optimizer switch, and then a model change.
