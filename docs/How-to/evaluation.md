# Evaluation System

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Add a task to score during training, or run it against a saved checkpoint.
Evaluations supply their own examples; validation loss uses the trainer's data
mixture. This example measures perplexity on two texts: how well the model
predicts their tokens. **Lower is better.**

## Choose an evaluation strategy

These base classes live in `theseus.evaluation.base`. Choose one for the kind of
output you want to score; follow its link for the methods to implement.

| Base class | What it scores | Direction |
|---|---|---|
| [`RolloutEvaluation`](adding/components/evaluation.md#rollout) | Generated answers, using your answer cleaning and scoring rules | Task-defined; accuracy is higher-is-better |
| [`EncodingEvaluation`](adding/components/evaluation.md#encoding) | Decoded tokenwise argmax predictions from a forward pass | Task-defined |
| [`PerplexityEvaluation`](adding/components/evaluation.md#perplexity) | Text likelihood, reported as perplexity | Lower is better |
| [`BitsPerByteEvaluation`](adding/components/evaluation.md#perplexity) | Negative log likelihood per UTF-8 input byte | Lower is better |
| [`PerplexityComparisonEvaluation`](adding/components/evaluation.md#perplexity-comparison) | Whether the correct continuation has the lowest mean token NLL | Higher accuracy is better |

## Make an evaluation

Save this as `my_evaluation.py`. You provide the text;
`PerplexityEvaluation` handles tokenization, model scoring, and aggregation.

```python
from theseus.evaluation.base import PerplexityEvaluation, Evaluator
from theseus.experiments.models.gpt import PretrainGPT
from theseus.model.models import GPT
from theseus.registry import evaluation, job

@evaluation("my/text_ppl")
class TextEval(PerplexityEvaluation):
    texts = ["The cat sat on the mat.", "The sun rises in the east."]

    @property
    def name(self):
        return "my_text_ppl"

    def __len__(self):
        return len(self.texts)

    def get(self, indx):
        return self.texts[indx]
```

`get(indx)` returns one text and `__len__()` tells the evaluator how many are
available. `name` labels the metric. Replace `texts` with your held-out data;
these strings do not need the training-data preparation pipeline.

`@evaluation` registers the task when you import this file. Perplexity registry
keys must end in `_ppl`, which lets plots group them separately from accuracy
scores. For generated answers and a custom answer checker, use
[`RolloutEvaluation`](adding/components/evaluation.md#rollout) instead.

## Put it inline during training

Append to `my_evaluation.py`. `EVALUATION` selects the tasks for this trainer;
the trainer constructs them and supplies its live model state.

```python
@job("my/gpt/train-with-evaluation")
class GPTWithEvaluation(PretrainGPT):
    EVALUATION = [TextEval]  # Classes, not instances.
```

Use the root containing your [prepared training data](../Tutorials/running.md)
in place of `./results`. This small GPT run checks the integration; replace its
model and training settings with yours:

```python
from my_evaluation import GPTWithEvaluation
from theseus.quick import quick

with quick("./results") as q:
    q.build(GPTWithEvaluation, "train-gpt")
    q.config.architecture.n_embd = 128
    q.config.architecture.n_layers = 4
    q.config.architecture.block_size = 128
    q.config.training.batch_size = 8
    q.config.training.per_device_batch_size = 4
    q.config.training.tokens = 32_768
    q.config.training.evaluate = True
    q.config.eval.length = -1  # All examples; use a positive limit for a subset.
    q.config.logging.validation_interval = 8
    trainer = q.create()
    trainer()
```

`training.evaluate` enables evaluation in the shared validation hook, whose
cadence is set by `logging.validation_interval`. Evaluation also runs at the
final step. It is independent of `training.validation`, which controls validation
loss. `eval.length` limits examples per task; `-1` uses all of them.

Scores are logged under the task's `name`; use distinct names for multiple tasks.
To add another task, include its class in `EVALUATION`.

## Evaluate a saved checkpoint

`TextEval` defines the task; `Evaluator` supplies a runnable job that loads the
model, runs the tasks, and saves scores. Append this to `my_evaluation.py`:

```python
@job("my/gpt/evaluate")
class EvaluateGPT(Evaluator[GPT]):
    MODEL = GPT
    EVALUATION = [TextEval]
```

Replace `./results` and `train-gpt` with your root and run name:

```python
from my_evaluation import EvaluateGPT
from theseus.quick import quick

with quick("./results") as q:
    q.build(EvaluateGPT, "evaluate-gpt")
    q.find().spec(run="train-gpt").checkpoint().latest().branch()
    q.config.eval.length = -1
    evaluator = q.create()
    evaluator()
```

Build the evaluator before selecting the checkpoint to keep the evaluation job
class while loading the saved configuration. `MODEL` must match the checkpoint's
architecture. `branch()` restores its weights into a new result lineage without
continuing training. The job logs scores and writes `results.json` in its results
directory.

More: [evaluation strategies and custom scoring](adding/components/evaluation.md).
